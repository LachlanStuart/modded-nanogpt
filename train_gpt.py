from collections import defaultdict
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import atexit

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
# torch._inductor.config.coordinate_descent_tuning = True # turn this off for a faster compile time (but slightly slower run)
import triton
import triton.language as tl

try:
    from lovely_tensors import monkey_patch
    monkey_patch()
except ImportError:
    pass


# -----------------------------------------------------------------------------
#region  Custom operators : FP8 matmul by @YouJiacheng
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        # x_f8 = x.mul(x_s).to(torch.float8_e4m3fn)
        x_f8 = x.mul(x_s).to(torch.float8_e5m2)
        # w_f8 = w.mul(w_s).to(torch.float8_e4m3fn)
        w_f8 = w.mul(w_s).to(torch.float8_e5m2)
        out = torch._scaled_mm(
            x_f8,
            w_f8.t(),
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(1 / x_s, dtype=torch.float32),
            scale_b=x.new_tensor(1 / w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    # return x @ w.t(), x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
    return x @ w.t(), x.to(torch.float8_e5m2), w.to(torch.float8_e5m2)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(1 / x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(1 / w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(1 / grad_s, dtype=torch.float32)
        grad_f8 = grad.mul(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.t().contiguous().t(),
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.t().contiguous(),
            grad_f8.t().contiguous().t(),
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).t()
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)
#endregion
# -----------------------------------------------------------------------------
#region Muon optimizer
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven"t tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}
        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
        param_groups = [
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buffer = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
#endregion
# -----------------------------------------------------------------------------
#region PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8: bool = False, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale)
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int, max_seq_len: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int, num_embeddings: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.num_embeddings = num_embeddings
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for _ in range(num_embeddings)])

    def forward(self, input_seq: Tensor) -> list[Tensor | None]:
        ve = [emb(input_seq) for emb in self.embed]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (self.num_layers - 2 * self.num_embeddings) + [ve[0], ve[1], ve[2]]
        return ve
#endregion
# -----------------------------------------------------------------------------
#region The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

def create_block_masks(input_seq: Tensor, sliding_window_num_blocks: Tensor):
    BLOCK_SIZE = 128
    docs = (input_seq == 50256).cumsum(0)

    def document_causal(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[q_idx] == docs[kv_idx]
        return causal_mask & document_mask

    def dense_to_ordered(dense_mask: Tensor):
        num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
        indices = dense_mask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
        return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

    # manual block mask creation by @YouJiacheng
    assert len(input_seq) % BLOCK_SIZE == 0
    NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
    block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
    any_causal_bm = block_idx[:, None] >= block_idx
    all_causal_bm = block_idx[:, None] > block_idx
    docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
    docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
    any_document_bm = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
    all_document_bm = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
    any_bm = any_causal_bm & any_document_bm
    all_bm = all_causal_bm & all_document_bm
    partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(any_bm & ~all_bm)
    full_kv_num_blocks, full_kv_indices = dense_to_ordered(all_bm)
    def build_bm(sw_num_blocks: Tensor) -> BlockMask:
        return BlockMask.from_kv_blocks(
            torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(sw_num_blocks - full_kv_num_blocks, 1)),
            partial_kv_indices,
            torch.clamp_max(full_kv_num_blocks, sw_num_blocks - 1),
            full_kv_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=document_causal,
        )
    # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
    return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, layer_idx, max_seq_len) for layer_idx in range(num_layers)])
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        # self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), use_fp8=True, x_s=2.0, w_s=2.0**9, grad_s=2.0**19)
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), x_s=2.0, w_s=2.0**9, grad_s=2.0**19)
        self.lm_head.weight.detach().zero_() # @Grad62304977


    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        long_bm, short_bm = create_block_masks(input_seq, sliding_window_num_blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        ve = self.value_embeds(input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        assert len(ve_enc) == self.num_encoder_layers and len(ve_dec) == self.num_decoder_layers

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        assert len(block_masks) == self.num_encoder_layers
        for i, block in enumerate(self.blocks[:self.num_encoder_layers]):
            x = block(x, ve_enc[i], x0, block_masks[i])
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        block_masks.reverse()
        assert len(block_masks) == self.num_decoder_layers
        for i, block in enumerate(self.blocks[self.num_encoder_layers:]):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = block(x, ve_dec[i], x0, block_masks[i])

        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss

#endregion
# -----------------------------------------------------------------------------
#region MoEUT Triton kernels
# From https://github.com/RobertCsordas/moeut/blob/master/moeut/cvmm.py

@dataclass
class CVMMSel:
    raw_sel: torch.Tensor
    sel: torch.Tensor
    sel_index: torch.Tensor
    out_index: torch.Tensor | None = None
    reduction_weight: torch.Tensor | None = None

    def clone(self) -> 'CVMMSel':
        return CVMMSel(self.raw_sel, self.sel, self.sel_index, self.out_index, self.reduction_weight)


def cvmm_prepare_sel(sel: torch.Tensor, n_experts: int) -> CVMMSel:
    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()
    return CVMMSel(sel, ssel.view_as(sel), sel_index, None)


def dtype_to_type_id(dtype: torch.dtype):
    if dtype == torch.float32:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.bfloat16:
        return 2

    raise ValueError("Unknown dtype")


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'dtype_id', 'allow_tf32']
)
@triton.jit
def cvmm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bo, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_index, stride_sel, stride_out_index,
    out_index_is_none: tl.constexpr,
    dtype_id: tl.constexpr, allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m = first_pid_m + (pid % group_size_m)

    sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
    sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel)
    sel_all = tl.load(sel_ptr + stride_sel * ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M))

    for matrix_id in range(sel_first, sel_last + 1):
        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetics` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        remap_offs_am = tl.load(index_ptr + stride_index * offs_am)

        # Create offset pointers
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (remap_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + matrix_id * stride_bo + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.

            # Triton was unhappy with passing dtypes as vars.
            if dtype_id == 1:
                a = a.to(tl.float16)
                b = b.to(tl.float16)
            elif dtype_id == 2:
                a = a.to(tl.bfloat16)
                b = b.to(tl.bfloat16)

            accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk


        if dtype_id == 1:
            c = accumulator.to(tl.float16)
        elif dtype_id == 2:
            c = accumulator.to(tl.bfloat16)
        else:
            c = accumulator

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        if out_index_is_none:
            remap_offs_cm = remap_offs_am
        else:
            remap_offs_cm = tl.load(out_index_ptr + stride_out_index * offs_am)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = ((offs_cm[:, None] < M) & (sel_all[:, None] == matrix_id)) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 4}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'out_dtype_id', 'allow_tf32', 'dtype_id'], reset_to_zero = ['c_ptr']
)
@triton.jit
def cvmm_backward_kernel3(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_co, stride_cm, stride_cn,
    stride_index, stride_sel, stride_out_index,
    out_index_is_none: tl.constexpr,
    out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr, dtype_id: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, K_BLOCKS: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    k_block_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.

    a_ptrs_this = a_ptr + offs_am[:, None] * stride_am
    b_ptrs_this = b_ptr + offs_bn[None, :] * stride_bn

    # Kactual = end_i - start_i
    # Nblocks = (Kactual + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # WORK_PER_WORKER = (Nblocks + K_BLOCKS - 1) // K_BLOCKS
    # WORK_PER_WORKER = WORK_PER_WORKER if WORK_PER_WORKER > MIN_WORK_SIZE else MIN_WORK_SIZE


    # # Kloop_start = (Kactual + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # first_block_k = k_block_id * WORK_PER_WORKER
    # last_block_k = min((k_block_id+1) * WORK_PER_WORKER, Nblocks)

    block_start_index = k_block_id * BLOCK_SIZE_K * K_BLOCKS
    block_end_index = min(block_start_index + BLOCK_SIZE_K * K_BLOCKS, K) - 1

    first_mat = tl.load(sel_ptr + stride_sel * block_start_index)
    last_mat = tl.load(sel_ptr + stride_sel * block_end_index)


    for matrix_index in range(first_mat, last_mat + 1):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        start_i = block_start_index
        end_i = block_end_index + 1
        while start_i < end_i:
            middle = (start_i + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix < matrix_index:
                start_i = middle + 1
            else:
                end_i = middle


        # # Continue binary search: find the first matrix that is > matrix_index
        start_i2 = start_i
        end_i = block_end_index + 1
        while start_i2 < end_i:
            middle = (start_i2 + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix <= matrix_index:
                start_i2 = middle + 1
            else:
                end_i = middle

        end_i = start_i2

        count = end_i - start_i

        block_mem_indices_f_base = start_i  + tl.arange(0, BLOCK_SIZE_K)

        if count > 0:
            for k in range((count + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
                # block_mem_indices = (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
                block_mem_indices_f = block_mem_indices_f_base + k * BLOCK_SIZE_K
                block_mem_indices = block_mem_indices_f % K
                a_index = tl.load(index_ptr + stride_index * block_mem_indices)
                if out_index_is_none:
                    b_index = a_index
                else:
                    b_index = tl.load(out_index_ptr + stride_out_index * block_mem_indices)
                sel_ok = block_mem_indices_f < end_i

                a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk

                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs, mask=sel_ok[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=sel_ok[:, None], other=0.0)

                if dtype_id == 1:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)
                elif dtype_id == 2:
                    a = a.to(tl.bfloat16)
                    b = b.to(tl.bfloat16)

                # We accumulate along the K dimension.
                accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            if out_dtype_id == 1:
                c = accumulator.to(tl.float16)
            elif out_dtype_id == 2:
                c = accumulator.to(tl.bfloat16)
            else:
                c = accumulator

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_co * matrix_index + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            # tl.store(c_ptrs, c, mask=c_mask)
            tl.atomic_add(c_ptrs, c, mask=c_mask)


torch.library.define("mylib::cvmm_triton", "(Tensor x, Tensor sel_index, Tensor sel, Tensor keys, ScalarType out_dtype, Tensor out_index) -> Tensor")
@torch.library.impl("mylib::cvmm_triton", "default")
def cvmm_triton(
    x: torch.Tensor,
    sel_index: torch.Tensor,
    sel: torch.Tensor,
    keys: torch.Tensor,
    out_dtype: torch.dtype,
    out_index: torch.Tensor
):
    x = x.flatten(end_dim=-2)
    assert x.shape[-1] == keys.shape[1]

    sel_shape = sel.shape
    sel = sel.flatten()

    M = sel.shape[0]
    O, K, N = keys.shape
    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    # out = torch.zeros((M, N), device=x.device, dtype=out_dtype)
    # 1D launch kernel where each block gets its own program.

    # expected_m_per_matrix = int(math.ceil(M / O * 1.5))
    # expected_m_per_matrix = M

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    out_index_is_none = False
    if out_index.numel() == 1 and out_index == -1:
        out_index_is_none = True

    cvmm_kernel[grid](
        x, keys, out, sel_index, sel, out_index,
        M, N, K,
        x.stride(0), x.stride(1),
        keys.stride(0), keys.stride(1), keys.stride(2),
        out.stride(0), out.stride(1),
        sel_index.stride(0), sel.stride(0), 0 if out_index_is_none else out_index.stride(0),
        out_index_is_none=out_index_is_none,
        dtype_id = dtype_to_type_id(out.dtype), allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
    )

    return out.view(*sel_shape, N)


@torch.library.register_fake("mylib::cvmm_triton", cvmm_triton)
def cvmm_triton_abstract(x, sel_idx, sel, keys, out_dtype, out_index):
    sel_shape = sel.shape
    sel = sel.flatten()
    M = sel.shape[0]
    O, K, N = keys.shape
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    sel_shape = sel.shape
    return out.view(*sel_shape, N)


def cvmm_triton_backward(
    x: torch.Tensor,
    sel_index: torch.Tensor,
    sel: torch.Tensor,
    grads: torch.Tensor,
    n_experts: int,
    key_dtype: torch.dtype,
    op_dtype: torch.dtype,
    out_index: torch.Tensor
):
    x = x.flatten(end_dim=-2)
    x = x.transpose(0, 1)
    grads = grads.flatten(end_dim=-2)
    sel = sel.flatten()
    M, _ = x.shape
    K, N = grads.shape
    # FIX: out must be atomic_add'able, which excludes bfloat16. Cast to key_dtype after. Maybe this could be f16
    out = torch.zeros((n_experts, M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K'] * META['K_BLOCKS'])
    )
    out_index_is_none = False
    if out_index.numel() == 1 and out_index == -1:
        out_index_is_none = True

    cvmm_backward_kernel3[grid](
        x, grads, out, sel_index, sel, out_index,
        M, N, K,
        x.stride(0), x.stride(1),
        grads.stride(0), grads.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        sel_index.stride(0), sel.stride(0), 0 if out_index_is_none else out_index.stride(0),
        out_index_is_none=out_index_is_none,
        out_dtype_id=dtype_to_type_id(out.dtype),
        dtype_id=dtype_to_type_id(op_dtype),
        allow_tf32=False #torch.backends.cuda.matmul.allow_tf32
    )
    return out.to(dtype=key_dtype)


class CVMM(torch.autograd.Function):
    warned = False

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        keys: torch.Tensor,
        out_index: torch.Tensor | None = None,
        reduction_weight: torch.Tensor | None = None
    ):
        ctx.save_for_backward(x, keys, sel, sel_index, out_index, reduction_weight)

        # out_type = get_dtype()
        out_type = x.dtype
        # out_type = torch.float32
        if out_index is None:
            out_index = torch.tensor(-1).cuda()
        res = torch.ops.mylib.cvmm_triton(x, sel_index, sel, keys, out_type, out_index)

        if reduction_weight is not None:
            res = res.view(*reduction_weight.shape, res.shape[-1])
            res = (reduction_weight.unsqueeze(-2).type_as(res) @ res).squeeze(-2)

        ctx.op_type = out_type
        ctx.keys_type = keys.dtype
        ctx.dtype = out_type
        return res.type_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, keys, sel, sel_index, out_index, reduction_weight = ctx.saved_tensors
        keys_dt = keys

        # Backward for weight
        if reduction_weight is not None:
            # Project back the grads with he reduction weight, so the grad for the weight matrix is ok
            grad_output_w = reduction_weight.unsqueeze(-1).type_as(grad_output) @ grad_output.unsqueeze(-2)
        else:
            grad_output_w  = grad_output

        out_index_is_none = False
        if out_index is None:
            out_index_is_none = True
            out_index = torch.tensor(-1).cuda()

        grad_w = cvmm_triton_backward(
            x,
            sel_index,
            sel,
            grad_output_w,
            keys_dt.shape[0],
            ctx.keys_type,
            ctx.dtype,
            out_index=out_index
        )

        # Backward for input and reduction weight
        grad_w_off = None

        bw_index = sel_index if out_index_is_none else out_index
        bw_index_out = torch.tensor(-1).cuda()
        if reduction_weight is not None:
            # Hack the output indices to emulate repeats
            bw_index_out = bw_index
            bw_index = bw_index // reduction_weight.shape[-1]

        grad_x_full = torch.ops.mylib.cvmm_triton(
            grad_output,
            bw_index,
            sel,
            keys_dt.transpose(1,2),
            ctx.op_type,
            bw_index_out
        )

        grad_x_full = grad_x_full.view(*x.shape[:-1], -1, x.shape[-1])
        if reduction_weight is not None:
            # grad_x_full is the unscaled grad. For the input, we have to scale it, for the reduction wegiht,
            # we have to compute dot products with the input.
            grad_x = (reduction_weight.view(*grad_x_full.shape[:-1]).unsqueeze(-2).type_as(grad_x_full) @ grad_x_full).squeeze(-2)
            grad_w_off = (grad_x_full.type_as(reduction_weight) @ x.unsqueeze(-1).type_as(reduction_weight)).squeeze(-1).view_as(reduction_weight)
        elif grad_x_full.shape[-2] != 1:
            grad_x = grad_x_full.sum(-2)
        else:
            grad_x = grad_x_full

        grad_x = grad_x.view_as(x)

        return grad_x, None, None, grad_w, None, grad_w_off


def cvmm(x: torch.Tensor, sel: torch.Tensor | CVMMSel, keys: torch.Tensor):
    if not isinstance(sel, CVMMSel):
        sel = cvmm_prepare_sel(sel, keys.shape[0])
    assert x.dtype == keys.dtype, f"{x.dtype=} != {keys.dtype=}"

    return CVMM.apply(x, sel.sel_index, sel.sel, keys, sel.out_index, sel.reduction_weight)


def cvmm_prepare_sel2(sel: torch.Tensor, w: torch.Tensor | None = None) -> CVMMSel:
    # Has multiple selections for each batch element
    n_per_batch = sel.shape[-1]

    # indices = torch.arange(sel.nelement() // n_per_batch, device=sel.device, dtype=torch.int32)
    # indices = indices.repeat_interleave(n_per_batch).flatten()

    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()

    # in_index = indices[sel_index]
    in_index = sel_index // n_per_batch

    return CVMMSel(sel, ssel.view_as(sel), in_index, sel_index, w)

#endregion
# -----------------------------------------------------------------------------
#region MoEUT
def log_mean(x: torch.Tensor, dim: int = 0):
    return x.logsumexp(dim) - torch.log(torch.tensor(x.shape[dim]))


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return - (l * l.exp()).sum(-1)


def entropy_reg(sel: torch.Tensor, dim: int) -> torch.Tensor:
    sel = F.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return - entropy_l(sel).mean()


class SigmaMoE(torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, k: int,
                 v_dim: int | None = None):

        super().__init__()
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.k_vec_dim = self.k_dim
        self.num_heads = k

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))
        self.sel_expert = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        # nanogpt equivalence would be std_scale=(3 ** 0.5) * 0.5
        kbound = std_scale / (self.k_dim ** 0.5)
        self.keys.uniform_(-kbound, kbound)
        self.values.zero_()
        self.sel_expert.normal_()
        self.sel_expert.div_(self.sel_expert.norm(dim=1, keepdim=True))
        self.sel_expert.mul_(std_scale / (self.k_dim) ** 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Selection score calculation
        xnorm = norm(x)
        sel = F.linear(xnorm, self.sel_expert, None)
        sel_r = sel

        # Selection activation and topk
        sel = F.sigmoid(sel)

        sel_val, sel_index = sel.topk(self.num_heads, dim=-1, sorted=False)

        # Preprocess the selection indices. They will be needed for both layers and save some time
        sel_indices = cvmm_prepare_sel2(sel_index.int())

        # "Up-projection" layer for each head
        scores = cvmm(xnorm, sel_indices, self.keys)
        scores = F.relu(scores).square()

        # Down projection layer for each head
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = sel_val
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None

        out = cvmm(scores, sel_indices, self.values)

        res = out.view(*x.shape[:-1], self.v_dim)
        return x + res, sel_r


class SwitchHeadRoPE(torch.nn.Module):
    def __init__(self, state_size: int, num_heads: int, n_experts: int, max_seq_len: int,
                 head_dim: int | None = None, moe_k: int = 2
                 ):

        super().__init__()

        self.input_size = state_size
        self.output_size = state_size
        self.pe_size = self.input_size
        self.moe_k = moe_k
        self.n_experts = n_experts

        self.num_heads = num_heads
        self.head_dim = head_dim or (state_size // num_heads)
        self.rotary = Rotary(self.head_dim, max_seq_len)

        self.q = torch.nn.Linear(self.input_size, self.head_dim * self.num_heads, bias=False)
        self.k = torch.nn.Linear(self.input_size, self.head_dim * self.num_heads, bias=False)

        self.v = torch.nn.Parameter(torch.empty(self.num_heads * self.n_experts, self.input_size, self.head_dim))
        self.o = torch.nn.Parameter(torch.empty(self.num_heads * self.n_experts, self.head_dim, self.output_size))
        self.sel_v = torch.nn.Parameter(torch.empty(self.num_heads * self.n_experts, self.input_size))

        self.sel_o = torch.nn.Parameter(torch.empty(self.num_heads * self.n_experts, self.input_size))

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        bound = (3 ** 0.5) * 0.5 * (self.input_size ** -0.5)
        self.q.weight.uniform_(-bound, bound)
        self.k.weight.uniform_(-bound, bound)

        self.v.normal_(0, std_scale / self.input_size ** 0.5)
        self.o.zero_()

        self.sel_v.normal_()
        self.sel_v.div_(self.sel_v.norm(dim=1, keepdim=True))
        self.sel_v.mul_(std_scale / self.input_size ** 0.5)

        self.sel_o.normal_()
        self.sel_o.div_(self.sel_v.norm(dim=1, keepdim=True))
        self.sel_o.mul_(std_scale / self.input_size ** 0.5)


    def get_sel(self, t: torch.Tensor, w: torch.Tensor) -> tuple[CVMMSel, torch.Tensor]:
        sel = F.linear(t, w).float()
        sel = sel_raw = sel.view(*sel.shape[:-1], self.num_heads, -1)
        sel = sel.sigmoid()

        with torch.no_grad():
            _, sel_index = sel.topk(self.moe_k, dim=-1, sorted=False)
        sel_val = torch.gather(sel, -1, sel_index)

        sel_index_shifted = (torch.arange(self.num_heads, device=sel_index.device, dtype=sel_index.dtype) * self.n_experts).unsqueeze(-1) + sel_index
        return cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1), sel_val), sel_raw


    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, block_mask: BlockMask
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # *src: [batch_size, out_len, c]
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"

        xnorm = norm(x)
        q = self.q(xnorm)
        k = self.k(xnorm)

        v_sel, v_sel_r = self.get_sel(xnorm, self.sel_v)
        o_sel, o_sel_r = self.get_sel(xnorm, self.sel_o)

        v = cvmm(x, v_sel, self.v).transpose(-2, -3)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)

        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q).transpose(1,2), self.rotary(k).transpose(1,2)
        # if ve is not None:
        #     v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        # else: # skip mid-layers token value embeddings by @YouJiacheng
        #     v = self.lambdas[0] * v

        res = flex_attention(q, k, v, block_mask=block_mask, scale=0.12)
        res = res.transpose(1, 2)

        # The output selection indices are calculated from the current state and are also used for projecting "q".
        # But that projection needs to create multiple copies for the different heads. Here we already have the
        # heads, but we have to create copies for the top-k elements. We can calculate that from the reduction
        # weight. We also want to compute not only the weighted average between the top-k elements, but also
        # of the different heads. So reshape the reduction weight accordingly.
        o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
        o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
        out = cvmm(res, o_sel, self.o)

        return x + out, o_sel_r, v_sel_r


class MoEUTLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_expert_size: int, ff_n_experts: int,
                 att_n_experts: int, max_seq_len: int, head_dim: int | None = None, att_k: int = 2,
                 ff_k: int = 8):

        super().__init__()
        self.attention = SwitchHeadRoPE(
            d_model, num_heads, att_n_experts, max_seq_len=max_seq_len, head_dim=head_dim, moe_k=att_k)
        self.ffn = SigmaMoE(d_model, ff_n_experts, ff_expert_size, k=ff_k)

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, block_mask: BlockMask) -> torch.Tensor:
        x, o_sel_r, v_sel_r = self.attention(x, ve, block_mask)
        x, ffn_sel_r = self.ffn(x)
        return x, o_sel_r, v_sel_r, ffn_sel_r


class MoEUT(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, num_heads: int, ff_expert_size: int, ff_n_experts: int,
                 att_n_experts: int, max_seq_len: int, head_dim: int | None = None, att_k: int = 2,
                 ff_k: int = 8,
                 entropy_reg: float = 0.01, att_entropy_reg: float = 0.001,
                 group_size: int = 2):
        super().__init__()

        self.entropy_reg = entropy_reg
        self.att_entropy_reg = att_entropy_reg

        self.n_repeats = n_layers // group_size
        self.layers = torch.nn.ModuleList([
            MoEUTLayer(d_model, num_heads, ff_expert_size, ff_n_experts, att_n_experts,
                       max_seq_len, head_dim, att_k, ff_k)
            for _ in range(group_size)
        ])

        self.reset_parameters()

    def forward(self, x: torch.Tensor, block_mask: BlockMask) -> tuple[torch.Tensor, torch.Tensor]:
        # Run the model
        o_sels = defaultdict(list)
        v_sels = defaultdict(list)
        ffn_sels = defaultdict(list)
        for r in range(self.n_repeats):
            for li, layer in enumerate(self.layers):
                x, o_sel_r, v_sel_r, ffn_sel_r = layer(x, ve=None, block_mask=block_mask)
                o_sels[li].append(o_sel_r)
                v_sels[li].append(v_sel_r)
                ffn_sels[li].append(ffn_sel_r)

        ffn_reg_loss = torch.stack([
            entropy_reg(torch.stack(sel_hist, dim=-2).flatten(-3, -2), -2)
            for sel_hist in ffn_sels.values()
        ]).sum()
        att_reg_loss = torch.stack([
            entropy_reg(torch.stack(sel_hist, dim=-3).flatten(-4, -3), -3)
            for sel_hist in [*o_sels.values(), *v_sels.values()]
        ]).sum()

        reg_loss = self.entropy_reg * ffn_reg_loss + self.att_entropy_reg * att_reg_loss

        return x, reg_loss

    @torch.no_grad
    def reset_parameters(self):
        scale = (2 / (self.n_repeats * len(self.layers))) ** 0.5
        for layer in self.modules():
            if isinstance(layer, (SwitchHeadRoPE, SigmaMoE)):
                layer.reset_parameters(scale)



class MoEUTWrapper(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.moeut = MoEUT(
            d_model=model_dim,
            n_layers=num_layers,
            num_heads=num_heads,
            ff_expert_size=128,
            ff_n_experts=64,  # FIXME: Arbitrary decision
            att_n_experts=10, # !! From MoEUT paper, but they also used really weird numbers like d_head=41 so I don't trust their judgement
            max_seq_len=max_seq_len,
            head_dim=None,
            att_k=2,
            ff_k=8,
            entropy_reg=0.01,
            att_entropy_reg=0.001,
            group_size=2,
        )
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), x_s=2.0, w_s=2.0**9, grad_s=2.0**19)
        self.lm_head.weight.detach().zero_() # @Grad62304977


    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        long_bm, _ = create_block_masks(input_seq, sliding_window_num_blocks)

        x = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        x, reg_loss = self.moeut(x, long_bm)
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        if self.training:
            loss = loss + reg_loss
        return loss


#endregion
# -----------------------------------------------------------------------------
#region Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn"t helpful.
        pos += batch_size
        yield inputs, targets
#endregion
# -----------------------------------------------------------------------------
#region utils, hyperparams
def print0(s, console=True):
    if master_process:
        timestamp = time.strftime("%H:%M:%S.") + f"{time.time() % 1:.3f}"[2:]
        s = f"{timestamp}: {s}"
        if console:
            print(s)
        if logfile:
            with open(logfile, "a") as f:
                print(s, file=f)

def log_mem():
    print0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        console=True,
    )

@dataclass(frozen=True, kw_only=True)
class Hyperparameters:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files: str = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # optimization
    num_iterations: int = 1770 # number of iterations to run
    cooldown_frac: float = 0.4 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every: int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    # implementation
    seq_len: int = 48*1024 # FlexAttention sequence length
    val_seq_len: int = 4*64*1024 # FlexAttention sequence length for validation
    save_checkpoint: bool = False
    dev: bool = False

TEST_HPARAMS = Hyperparameters(
    train_files = "data/fineweb1B/fineweb_train_*.bin",
    val_files = "data/fineweb1B/fineweb_val_*.bin",
    val_tokens = 1048576,
    num_iterations = 1000, #770,
    cooldown_frac = 0.4,
    val_loss_every = 125,
    seq_len = 16*1024,
    val_seq_len = 4*16*1024,
    save_checkpoint = False,
    dev=False,
)
DEV_HPARAMS = Hyperparameters(
    train_files = "data/fineweb1B/fineweb_train_*.bin",
    val_files = "data/fineweb1B/fineweb_val_*.bin",
    val_tokens = 1024,
    num_iterations = 20,
    cooldown_frac = 0.4,
    val_loss_every = 125,
    seq_len = 512,
    val_seq_len = 512,
    save_checkpoint = False,
    dev=True,
)

#endregion
# -----------------------------------------------------------------------------
#region main()
master_process = None
logfile = None
def main(args = TEST_HPARAMS):
# def main(args = DEV_HPARAMS):
    global master_process, logfile
    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    atexit.register(dist.destroy_process_group)
    dist.barrier()
    master_process = (rank == 0) # this process will do logging, checkpointing etc.

    # begin logging
    if master_process and not args.dev:
        run_id = uuid.uuid4()
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)


    # begin by printing this file (the Python code)
    print0(code, console=False)
    print0("="*100, console=False)
    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}", console=False)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}", console=False)
    def nvidia_smi():
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi(), console=False)
    print0("="*100, console=False)
    atexit.register(log_mem)

    torch.random.manual_seed(0)
    torch.cuda.synchronize()
    print0("Init data")
    # load data
    train_batch_size = world_size * args.seq_len
    train_loader = distributed_data_generator(args.train_files, train_batch_size, rank, world_size)

    torch.cuda.synchronize()
    print0("Init model")
    # REF: model: nn.Module = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, max_seq_len=max(args.seq_len, args.val_seq_len)).cuda()
    model: nn.Module = MoEUTWrapper(vocab_size=50257, num_layers=12, num_heads=3, model_dim=384, max_seq_len=max(args.seq_len, args.val_seq_len)).cuda()
    model.bfloat16()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

    # count parameters
    n_params_by_dtype = defaultdict(lambda: 0)
    for name, param in model.named_parameters():
        dist.broadcast(param.detach(), 0)
        n_params_by_dtype[param.dtype] += param.numel()
    for dt, n_params in n_params_by_dtype.items():
        print0(f"{dt}: {n_params/1024/1024:.3f}Mi params")
    print0(f"total: {sum(n_params_by_dtype.values())/1024/1024:.3f}Mi params")


    torch.cuda.synchronize()
    print0("Init optimizers")
    # collect the parameters to optimize
    hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]
    params_sets = [hidden_matrix_params, embed_params, scalar_params, head_params]
    assert all(set(a).isdisjoint(b) for a in params_sets for b in params_sets if a is not b)

    # init the optimizer(s)
    lr_mod = (args.seq_len/Hyperparameters().seq_len/8) ** 0.5  # Correct LR based on difference in batch size vs original w/ 8 nodes
    print(f"{lr_mod=}")
    adam_params = [dict(params=head_params, lr=0.008*lr_mod), dict(params=embed_params, lr=0.6*lr_mod), dict(params=scalar_params, lr=0.04*lr_mod)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05*lr_mod, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        t = 1 - step / args.num_iterations # time remaining in training
        assert 1 >= t >= 0
        w = min(t / args.cooldown_frac, 1.0) # 1 -> 0
        return w * 1.0 + (1 - w) * 0.1
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
    @lru_cache(1)
    def sw_num_blks(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    if not args.dev:
        model: nn.Module = torch.compile(model) #, dynamic=False)

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    print0("Starting train loop")
    train_steps = args.num_iterations
    prof = None
    for step in range(train_steps + 1):
        last_step = (step == train_steps)
        # if step == 5:
        #     prof = profile(record_shapes=True, profile_memory=True, with_stack=True)
        #     prof.__enter__()
        #     prof.start()
        # if prof is not None:
        #     if step == 9:
        #         prof.__exit__(None, None, None)
        #         prof.export_chrome_trace("trace.json")
        #         prof = None
        #     else:
        #         prof.step()

        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float("nan") if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # Linearly increase the block-wise sliding window size over training 128 -> 1792:
        # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
        window_size = next_multiple_of_n(1728 * step / train_steps, n=128)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
            val_loss = 0
            with torch.no_grad():
                for i in range(val_steps):
                    x, y = next(val_loader)
                    val_loss += model(x, y, sw_num_blks(window_size))
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} step_avg:{training_time_ms/(timed_steps-1):.2f}ms train_time:{training_time_ms/1000:.0f}s", console=True)
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        train_losses = []
        for input_seq, target_seq in zip(inputs.split(args.seq_len), targets.split(args.seq_len)):
            loss = model(input_seq, target_seq, sw_num_blks(window_size))
            loss.backward()
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            train_losses.append(loss.item())
            del loss
        train_loss = sum(train_losses or [torch.nan]) / max(len(train_losses), 1)
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        del param
        # momentum warmup for Muon
        frac = min(step / 300, 1)
        for group in optimizer2.param_groups:
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)

        # logging
        approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.4f} step_avg:{approx_time/timed_steps:.2f}ms train_time:{approx_time/1000:.0f}s {torch.cuda.max_memory_allocated()=}", console=True)

    if master_process and logfile is not None:
        try:
            new_logfile = input("Name run? ")
        except KeyboardInterrupt:
            breakpoint()
        if new_logfile:
            old_logfile = logfile
            logfile = f"logs/{new_logfile}.txt"
            os.rename(old_logfile, logfile)
            print0(f"Renamed {old_logfile} -> {logfile}")
    else:
        print(logfile)
#endregion
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
