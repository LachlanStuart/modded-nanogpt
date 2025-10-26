"""
Created with instruction:
@train_gpt.py is a single-file implementation of the training loop of a highly optimized GPT model. I'd like to start a parallel project, but multi-modal using VAR @https://arxiv.org/abs/2404.02905  @https://github.com/FoundationVision/VAR for images, and regular text tokens for text.
Make a new train_var.py. Copy the model from train_gpt, but make the training loop read images and tokenize them following VAR's method.
"""

from __future__ import annotations

from collections import defaultdict, deque
import os
import sys
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import atexit
from typing import Iterable, Iterator, Literal

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# FlexAttention
from torch.nn.attention.flex_attention import BlockMask, flex_attention

# Optional pillow for image IO
try:
    from PIL import Image
except Exception as _pil_err:  # pragma: no cover
    Image = None  # type: ignore


# -------------------------------------------------------------------------------------
# Custom operators: FP8 matmul (copied from train_gpt.py)
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.mul(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.mul(w_s).to(torch.float8_e4m3fn)
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
    return x @ w.t(), x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(
    g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float
) -> tuple[Tensor, Tensor]:
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


def _mm_backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(grad_out, x_f8, w_f8, x_s, w_s, grad_s)
    return grad_x, grad_w, None, None, None


def _mm_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


mm_op.register_autograd(_mm_backward, setup_context=_mm_setup_context)


# -------------------------------------------------------------------------------------
# Muon optimizer (copied from train_gpt.py)
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
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
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes
        ]
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
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )

            for base_i in range(len(params))[:: self.world_size]:
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
                update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


# -------------------------------------------------------------------------------------
# Model components (copied from train_gpt.py)
def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_fp8: bool = False,
        x_s: float = 1.0,
        w_s: float = 1.0,
        grad_s: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:  # type: ignore[override]
        std = 0.5 * (self.in_features**-0.5)
        bound = (3**0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):  # type: ignore[override]
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):  # type: ignore[override]
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, : x_BTHD.size(-3), None, :], self.sin[None, : x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):  # type: ignore[override]
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = (
            F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        y = flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):  # type: ignore[override]
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int, max_seq_len: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):  # type: ignore[override]
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

    def forward(self, input_seq: Tensor) -> list[Tensor | None]:  # type: ignore[override]
        ve = [emb(input_seq) for emb in self.embed]
        ve = [ve[0], ve[1], ve[2]] + [None] * (self.num_layers - 2 * self.num_embeddings) + [ve[0], ve[1], ve[2]]
        return ve


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

    return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, layer_idx, max_seq_len) for layer_idx in range(num_layers)])
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.lm_head = CastedLinear(
            model_dim, next_multiple_of_n(vocab_size, n=128), use_fp8=False, x_s=2.0, w_s=2.0**9, grad_s=2.0**19
        )
        self.lm_head.weight.detach().zero_()

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):  # type: ignore[override]
        assert input_seq.ndim == 1
        long_bm, short_bm = create_block_masks(input_seq, sliding_window_num_blocks)

        x = x0 = norm(self.embed(input_seq)[None])
        ve = self.value_embeds(input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]
        assert len(ve_enc) == self.num_encoder_layers and len(ve_dec) == self.num_decoder_layers

        skip_connections: list[Tensor] = []
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        assert len(block_masks) == self.num_encoder_layers
        for i, block in enumerate(self.blocks[: self.num_encoder_layers]):
            x = block(x, ve_enc[i], x0, block_masks[i])
            skip_connections.append(x)
        block_masks.reverse()
        assert len(block_masks) == self.num_decoder_layers
        for i, block in enumerate(self.blocks[self.num_encoder_layers :]):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = block(x, ve_dec[i], x0, block_masks[i])
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss


# -------------------------------------------------------------------------------------
# Multimodal VAR-style image tokenizer and data loader

TEXT_VOCAB_SIZE = 50257
DOCSEP_TOKEN_ID = 50256  # GPT-2 EOT used as document separator in block masks


@dataclass(frozen=True)
class ImageTokenizerConfig:
    image_scales: tuple[int, ...] = (16, 32, 64)
    grayscale: bool = True
    quantization_levels: int = 256  # number of discrete pixel values per pixel (grayscale)
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    img_token_base: int = 60000  # starting ID for all image tokens to avoid collision with text


@dataclass(frozen=True)
class VocabularyLayout:
    text_vocab_size: int
    docsep_id: int
    img_token_base: int
    img_pixel_base: int
    img_num_pixel_values: int
    img_scale_base: int
    img_scale_ids: dict[int, int]  # map scale size -> token id
    img_sos_id: int
    total_vocab_size: int


def build_vocabulary_layout(cfg: ImageTokenizerConfig) -> VocabularyLayout:
    img_pixel_base = cfg.img_token_base
    img_num_pixel_values = cfg.quantization_levels
    img_scale_base = img_pixel_base + img_num_pixel_values
    img_scale_ids = {s: (img_scale_base + i) for i, s in enumerate(cfg.image_scales)}
    img_sos_id = img_scale_base + len(cfg.image_scales)  # Start-of-image marker for image sequences
    total_vocab_size = max(TEXT_VOCAB_SIZE, img_sos_id + 1)
    return VocabularyLayout(
        text_vocab_size=TEXT_VOCAB_SIZE,
        docsep_id=DOCSEP_TOKEN_ID,
        img_token_base=cfg.img_token_base,
        img_pixel_base=img_pixel_base,
        img_num_pixel_values=img_num_pixel_values,
        img_scale_base=img_scale_base,
        img_scale_ids=img_scale_ids,
        img_sos_id=img_sos_id,
        total_vocab_size=total_vocab_size,
    )


def _ensure_pil():
    if Image is None:
        raise RuntimeError(
            "Pillow is required for image tokenization. Install with: pip install pillow"
        )


def _load_image_grayscale(path: Path, size: int) -> torch.Tensor:
    _ensure_pil()
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize((size, size), resample=Image.BICUBIC)
        arr = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
        arr = arr.view(size, size)
        return arr


def tokenize_image_var(path: Path, cfg: ImageTokenizerConfig, vocab: VocabularyLayout) -> torch.Tensor:
    # Build tokens as: [SOI, SCALE(s1), pixels..., SCALE(s2), pixels..., ..., DOCSEP]
    tokens: list[int] = [vocab.img_sos_id]
    for s in cfg.image_scales:
        scale_tok = vocab.img_scale_ids[s]
        tokens.append(scale_tok)
        px = _load_image_grayscale(path, s)
        # px values already 0..255, map to image pixel token range
        tokens.extend((vocab.img_pixel_base + px.flatten().to(torch.int32)).tolist())
    tokens.append(vocab.docsep_id)
    return torch.tensor(tokens, dtype=torch.int32)


def _list_image_files(root: Path, exts: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def _load_text_shard(file: Path) -> torch.Tensor:
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


class MultimodalStream:
    """Deterministic interleaving of text shards and image files to build a single token stream.

    At each request for more tokens, this class appends chunks from the next modality in a fixed cycle
    so that all ranks see the same global stream and consume disjoint slices deterministically.
    """

    def __init__(
        self,
        *,
        text_files_glob: str | None,
        image_dir: str | None,
        image_cfg: ImageTokenizerConfig,
        vocab: VocabularyLayout,
        image_to_text_ratio: float = 1.0,  # 1.0 => equal image/text blocks in cycle
    ) -> None:
        self.vocab = vocab
        self.image_cfg = image_cfg

        # Prepare text shard iterator
        self.text_files: list[Path] = []
        if text_files_glob:
            self.text_files = sorted(Path.cwd().glob(text_files_glob))
        self._text_it: Iterator[Path] | None = iter(self.text_files) if self.text_files else None
        self._text_tokens: torch.Tensor | None = _load_text_shard(next(self._text_it)) if self._text_it else None
        self._text_pos: int = 0

        # Prepare image file iterator
        self.image_files: list[Path] = _list_image_files(Path(image_dir), image_cfg.image_extensions) if image_dir else []
        self._image_it: Iterator[Path] | None = iter(self.image_files) if self.image_files else None

        # Modality schedule (deterministic cycle)
        if self._text_it and self._image_it:
            # compute small integers a:b that approximate ratio
            ratio = max(image_to_text_ratio, 1e-6)
            image_blocks = max(1, int(round(4 * ratio)))
            text_blocks = 4
            self._cycle: deque[Literal["img", "txt"]] = deque(["img"] * image_blocks + ["txt"] * text_blocks)
        elif self._image_it:
            self._cycle = deque(["img"])  # type: ignore[assignment]
        elif self._text_it:
            self._cycle = deque(["txt"])  # type: ignore[assignment]
        else:
            raise ValueError("No data sources provided (both text_files_glob and image_dir are None/empty)")

        self._buffer: list[int] = []  # accumulated token IDs

    def _append_text_chunk(self, min_tokens: int):
        # Append at least min_tokens from current shard(s)
        while min_tokens > 0:
            assert self._text_it is not None and self._text_tokens is not None
            shard = self._text_tokens
            remaining = shard.numel() - self._text_pos
            if remaining <= 0:
                try:
                    self._text_tokens = _load_text_shard(next(self._text_it))
                    self._text_pos = 0
                    continue
                except StopIteration:
                    # loop indefinitely over shards
                    self._text_it = iter(self.text_files)
                    self._text_tokens = _load_text_shard(next(self._text_it))
                    self._text_pos = 0
                    continue
            take = min(remaining, min_tokens)
            chunk = shard[self._text_pos : self._text_pos + take].to(torch.int32)
            self._buffer.extend(chunk.tolist())
            self._text_pos += take
            min_tokens -= take

    def _append_image_chunk(self):
        assert self._image_it is not None
        try:
            p = next(self._image_it)
        except StopIteration:
            self._image_it = iter(self.image_files)
            p = next(self._image_it)
        toks = tokenize_image_var(p, self.image_cfg, self.vocab)
        self._buffer.extend(toks.tolist())

    def ensure(self, total_needed: int):
        # Ensure buffer has at least total_needed tokens
        while len(self._buffer) < total_needed:
            modality = self._cycle[0]
            self._cycle.rotate(-1)
            if modality == "txt":
                # append roughly one image-doc worth of tokens from text to keep chunk sizes comparable
                # heuristic: fetch at least 4K tokens per text chunk
                self._append_text_chunk(4096)
            else:
                self._append_image_chunk()

    def take(self, n: int) -> torch.Tensor:
        self.ensure(n)
        out = torch.tensor(self._buffer[:n], dtype=torch.int32)
        # drop from buffer
        del self._buffer[:n]
        return out


def distributed_multimodal_data_generator(
    *,
    text_files_glob: str | None,
    image_dir: str | None,
    image_cfg: ImageTokenizerConfig,
    vocab: VocabularyLayout,
    image_to_text_ratio: float,
    batch_size: int,
    rank: int,
    world_size: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    stream = MultimodalStream(
        text_files_glob=text_files_glob,
        image_dir=image_dir,
        image_cfg=image_cfg,
        vocab=vocab,
        image_to_text_ratio=image_to_text_ratio,
    )
    pos = 0
    while True:
        # Build a big contiguous batch for all ranks + next token
        need = batch_size + 1
        buf = stream.take(need)
        # Slice this process's local segment
        local = buf[rank * local_batch_size : rank * local_batch_size + local_batch_size + 1]
        inputs = local[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = local[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets


# -------------------------------------------------------------------------------------
# Logging, hyperparameters, and main()


def print0(s: str, console: bool = True):
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
class VarHyperparameters:
    # text data
    text_train_files: str | None = None
    text_val_files: str | None = None
    val_tokens: int = 1_048_576  # 1M tokens default
    # image data
    image_train_dir: str | None = None
    image_val_dir: str | None = None
    image_scales: tuple[int, ...] = (16, 32, 64)
    grayscale: bool = True
    image_to_text_ratio: float = 1.0  # 1.0 => balanced cycle
    # optimization
    num_iterations: int = 2000
    cooldown_frac: float = 0.4
    # evaluation/logging
    val_loss_every: int = 200
    # implementation
    seq_len: int = 8192
    val_seq_len: int = 8192
    save_checkpoint: bool = False
    dev: bool = False


TEST_HPARAMS = VarHyperparameters(
    text_train_files="data/fineweb1B/fineweb_train_*.bin",
    text_val_files="data/fineweb1B/fineweb_val_*.bin",
    val_tokens=524_288,
    image_train_dir="data/images/train",
    image_val_dir="data/images/val",
    image_scales=(16, 32, 64),
    grayscale=True,
    image_to_text_ratio=1.0,
    num_iterations=1500,
    cooldown_frac=0.4,
    val_loss_every=200,
    seq_len=8192,
    val_seq_len=8192,
    save_checkpoint=False,
    dev=False,
)

DEV_HPARAMS = VarHyperparameters(
    text_train_files=None,
    text_val_files=None,
    val_tokens=131_072,
    image_train_dir="data/images/train",
    image_val_dir="data/images/val",
    image_scales=(16, 32),
    grayscale=True,
    image_to_text_ratio=1.0,
    num_iterations=50,
    cooldown_frac=0.4,
    val_loss_every=25,
    seq_len=1024,
    val_seq_len=1024,
    save_checkpoint=False,
    dev=True,
)


master_process: bool | None = None
logfile: str | None = None
if len(sys.argv) > 1:
    run_id = sys.argv[1]
else:
    run_id = str(uuid.uuid4())


def main(args: VarHyperparameters = TEST_HPARAMS):
    global master_process, logfile

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert torch.cuda.is_available(), "CUDA device required"
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    atexit.register(dist.destroy_process_group)
    dist.barrier()
    master_process = rank == 0

    if master_process and not args.dev:
        os.makedirs("logs", exist_ok=True)
        logfile_path = f"logs/{run_id}.txt"
        logfile = logfile_path
        print(logfile)

    atexit.register(log_mem)

    torch.random.manual_seed(0)
    torch.cuda.synchronize()
    print0("Init data")

    # Build vocabulary layout for image tokens
    img_cfg = ImageTokenizerConfig(
        image_scales=args.image_scales,
        grayscale=args.grayscale,
        quantization_levels=256,
        img_token_base=60000,
    )
    vocab = build_vocabulary_layout(img_cfg)

    # Data generators
    train_batch_size = world_size * args.seq_len
    train_loader = distributed_multimodal_data_generator(
        text_files_glob=args.text_train_files,
        image_dir=args.image_train_dir,
        image_cfg=img_cfg,
        vocab=vocab,
        image_to_text_ratio=args.image_to_text_ratio,
        batch_size=train_batch_size,
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.synchronize()
    print0("Init model")
    vocab_size = vocab.total_vocab_size
    model: nn.Module = GPT(
        vocab_size=vocab_size, num_layers=12, num_heads=6, model_dim=384, max_seq_len=max(args.seq_len, args.val_seq_len)
    ).cuda()
    model.bfloat16()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

    # count parameters
    n_params_by_dtype: dict[torch.dtype, int] = defaultdict(lambda: 0)
    for _, param in model.named_parameters():
        dist.broadcast(param.detach(), 0)
        n_params_by_dtype[param.dtype] += param.numel()
    for dt, n_params in n_params_by_dtype.items():
        print0(f"{dt}: {n_params / 1024 / 1024:.3f}Mi params")
    print0(f"total: {sum(n_params_by_dtype.values()) / 1024 / 1024:.3f}Mi params")

    torch.cuda.synchronize()
    print0("Init optimizers")
    hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]
    params_sets = [hidden_matrix_params, embed_params, scalar_params, head_params]
    assert all(set(a).isdisjoint(b) for a in params_sets for b in params_sets if a is not b)
    assert set().union(*params_sets) == set(model.parameters())

    # LR scaling heuristic (copied, adjusted only for seq_len reference)
    lr_mod = (args.seq_len / (48 * 1024) / 8) ** 0.5
    print(f"{lr_mod=}")
    adam_params = [
        dict(params=head_params, lr=0.008 * lr_mod),
        dict(params=embed_params, lr=0.6 * lr_mod),
        dict(params=scalar_params, lr=0.04 * lr_mod),
    ]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05 * lr_mod, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]

    def get_lr(step: int):
        t = 1 - step / args.num_iterations
        assert 1 >= t >= 0
        w = min(t / args.cooldown_frac, 1.0)
        return w * 1.0 + (1 - w) * 0.1

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    @lru_cache(1)
    def sw_num_blks(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    if not args.dev:
        model = torch.compile(model)  # type: ignore[assignment]

    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print0("Starting train loop")
    train_steps = args.num_iterations
    train_losses: list[float] = []
    val_losses: dict[int, float] = {}

    for step in range(train_steps + 1):
        last_step = step == train_steps
        if step == 10:
            training_time_ms = 0.0
            t0 = time.perf_counter()
        timed_steps = float("nan") if step <= 11 else (step - 10) + 1

        # Sliding window schedule
        window_size = next_multiple_of_n(1728 * step / train_steps, n=128)

        # Validation
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_multimodal_data_generator(
                text_files_glob=args.text_val_files,
                image_dir=args.image_val_dir,
                image_cfg=img_cfg,
                vocab=vocab,
                image_to_text_ratio=args.image_to_text_ratio,
                batch_size=val_batch_size,
                rank=rank,
                world_size=world_size,
            )
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_steps):
                    x, y = next(val_loader)
                    val_loss += model(x, y, sw_num_blks(window_size)).item()
            val_loss /= val_steps
            dist.all_reduce(torch.tensor(val_loss, device="cuda"), op=dist.ReduceOp.AVG)
            val_losses[step] = float(val_loss)
            print0(
                f"step:{step}/{train_steps} val_loss:{val_loss:.4f} step_avg:{training_time_ms / (timed_steps - 1):.2f}ms train_time:{training_time_ms / 1000:.0f}s",
                console=True,
            )
            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            break

        # Training step
        inputs, targets = next(train_loader)
        step_train_losses: list[float] = []
        for input_seq, target_seq in zip(inputs.split(args.seq_len), targets.split(args.seq_len)):
            loss = model(input_seq, target_seq, sw_num_blks(window_size))
            loss.backward()
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            step_train_losses.append(loss.detach().item())
            del loss
        train_losses.append(sum(step_train_losses) / len(step_train_losses))
        train_loss = sum(train_losses[-10:]) / len(train_losses[-10:])
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        del param

        # Muon momentum warmup
        frac = min(step / 300, 1)
        for group in optimizer2.param_groups:
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)

        if step < 20 or (step + 1) % 50 == 0:
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(
                f"step:{step + 1}/{train_steps} train_loss:{train_loss:.4f} step_avg:{approx_time / timed_steps:.2f}ms train_time:{approx_time / 1000:.0f}s vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                console=True,
            )

    print0(f"train_losses={train_losses}")
    print0(f"val_losses={val_losses}")
    if master_process and logfile is not None and run_id:
        print(logfile)


if __name__ == "__main__":
    main()


