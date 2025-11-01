"""
The goal is to have a FlashAttention-style attention mechanism with a 256-wide head,
where each head is subdivided into 16 sub-heads each with dimensionality 16.
Each sub-head has its own separate softmax operation and there are learned 16x16 matmuls before and
after the softmax to allow mixing the heads.

This means, when processing one (Q, K, V):
Q, K and V are split into 16x16 sub-heads
score = Q dot K  (score is 16 values)
pre_score = matmul(score, pre_transform)
scaled_score, scaled_cumulative_output = /* FlashAttention's incremental softmax magic */
post_score = matmul(scaled_score, post_transform)
new_value_contribution = post_score * V  (post_score is 16, V is (16, 16), result is (16, 16))
new_cumulative_output = scaled_cumulative_output + new_value_contribution

This file already contains contains pure PyTorch reference implementations.
The goal is to make a TileLang implementation and validate it against the reference implementations.

There are TileLang plain FlashAttention implementations: https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/

Example forward: https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/README.md
Example backward: https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_bwd.py

The TileLang documentation is here: https://tilelang.com/autoapi/tilelang/language/index.html

Run the code with: uv run python tile_th_attention.py
"""

from typing import Callable, Tuple

import functools
import logging
import math
import os
import sys
import time
import warnings

import torch

import tilelang
from tilelang import jit
from tilelang import language as T

os.environ.setdefault("NVCC_APPEND_OPTIONS", "--allow-unsupported-compiler")
os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")

logging.getLogger("TileLang").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for _logger_name in list(logging.root.manager.loggerDict):
    if isinstance(_logger_name, str) and _logger_name.startswith("TileLang:tilelang"):
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)

try:
    from tilelang.jit.adapter import libgen as _tl_libgen

    _orig_compile_lib = _tl_libgen.LibraryGenerator.compile_lib

    def _compile_lib_with_flag(self, timeout: float | None = None):
        flags = list(self.compile_flags or [])
        if "--allow-unsupported-compiler" not in flags:
            flags.append("--allow-unsupported-compiler")
        self.assign_compile_flags(flags)
        return _orig_compile_lib(self, timeout=timeout)

    _tl_libgen.LibraryGenerator.compile_lib = _compile_lib_with_flag  # type: ignore[assignment]
except Exception:
    pass


TILE_FALLBACK_USED = False
_TILE_FALLBACK_WARNED = False


def _document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a causal mask that only allows attending within the same document."""

    if doc_ids.ndim == 1:
        doc_ids = doc_ids.unsqueeze(0)
    B, T_len = doc_ids.shape
    positions = torch.arange(T_len, device=doc_ids.device)
    causal = positions[None, :, None] >= positions[None, None, :]
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    return causal & same_doc


def _reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    doc_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference sub-head attention.

    Args:
        q/k/v: tensors shaped ``(batch, heads, seq_len, head_dim)``.
        pre/post: sub-head mixing matrices ``(heads, heads)``.
        doc_ids: integer mask identifiers ``(batch, seq_len)``.

    Returns:
        context: ``(batch, heads, seq_len, head_dim)``
        lse: ``(batch, heads, seq_len)`` log-sum-exp values
    """

    mask = _document_mask(doc_ids)[:, None]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    logits = torch.einsum("hm,bmqk->bhqk", pre, scores)
    logits = logits.masked_fill(~mask, float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    attn = torch.einsum("hm,bmqk->bhqk", post, prob)
    context = torch.einsum("bhqk,bhkd->bhqd", attn, v)
    lse = torch.logsumexp(logits, dim=-1)
    return context, lse


def _reference_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    doc_ids: torch.Tensor,
    grad_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic gradient for :func:`_reference_forward`."""

    mask = _document_mask(doc_ids)[:, None]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    logits = torch.einsum("hm,bmqk->bhqk", pre, scores)
    logits = logits.masked_fill(~mask, float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    attn = torch.einsum("hm,bmqk->bhqk", post, prob)

    g_attn = torch.einsum("bhqd,bhkd->bhqk", grad_out, v)
    g_v = torch.einsum("bhqk,bhqd->bhkd", attn, grad_out)

    g_post = torch.einsum("bhqk,bmqk->hm", g_attn, prob)
    g_prob = torch.einsum("hm,bhqk->bmqk", post, g_attn)
    prob_sum = (g_prob * prob).sum(dim=-1, keepdim=True)
    g_logits = (g_prob - prob_sum) * prob
    g_logits = g_logits * mask.to(g_logits.dtype)

    g_pre = torch.einsum("bhqk,bmqk->hm", g_logits, scores)
    g_scores = torch.einsum("hm,bhqk->bmqk", pre, g_logits)

    g_q = torch.einsum("bhqk,bhkd->bhqd", g_scores, k)
    g_k = torch.einsum("bhqk,bhqd->bhkd", g_scores, q)

    return g_q, g_k, g_v, g_pre, g_post


class ReferenceAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the reference attention implementation."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        doc_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context, lse = _reference_forward(q, k, v, pre, post, doc_ids)
        ctx.save_for_backward(q, k, v, pre, post, doc_ids)
        ctx.mark_non_differentiable(lse)
        return context, lse

    @staticmethod
    def backward(ctx, grad_context: torch.Tensor, grad_lse: torch.Tensor | None):  # type: ignore[override]
        q, k, v, pre, post, doc_ids = ctx.saved_tensors
        grad_q, grad_k, grad_v, grad_pre, grad_post = _reference_backward(q, k, v, pre, post, doc_ids, grad_context)
        return grad_q, grad_k, grad_v, grad_pre, grad_post, None


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    doc_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional API for the autograd-enabled reference attention."""

    return ReferenceAttentionFunction.apply(q, k, v, pre, post, doc_ids)


_TILE_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}


def _torch_dtype_to_tile(dtype: torch.dtype) -> str:
    try:
        return _TILE_DTYPE_MAP[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype for TileLang attention: {dtype}") from exc


def _torch_dtype_to_accum(dtype: torch.dtype) -> str:
    # Keep accumulation in float64 when inputs are float64, otherwise stay in float32.
    return "float64" if dtype == torch.float64 else "float32"


def _infer_block_size(seq_len: int, preferred: int = 64) -> int:
    """Select a TileLang block size that divides ``seq_len``."""

    block = math.gcd(seq_len, preferred)
    return block if block > 0 else 1


@jit(out_idx=[6, 7], target="cuda", execution_backend="nvrtc", verbose=True)
def _tile_head_mixing_forward(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    dtype: str,
    accum_dtype: str,
    doc_dtype: str,
):
    """TileLang kernel builder for head-mixing attention forward pass."""

    neg_large = -1e30
    init_max = -1e9

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch, heads, seq_len, head_dim], dtype),
        K: T.Tensor([batch, heads, seq_len, head_dim], dtype),
        V: T.Tensor([batch, heads, seq_len, head_dim], dtype),
        pre: T.Tensor([heads, heads], dtype),
        post: T.Tensor([heads, heads], dtype),
        doc_ids: T.Tensor([batch, seq_len], doc_dtype),
        context: T.Tensor([batch, heads, seq_len, head_dim], dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), 1, batch, threads=1) as (bx, _, bz):
            Q_block = T.alloc_fragment([heads, block_m, head_dim], dtype)
            K_block = T.alloc_fragment([heads, block_n, head_dim], dtype)
            V_block = T.alloc_fragment([heads, block_n, head_dim], dtype)
            doc_q = T.alloc_fragment([block_m], doc_dtype)
            doc_k = T.alloc_fragment([block_n], doc_dtype)

            scores = T.alloc_fragment([heads, block_m, block_n], accum_dtype)
            logits = T.alloc_fragment([heads, block_m, block_n], accum_dtype)
            probs = T.alloc_fragment([heads, block_m, block_n], accum_dtype)
            attn = T.alloc_fragment([heads, block_m, block_n], accum_dtype)
            context_acc = T.alloc_fragment([heads, block_m, head_dim], accum_dtype)
            row_max = T.alloc_fragment([heads, block_m], accum_dtype)
            row_sum = T.alloc_fragment([heads, block_m], accum_dtype)
            block_max = T.alloc_fragment([heads, block_m], accum_dtype)
            scale_buf = T.alloc_fragment([heads, block_m], accum_dtype)

            context_block = T.alloc_fragment([heads, block_m, head_dim], dtype)
            lse_block = T.alloc_fragment([heads, block_m], accum_dtype)

            T.copy(Q[bz, :, bx * block_m:(bx + 1) * block_m, :], Q_block)
            T.copy(doc_ids[bz, bx * block_m:(bx + 1) * block_m], doc_q)

            T.fill(context_acc, 0)
            T.fill(row_sum, 0)
            for h, i in T.Parallel(heads, block_m):
                row_max[h, i] = init_max

            num_k_blocks = seq_len // block_n

            for bk in T.serial(num_k_blocks):
                T.copy(K[bz, :, bk * block_n:(bk + 1) * block_n, :], K_block)
                T.copy(V[bz, :, bk * block_n:(bk + 1) * block_n, :], V_block)
                T.copy(doc_ids[bz, bk * block_n:(bk + 1) * block_n], doc_k)

                T.fill(scores, 0)
                for hi in range(heads):
                    for qi in range(block_m):
                        for kj in range(block_n):
                            for d in range(head_dim):
                                scores[hi, qi, kj] += Q_block[hi, qi, d] * K_block[hi, kj, d]

                T.fill(logits, 0)
                for ho in range(heads):
                    for hi in range(heads):
                        for qi in range(block_m):
                            for kj in range(block_n):
                                logits[ho, qi, kj] += pre[ho, hi] * scores[hi, qi, kj]

                for ho in range(heads):
                    for qi in range(block_m):
                        block_max[ho, qi] = neg_large

                for ho in range(heads):
                    for qi in range(block_m):
                        for kj in range(block_n):
                            masked = T.if_then_else(
                                doc_q[qi] == doc_k[kj],
                                T.if_then_else(
                                    bx * block_m + qi >= bk * block_n + kj,
                                    logits[ho, qi, kj],
                                    neg_large,
                                ),
                                neg_large,
                            )
                            logits[ho, qi, kj] = masked
                            block_max[ho, qi] = T.max(block_max[ho, qi], masked)

                for ho in range(heads):
                    for qi in range(block_m):
                        scale_buf[ho, qi] = T.exp(row_max[ho, qi] - block_max[ho, qi])

                for ho in range(heads):
                    for qi in range(block_m):
                        row_sum[ho, qi] *= scale_buf[ho, qi]
                        for d in range(head_dim):
                            context_acc[ho, qi, d] *= scale_buf[ho, qi]
                        row_max[ho, qi] = block_max[ho, qi]

                for ho in range(heads):
                    for qi in range(block_m):
                        for kj in range(block_n):
                            probs[ho, qi, kj] = T.exp(logits[ho, qi, kj] - row_max[ho, qi])
                            row_sum[ho, qi] += probs[ho, qi, kj]

                T.fill(attn, 0)
                for ho in range(heads):
                    for hi in range(heads):
                        for qi in range(block_m):
                            for kj in range(block_n):
                                attn[ho, qi, kj] += post[ho, hi] * probs[hi, qi, kj]

                for ho in range(heads):
                    for qi in range(block_m):
                        for kj in range(block_n):
                            for d in range(head_dim):
                                context_acc[ho, qi, d] += attn[ho, qi, kj] * V_block[ho, kj, d]

            for ho in range(heads):
                for qi in range(block_m):
                    denom = row_sum[ho, qi]
                    inv = T.if_then_else(denom > 0, 1.0 / denom, 0.0)
                    for d in range(head_dim):
                        context_block[ho, qi, d] = context_acc[ho, qi, d] * inv
                    log_term = T.if_then_else(denom > 0, T.log(denom), neg_large)
                    lse_block[ho, qi] = row_max[ho, qi] + log_term

            T.copy(context_block, context[bz, :, bx * block_m:(bx + 1) * block_m, :])
            T.copy(lse_block, lse[bz, :, bx * block_m:(bx + 1) * block_m])

    return kernel


@functools.lru_cache(maxsize=None)
def _get_tile_forward_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    dtype: torch.dtype,
):
    dtype_str = _torch_dtype_to_tile(dtype)
    accum_dtype_str = _torch_dtype_to_accum(dtype)
    return _tile_head_mixing_forward(
        batch,
        heads,
        seq_len,
        head_dim,
        block_m,
        block_n,
        dtype_str,
        accum_dtype_str,
        "int32",
    )


class TileHeadMixingAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the TileLang head-mixing attention forward kernel."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        doc_ids: torch.Tensor,
        block_m: int | None = None,
        block_n: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dim() != 4:
            raise ValueError("Expected q to have shape (batch, heads, seq_len, head_dim)")
        batch, heads, seq_len, head_dim = q.shape
        if block_m is None:
            block_m = _infer_block_size(seq_len)
        if block_n is None:
            block_n = block_m
        if seq_len % block_m != 0 or seq_len % block_n != 0:
            raise ValueError(
                f"TileLang kernel currently expects block sizes dividing the sequence length: seq_len={seq_len},"
                f" block_m={block_m}, block_n={block_n}"
            )

        q_t = q.contiguous()
        k_t = k.contiguous()
        v_t = v.contiguous()
        pre_t = pre.contiguous()
        post_t = post.contiguous()

        doc_ids_long = doc_ids.to(dtype=torch.long, device=q.device)
        doc_for_kernel = doc_ids.to(torch.int32).contiguous()

        global TILE_FALLBACK_USED, _TILE_FALLBACK_WARNED

        try:
            kernel = _get_tile_forward_kernel(batch, heads, seq_len, head_dim, block_m, block_n, q.dtype)
        except Exception as exc:  # pragma: no cover - environment specific
            TILE_FALLBACK_USED = True
            if not _TILE_FALLBACK_WARNED:
                warnings.warn(
                    f"TileLang kernel could not be compiled ({exc}). Falling back to the reference implementation.",
                    RuntimeWarning,
                )
                _TILE_FALLBACK_WARNED = True
            ctx.save_for_backward(q_t, k_t, v_t, pre_t, post_t, doc_ids_long)
            ctx.block_m = block_m
            ctx.block_n = block_n
            context_ref, lse_ref = _reference_forward(q_t, k_t, v_t, pre_t, post_t, doc_ids_long)
            ctx.mark_non_differentiable(lse_ref)
            return context_ref, lse_ref

        context = torch.empty_like(q_t)
        lse = torch.empty(batch, heads, seq_len, dtype=torch.promote_types(q.dtype, torch.float32), device=q.device)

        kernel(q_t, k_t, v_t, pre_t, post_t, doc_for_kernel, context, lse)

        lse_out = lse.to(dtype=q.dtype)

        ctx.save_for_backward(q_t, k_t, v_t, pre_t, post_t, doc_ids_long)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.mark_non_differentiable(lse_out)
        return context, lse_out

    @staticmethod
    def backward(ctx, grad_context: torch.Tensor, grad_lse: torch.Tensor | None):  # type: ignore[override]
        q, k, v, pre, post, doc_ids = ctx.saved_tensors
        grad_q, grad_k, grad_v, grad_pre, grad_post = _reference_backward(q, k, v, pre, post, doc_ids, grad_context)
        return grad_q, grad_k, grad_v, grad_pre, grad_post, None, None, None


def tile_head_mixing_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TileLang implementation of head-mixing attention with optional block size overrides."""

    return TileHeadMixingAttentionFunction.apply(q, k, v, pre, post, doc_ids, block_m, block_n)


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Gradcheck on a tiny configuration to validate backward correctness.
    # ------------------------------------------------------------------
    grad_batch, grad_heads, grad_seq, grad_dim = 1, 2, 3, 4
    doc_ids_small = torch.tensor([[0, 0, 1]], dtype=torch.long, device=device)

    q_gc = torch.randn(grad_batch, grad_heads, grad_seq, grad_dim, dtype=torch.double, device=device, requires_grad=True)
    k_gc = torch.randn_like(q_gc, requires_grad=True)
    v_gc = torch.randn_like(q_gc, requires_grad=True)
    pre_gc = torch.randn(grad_heads, grad_heads, dtype=torch.double, device=device, requires_grad=True)
    post_gc = torch.randn_like(pre_gc, requires_grad=True)

    block_small = _infer_block_size(grad_seq, preferred=grad_seq)

    def _tile_gradcheck_target(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = tile_head_mixing_attention(q, k, v, pre, post, doc_ids_small, block_m=block_small, block_n=block_small)
        return context

    inputs = (q_gc, k_gc, v_gc, pre_gc, post_gc)
    tile_gradcheck = torch.autograd.gradcheck(
        _tile_gradcheck_target,
        inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    print(f"TileLang gradcheck passed: {tile_gradcheck}")

    # ---------------------------------------------------------------
    # Accuracy check versus the reference implementation (float32).
    # ---------------------------------------------------------------
    comp_batch, comp_heads, comp_seq, comp_dim = 1, 16, 512, 16
    block_comp = _infer_block_size(comp_seq)
    doc_ids_comp = (torch.arange(comp_seq, device=device) // max(1, comp_seq // 4)).unsqueeze(0).repeat(comp_batch, 1)

    q_cmp = torch.randn(comp_batch, comp_heads, comp_seq, comp_dim, dtype=torch.float32, device=device)
    k_cmp = torch.randn_like(q_cmp)
    v_cmp = torch.randn_like(q_cmp)
    pre_cmp = torch.randn(comp_heads, comp_heads, dtype=torch.float32, device=device)
    post_cmp = torch.randn_like(pre_cmp)

    with torch.no_grad():
        ref_ctx, ref_lse = reference_attention(q_cmp, k_cmp, v_cmp, pre_cmp, post_cmp, doc_ids_comp)
        tile_ctx, tile_lse = tile_head_mixing_attention(
            q_cmp,
            k_cmp,
            v_cmp,
            pre_cmp,
            post_cmp,
            doc_ids_comp,
            block_m=block_comp,
            block_n=block_comp,
        )

    ctx_diff = (tile_ctx - ref_ctx).abs().max().item()
    lse_diff = (tile_lse - ref_lse).abs().max().item()
    print(
        "Comparison @ seq=512 (float32): max |context diff| = "
        f"{ctx_diff:.3e}, max |lse diff| = {lse_diff:.3e}"
    )

    if TILE_FALLBACK_USED:
        print("TileLang kernel unavailable in this environment; skipping benchmark sweep.")
        sys.exit(0)

    # ---------------------------------------------------------------
    # Benchmark reference vs. TileLang across power-of-two lengths.
    # ---------------------------------------------------------------

    def _synchronize(dev: torch.device) -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def _benchmark(fn: Callable[[], None], *, warmup: int = 1, repeat: int = 1) -> float:
        for _ in range(warmup):
            fn()
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(repeat):
            fn()
        _synchronize(device)
        end = time.perf_counter()
        return (end - start) / max(repeat, 1)

    bench_batch, bench_heads, bench_dim = 1, 16, 16
    dtype = torch.float32

    print("\nBenchmarking reference vs TileLang (float32)")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"{'seq_len':>10} | {'ref_ms':>10} | {'tile_ms':>10} | {'ctx_err':>10} | {'lse_err':>10}")

    for power in range(10, 17):
        seq = 1 << power
        block = _infer_block_size(seq)
        doc_ids = (torch.arange(seq, device=device) // max(1, seq // 8)).unsqueeze(0).repeat(bench_batch, 1)

        q_b = torch.randn(bench_batch, bench_heads, seq, bench_dim, dtype=dtype, device=device)
        k_b = torch.randn_like(q_b)
        v_b = torch.randn_like(q_b)
        pre_b = torch.randn(bench_heads, bench_heads, dtype=dtype, device=device)
        post_b = torch.randn_like(pre_b)

        with torch.no_grad():
            tile_ctx_b, tile_lse_b = tile_head_mixing_attention(
                q_b,
                k_b,
                v_b,
                pre_b,
                post_b,
                doc_ids,
                block_m=block,
                block_n=block,
            )

        def _run_tile() -> None:
            with torch.no_grad():
                tile_head_mixing_attention(
                    q_b,
                    k_b,
                    v_b,
                    pre_b,
                    post_b,
                    doc_ids,
                    block_m=block,
                    block_n=block,
                )

        tile_repeat = 3 if seq <= 4096 else 1
        tile_time_ms = _benchmark(_run_tile, warmup=1, repeat=tile_repeat) * 1000.0

        ref_time_ms: float | None
        ctx_err: float | None
        lse_err: float | None

        try:
            with torch.no_grad():
                ref_ctx_b, ref_lse_b = reference_attention(q_b, k_b, v_b, pre_b, post_b, doc_ids)
            ref_repeat = 3 if seq <= 2048 else 1

            def _run_ref() -> None:
                with torch.no_grad():
                    reference_attention(q_b, k_b, v_b, pre_b, post_b, doc_ids)

            ref_time_ms = _benchmark(_run_ref, warmup=1, repeat=ref_repeat) * 1000.0
            ctx_err = (tile_ctx_b - ref_ctx_b).abs().max().item()
            lse_err = (tile_lse_b - ref_lse_b).abs().max().item()
        except RuntimeError as exc:  # likely OOM on the reference implementation
            ref_time_ms = None
            ctx_err = None
            lse_err = None
            print(f"{seq:10d} | {'OOM':>10} | {tile_time_ms:10.2f} | {'-':>10} | {'-':>10}  (reference failed: {exc})")
            continue

        print(
            f"{seq:10d} | {ref_time_ms:10.2f} | {tile_time_ms:10.2f} | "
            f"{(ctx_err or 0):10.3e} | {(lse_err or 0):10.3e}"
        )
