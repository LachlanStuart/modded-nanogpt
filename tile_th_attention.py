from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch

LOG2E = 1.4426950408889634

try:
    import tilelang
    from tilelang import jit
    from tilelang import language as T
except ImportError:  # pragma: no cover - optional dependency
    tilelang = None
    jit = None
    T = None


def _require_tilelang():
    if tilelang is None:
        raise RuntimeError("TileLang is required but not installed. Please `pip install tilelang`.")  # pragma: no cover


def _dtype_to_tilelang(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"Unsupported dtype for TileLang kernel: {dtype}")


def _document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    if doc_ids.ndim == 1:
        doc_ids = doc_ids.unsqueeze(0)
    B, T = doc_ids.shape
    positions = torch.arange(T, device=doc_ids.device)
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
    mask = _document_mask(doc_ids)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    logits = torch.einsum("mh,bhqk->bhqk", pre, scores)
    logits = logits.masked_fill(~mask[:, None], float("-inf"))
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
    q_f = q.detach().requires_grad_(True)
    k_f = k.detach().requires_grad_(True)
    v_f = v.detach().requires_grad_(True)
    pre_f = pre.detach().requires_grad_(True)
    post_f = post.detach().requires_grad_(True)

    with torch.enable_grad():
        ctx, _ = _reference_forward(q_f, k_f, v_f, pre_f, post_f, doc_ids)
    grads = torch.autograd.grad(ctx, (q_f, k_f, v_f, pre_f, post_f), grad_out, retain_graph=False)
    return grads  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _compile_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    block_m: int,
    block_n: int,
    threads: int,
):
    _require_tilelang()
    dtype_str = _dtype_to_tilelang(dtype)
    accum_dtype = "float32"

    @jit(out_idx=[-2, -1])
    def th_fwd(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        threads=threads,
    ):
        sm_scale = LOG2E

        q_shape = [batch, heads, seq_len, head_dim]
        k_shape = [batch, heads, seq_len, head_dim]
        v_shape = [batch, heads, seq_len, head_dim]
        pre_shape = [heads, heads]
        post_shape = [heads, heads]
        doc_shape = [batch, seq_len]
        out_shape = [batch, heads, seq_len, head_dim]
        lse_shape = [batch, heads, seq_len]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype_str),  # type: ignore
            K: T.Tensor(k_shape, dtype_str),  # type: ignore
            V: T.Tensor(v_shape, dtype_str),  # type: ignore
            PreTH: T.Tensor(pre_shape, accum_dtype),  # type: ignore
            PostTH: T.Tensor(post_shape, accum_dtype),  # type: ignore
            DocIds: T.Tensor(doc_shape, "int32"),  # type: ignore
            Output: T.Tensor(out_shape, dtype_str),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                if by < heads:
                    head_out = by
                    q_offset = bx * block_m
                    valid_q = T.min(block_m, seq_len - q_offset)
                    num_k_tiles = T.ceildiv(seq_len, block_n)

                    Q_panel = T.alloc_shared([heads, block_m, head_dim], dtype_str)
                    q_doc = T.alloc_shared([block_m], "int32")
                    logits_buf = T.alloc_fragment([heads, block_m, block_n], accum_dtype)
                    scores_max = T.alloc_fragment([heads, block_m], accum_dtype)
                    scores_prev = T.alloc_fragment([heads, block_m], accum_dtype)
                    scores_scale = T.alloc_fragment([heads, block_m], accum_dtype)
                    scores_sum = T.alloc_fragment([heads, block_m], accum_dtype)
                    logsum = T.alloc_fragment([heads, block_m], accum_dtype)
                    acc_context = T.alloc_fragment([heads, block_m, head_dim], accum_dtype)
                    final_out = T.alloc_fragment([block_m, head_dim], accum_dtype)
                    K_tile = T.alloc_shared([block_n, head_dim], dtype_str)
                    V_tile = T.alloc_shared([block_n, head_dim], dtype_str)
                    dot_temp = T.alloc_fragment([block_m, block_n], accum_dtype)
                    doc_k = T.alloc_shared([block_n], "int32")
                    prob_tile = T.alloc_fragment([block_m, block_n], accum_dtype)

                    T.fill(scores_max, -2**30)
                    T.fill(logsum, 0)
                    T.fill(acc_context, 0)
                    T.fill(final_out, 0)

                    for h in range(heads):
                        for i, j in T.Parallel(block_m, head_dim):
                            q_idx = q_offset + i
                            cond = T.tir.all(i < valid_q, q_idx < seq_len)
                            Q_panel[h, i, j] = T.if_then_else(cond, Q[bz, h, q_idx, j], 0)
                    for i in T.Parallel(block_m):
                        q_idx = q_offset + i
                        cond = T.tir.all(i < valid_q, q_idx < seq_len)
                        q_doc[i] = T.if_then_else(cond, DocIds[bz, q_idx], -1)

                    for kb in T.Pipelined(num_k_tiles, num_stages=1):
                        k_offset = kb * block_n
                        valid_k = T.min(block_n, seq_len - k_offset)

                        for j in T.Parallel(block_n):
                            k_idx = k_offset + j
                            doc_k[j] = T.if_then_else(T.tir.all(j < valid_k, k_idx < seq_len), DocIds[bz, k_idx], -1)

                        for h_mid in range(heads):
                            T.fill(logits_buf[h_mid], 0)

                        for h_in in range(heads):
                            T.fill(K_tile, 0)
                            T.copy(K[bz, h_in, k_offset : k_offset + valid_k, :], K_tile[:valid_k, :])
                            T.fill(dot_temp, 0)
                            T.gemm(
                                Q_panel[h_in],
                                K_tile,
                                dot_temp,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                            for h_mid in range(heads):
                                coeff = PreTH[h_mid, h_in]
                                for i, j in T.Parallel(block_m, block_n):
                                    logits_buf[h_mid, i, j] += coeff * dot_temp[i, j]

                        for h_mid in range(heads):
                            for i, j in T.Parallel(block_m, block_n):
                                q_idx = q_offset + i
                                k_idx = k_offset + j
                                valid = T.tir.all(
                                    i < valid_q,
                                    j < valid_k,
                                    q_idx < seq_len,
                                    k_idx < seq_len,
                                    q_idx >= k_idx,
                                    q_doc[i] == doc_k[j],
                                )
                                logits_buf[h_mid, i, j] = T.if_then_else(
                                    valid, logits_buf[h_mid, i, j], -T.infinity(accum_dtype)
                                )

                            T.copy(scores_max[h_mid], scores_prev[h_mid])
                            T.reduce_max(logits_buf[h_mid], scores_max[h_mid], dim=1, clear=False)
                            for i in T.Parallel(block_m):
                                prev = scores_prev[h_mid, i]
                                new = scores_max[h_mid, i]
                                scale = T.exp2((prev - new) * sm_scale)
                                scores_scale[h_mid, i] = scale
                                logsum[h_mid, i] *= scale
                                for d in T.Parallel(head_dim):
                                    acc_context[h_mid, i, d] *= scale

                            for i, j in T.Parallel(block_m, block_n):
                                logits_buf[h_mid, i, j] = T.exp2(
                                    (logits_buf[h_mid, i, j] - scores_max[h_mid, i]) * sm_scale
                                )
                            T.reduce_sum(logits_buf[h_mid], scores_sum[h_mid], dim=1)
                            for i in T.Parallel(block_m):
                                logsum[h_mid, i] += scores_sum[h_mid, i]

                        T.fill(V_tile, 0)
                        T.copy(V[bz, head_out, k_offset : k_offset + valid_k, :], V_tile[:valid_k, :])

                        for h_mid in range(heads):
                            post_coeff = PostTH[head_out, h_mid]
                            for i, j in T.Parallel(block_m, block_n):
                                prob_tile[i, j] = logits_buf[h_mid, i, j] * post_coeff
                            T.gemm(
                                prob_tile,
                                V_tile,
                                acc_context[h_mid],
                                policy=T.GemmWarpPolicy.FullRow,
                                accumulate=True,
                            )

                    eps = T.float32(1e-6)
                    for h_mid in range(heads):
                        for i, d in T.Parallel(block_m, head_dim):
                            denom = T.max(logsum[h_mid, i], eps)
                            final_out[i, d] += acc_context[h_mid, i, d] / denom

                    for i, d in T.Parallel(block_m, head_dim):
                        q_idx = q_offset + i
                        cond = T.tir.all(i < valid_q, q_idx < seq_len)
                        Output[bz, head_out, q_idx, d] = T.if_then_else(
                            cond, final_out[i, d], Output[bz, head_out, q_idx, d]
                        )

                    for h_mid in range(heads):
                        for i in T.Parallel(block_m):
                            q_idx = q_offset + i
                            cond = T.tir.all(i < valid_q, q_idx < seq_len)
                            if cond:
                                Lse[bz, h_mid, q_idx] = T.log2(logsum[h_mid, i]) + scores_max[h_mid, i] * sm_scale

        return main

    return th_fwd()


def fast_talking_heads_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pre_th: torch.Tensor,
    post_th: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
    threads: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tilelang is None:
        q_t = q.transpose(1, 2).to(torch.float32)
        k_t = k.transpose(1, 2).to(torch.float32)
        v_t = v.transpose(1, 2).to(torch.float32)
        pre = pre_th.to(torch.float32)
        post = post_th.to(torch.float32)
        mask_ids = doc_ids.to(torch.int32)
        ctx, lse = _reference_forward(q_t, k_t, v_t, pre, post, mask_ids)
        return ctx.to(q.dtype).transpose(1, 2), lse

    batch, seq_len, heads, head_dim = q.shape
    assert batch == 1, "TileLang kernel currently assumes batch size 1."
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    pre = pre_th.to(torch.float32).contiguous()
    post = post_th.to(torch.float32).contiguous()
    docs = doc_ids.to(torch.int32).contiguous()
    out = torch.empty_like(q_t)
    lse = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=q.device)

    kernel = _compile_kernel(batch, seq_len, heads, head_dim, q.dtype, block_m, block_n, threads)
    out, lse = kernel(q_t, k_t, v_t, pre, post, docs, out, lse)
    return out.transpose(1, 2), lse


def fast_talking_heads_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pre_th: torch.Tensor,
    post_th: torch.Tensor,
    doc_ids: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
    threads: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del lse, block_m, block_n, threads  # gradient fallback uses PyTorch

    q_t = q.transpose(1, 2).to(torch.float32)
    k_t = k.transpose(1, 2).to(torch.float32)
    v_t = v.transpose(1, 2).to(torch.float32)
    pre = pre_th.to(torch.float32)
    post = post_th.to(torch.float32)
    grad_t = grad_out.transpose(1, 2).to(torch.float32)
    mask_ids = doc_ids.to(torch.int32)
    dq, dk, dv, dpre, dpost = _reference_backward(q_t, k_t, v_t, pre, post, mask_ids, grad_t)
    return (
        dq.to(q.dtype).transpose(1, 2),
        dk.to(k.dtype).transpose(1, 2),
        dv.to(v.dtype).transpose(1, 2),
        dpre,
        dpost,
    )
