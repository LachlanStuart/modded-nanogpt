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

from typing import Tuple

import torch

import tilelang
from tilelang import jit
from tilelang import language as T


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


if __name__ == "__main__":
    torch.manual_seed(0)

    batch, heads, seq_len, head_dim = 1, 2, 3, 4
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.double, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.double, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.double, requires_grad=True)
    pre = torch.randn(heads, heads, dtype=torch.double, requires_grad=True)
    post = torch.randn(heads, heads, dtype=torch.double, requires_grad=True)
    doc_ids = torch.tensor([[0, 0, 1]], dtype=torch.long)

    def _gradcheck_target(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = reference_attention(q, k, v, pre, post, doc_ids)
        return context

    inputs = (q, k, v, pre, post)
    passed = torch.autograd.gradcheck(
        _gradcheck_target,
        inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    print("gradcheck passed:", passed)
