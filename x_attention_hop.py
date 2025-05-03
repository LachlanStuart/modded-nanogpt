import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch._higher_order_ops.utils import (
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
)


def _construct_strides(
    sizes: Sequence[int],
    fill_order: Sequence[int],
) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    assert len(sizes) == len(fill_order), "Length of sizes must match the length of the fill order"
    strides = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def argsort(seq) -> list[int]:
    # preserve original order for equal strides
    getter = seq.__getitem__
    a_r = range(len(seq))
    return list(reversed(sorted(a_r, key=getter, reverse=True)))  # noqa: C413


def _permute_strides(out: torch.Tensor, query_strides: Tuple[int, ...]) -> torch.Tensor:
    """
    Create a new tensor with the same data and shape as the input,
    but with strides permuted based on the input tensor's stride order.

    Args:
        out (torch.Tensor): The output tensor of attention.
        query_strides (List[int]): The stride order of the input query tensor

    Returns:
        torch.Tensor: A new tensor with same shape and data as the input,
        but with strides permuted based on the query tensor's stride order.
    """
    fill_order = argsort(query_strides)
    # assert out.storage_offset() == 0, "Only support storage_offset == 0"
    out_strides = _construct_strides(out.shape, fill_order)
    new_out = out.new_empty(out.shape).as_strided(out.shape, out_strides)
    new_out.copy_(out)
    return new_out


def _math_attention_inner(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    mask_mod_other_buffers: Tuple = (),
) -> torch.Tensor:
    # from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32

    scores = (query @ key.transpose(-2, -1)).to(dtype=working_precision)

    from x_attention import _vmap_for_bhqkv

    mask_mod = block_mask[-1]
    mask_mod_in_dim_buffers = (None,) * len(mask_mod_other_buffers)
    mask_mod = _vmap_for_bhqkv(mask_mod, prefix=(), suffix=mask_mod_in_dim_buffers)

    # with TransformGetItemToIndex():
    scores = (scores * scale).to(working_precision)

    return scores


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager implementation

    This implementation uses vmap to vectorize the score_mod function over the batch, head, m, and n dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, m, or n dimensions. We need to apply vmap 4 times to vectorized over all 4 dimensions.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        other_buffers: Other buffers that are passed to the score_mod function
    """
    # broadcast query & key along head dim for GQA
    G = query.size(1) // key.size(1)
    value = torch.repeat_interleave(value, G, dim=1)
    key = torch.repeat_interleave(key, G, dim=1)

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    scores = _math_attention_inner(
        query,
        key,
        value,
        block_mask,
        scale,
        kernel_options,
        mask_mod_other_buffers,
    )

    # Set fully masked rows' sumexp to 0.0
    logsumexp = scores.logsumexp(dim=-1)
    masked_rows = torch.all(scores == -float("inf"), dim=-1)
    logsumexp = torch.where(masked_rows, -float("inf"), logsumexp)

    # scores = torch._safe_softmax(scores, dim=-1)
    scores = torch.softmax(scores, dim=-1)

    return scores.to(query.dtype) @ value, logsumexp / math.log(2)


def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = math_attention(
        query,
        key,
        value,
        block_mask,
        scale,
        kernel_options,
        mask_mod_other_buffers,
    )
    out = _permute_strides(out, query.stride())
    return out, lse


# ---------------------------- Autograd Implementation ----------------------------
class XAttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        block_mask: Tuple[Any, ...],
        scale: float,
        kernel_options: Dict[str, Any],
        mask_mod_other_buffers: Tuple[Any, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        any_buffer_requires_grad = any(
            buffer.requires_grad for buffer in mask_mod_other_buffers if isinstance(buffer, torch.Tensor)
        )
        assert not any_buffer_requires_grad, "Captured buffers from mask mod that require grad are not supported."
        ctx._mask_graph = block_mask[-1]
        ctx.scale = scale
        ctx.kernel_options = kernel_options
        # with torch._C._AutoDispatchBelowAutograd():
        out, logsumexp = sdpa_dense(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            mask_mod_other_buffers,
        )

        save_tensors_and_symints_for_backward(
            ctx,
            (
                query,
                key,
                value,
                out,
                logsumexp,
                *block_mask[:-1],
                *mask_mod_other_buffers,
            ),
        )
        return out, logsumexp

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor, grad_logsumexp: Tensor) -> Tuple[Optional[Tensor], ...]:  # type: ignore[override]
        fw_args = saved_tensors_and_symints(ctx)
        (
            query,
            key,
            value,
            out,
            logsumexp,
            query_lengths,
            kv_lengths,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            *mask_mod_other_buffers,
        ) = fw_args
        mask_graph = ctx._mask_graph
        scale = ctx.scale
        kernel_options = ctx.kernel_options
        # We have asserted that mask_mod_other_buffers do not require grad,
        # but score_mod_other_buffers can require grad.
        none_grads = [None] * 6
        (
            grad_query,
            grad_key,
            grad_value,
        ) = sdpa_dense_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            (
                query_lengths,
                kv_lengths,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                q_num_blocks,
                q_indices,
                full_q_num_blocks,
                full_q_indices,
                Q_BLOCK_SIZE,
                KV_BLOCK_SIZE,
                mask_graph,
            ),
            scale,
            kernel_options,
            mask_mod_other_buffers,
        )
        return grad_query, grad_key, grad_value, *none_grads


# ---------------------------- Backward HOP Implementation ----------------------------


def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    mask_mod_other_buffers: Tuple,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    # Get outputs before calling repeat interleave
    actual_grad_query = torch.empty_like(query)
    actual_grad_key = torch.empty_like(key)
    actual_grad_value = torch.empty_like(value)

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    G = query.size(1) // key.size(1)
    key = torch.repeat_interleave(key, G, dim=1)
    value = torch.repeat_interleave(value, G, dim=1)

    # We're undoing the log -> log2 change of base in the forwards
    logsumexp = logsumexp * math.log(2)
    # The backwards formula for the log -> log2 change of base in the forwards
    grad_logsumexp = grad_logsumexp / math.log(2)
    scores = _math_attention_inner(
        query,
        key,
        value,
        block_mask,
        scale,
        kernel_options,
        mask_mod_other_buffers,
    )
    masked_out_rows = logsumexp == -float("inf")
    softmax_scores = torch.exp(scores - logsumexp.unsqueeze(-1))
    softmax_scores = torch.where(masked_out_rows.unsqueeze(-1), 0, softmax_scores)

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out @ value.transpose(-2, -1)

    sum_scores = torch.sum(out * grad_out, -1, keepdim=True)
    grad_score_mod = softmax_scores * (grad_softmax_scores - sum_scores + grad_logsumexp.unsqueeze(-1))

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    mask_graph = block_mask[-1]
    from x_attention import _vmap_for_bhqkv

    # inputs are [score, b, h, q_idx, kv_idx, gradOut, ...]
    # score and gradOut are "fully" batched

    # with TransformGetItemToIndex():
    grad_scores = grad_score_mod  # FIXME: IDK?
    grad_scores = grad_scores * scale
    grad_scores = grad_scores.to(query.dtype)

    # mask_mod = _vmap_for_bhqkv(mask_graph, prefix=(), suffix=(None,) * len(mask_mod_other_buffers))
    mask_mod = mask_graph
    # mask_mod = torch.vmap(mask_mod, in_dims=(None, None, 0, None))
    # mask_mod = torch.vmap(mask_mod, in_dims=(None, 0, None, None))
    mask_mod = torch.vmap(mask_mod, in_dims=(0, None, None, None))
    # with TransformGetItemToIndex():
    mask_scores = mask_mod(b, h, m, n, *mask_mod_other_buffers)
    grad_scores = torch.where(mask_scores, grad_scores, torch.tensor(0, dtype=query.dtype))

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query

    # Reduce DK, DV along broadcasted heads.
    grad_key = grad_key.view(grad_key.size(0), -1, G, grad_key.size(-2), grad_key.size(-1))
    grad_value = grad_value.view(grad_value.size(0), -1, G, grad_value.size(-2), grad_value.size(-1))

    grad_key = torch.sum(grad_key, 2, keepdim=False)
    grad_value = torch.sum(grad_value, 2, keepdim=False)

    if Bq != Bkv:
        assert Bq > 1 and Bkv == 1, f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}"

        # Reduce DK, DV along broadcasted batches.
        grad_key = torch.sum(grad_key, 0, keepdim=True)
        grad_value = torch.sum(grad_value, 0, keepdim=True)

    actual_grad_query.copy_(grad_query)
    actual_grad_key.copy_(grad_key)
    actual_grad_value.copy_(grad_value)

    return (
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
    )
