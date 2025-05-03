import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_mutation,
    _maybe_reenter_make_fx,
    autograd_not_implemented,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    UnsupportedAliasMutationException,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule


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
    from torch._inductor.ir import get_fill_order

    fill_order = get_fill_order(query_strides)
    assert out.storage_offset() == 0, "Only support storage_offset == 0"
    out_strides = _construct_strides(out.shape, fill_order)
    new_out = out.new_empty(out.shape).as_strided(out.shape, out_strides)
    new_out.copy_(out)
    return new_out


def _math_attention_inner(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = (),
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32

    scores = (query @ key.transpose(-2, -1)).to(dtype=working_precision)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    from .x_attention import _vmap_for_bhqkv

    # first input is score
    score_mod = _vmap_for_bhqkv(score_mod, prefix=(0,), suffix=captured_buffers_in_dim)

    mask_mod = block_mask[-1]
    mask_mod_in_dim_buffers = (None,) * len(mask_mod_other_buffers)
    mask_mod = _vmap_for_bhqkv(mask_mod, prefix=(), suffix=mask_mod_in_dim_buffers)

    with TransformGetItemToIndex():
        scores = (scores * scale).to(working_precision)
        post_mod_scores = torch.where(
            mask_mod(b, h, m, n, *mask_mod_other_buffers),
            score_mod(scores, b, h, m, n, *score_mod_other_buffers),
            torch.tensor(-float("inf"), dtype=working_precision, device=scores.device),
        )

    return scores, post_mod_scores


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = (),
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
        score_mod: The score_mod function
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

    _, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )

    # Set fully masked rows' sumexp to 0.0
    logsumexp = post_mod_scores.logsumexp(dim=-1)
    masked_rows = torch.all(post_mod_scores == -float("inf"), dim=-1)
    logsumexp = torch.where(masked_rows, -float("inf"), logsumexp)

    post_mod_scores = torch._safe_softmax(post_mod_scores, dim=-1)

    return post_mod_scores.to(query.dtype) @ value, logsumexp / math.log(2)


def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = (),
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = math_attention(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
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
        fw_graph: Callable,
        joint_graph: Callable,
        block_mask: Tuple[Any, ...],
        scale: float,
        kernel_options: Dict[str, Any],
        mask_mod_other_buffers: Tuple[Any, ...],
        *score_mod_other_buffers: Tuple[Any, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        any_buffer_requires_grad = any(
            buffer.requires_grad for buffer in mask_mod_other_buffers if isinstance(buffer, torch.Tensor)
        )
        assert not any_buffer_requires_grad, "Captured buffers from mask mod that require grad are not supported."
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        ctx._mask_graph = block_mask[-1]
        ctx.scale = scale
        ctx.kernel_options = kernel_options
        ctx._score_mod_other_buffers_len = len(score_mod_other_buffers)
        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp = sdpa_dense(
                query,
                key,
                value,
                fw_graph,
                block_mask,
                scale,
                kernel_options,
                score_mod_other_buffers,
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
                *score_mod_other_buffers,
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
            *other_buffers,
        ) = fw_args
        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        mask_graph = ctx._mask_graph
        scale = ctx.scale
        kernel_options = ctx.kernel_options
        score_mod_other_buffers = tuple(other_buffers[: ctx._score_mod_other_buffers_len])
        mask_mod_other_buffers = tuple(other_buffers[ctx._score_mod_other_buffers_len :])
        # We have asserted that mask_mod_other_buffers do not require grad,
        # but score_mod_other_buffers can require grad.
        none_grads = [None] * 6
        (
            grad_query,
            grad_key,
            grad_value,
            grad_score_mod_captured,
        ) = sdpa_dense_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
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
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
        return grad_query, grad_key, grad_value, *none_grads, *grad_score_mod_captured


# ---------------------------- Backward HOP Implementation ----------------------------


def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple,
    mask_mod_other_buffers: Tuple,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], ...]]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    # Get outputs before calling repeat interleave
    actual_grad_query = torch.empty_like(query)
    actual_grad_key = torch.empty_like(key)
    actual_grad_value = torch.empty_like(value)

    def _maybe_new_buffer(
        buffer: Union[torch.Tensor, torch.SymInt, int],
    ) -> Optional[Union[torch.Tensor, torch.SymInt, int]]:
        if isinstance(buffer, torch.Tensor):
            return torch.empty_like(buffer) if buffer.requires_grad else None
        return buffer

    actual_grad_score_mod_captured = [_maybe_new_buffer(buffer) for buffer in score_mod_other_buffers]

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
    scores, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        fw_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    masked_out_rows = logsumexp == -float("inf")
    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))
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
    # Gradient of the inline score_mod function, with respect to the scores
    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    out_dims = [0, None, None, None, None] + [None] * len(score_mod_other_buffers)
    from .x_attention import _vmap_for_bhqkv

    # inputs are [score, b, h, q_idx, kv_idx, gradOut, ...]
    # score and gradOut are "fully" batched
    joint_score_mod = _vmap_for_bhqkv(
        joint_graph,
        prefix=(0,),
        suffix=(0,) + captured_buffers_in_dim,
        out_dims=out_dims,
    )
    with TransformGetItemToIndex():
        grad_scores, _, _, _, _, *grad_score_mod_captured = joint_score_mod(
            scores, b, h, m, n, grad_score_mod, *score_mod_other_buffers
        )
    grad_scores = grad_scores * scale
    grad_scores = grad_scores.to(query.dtype)

    mask_mod = _vmap_for_bhqkv(mask_graph, prefix=(), suffix=(None,) * len(mask_mod_other_buffers))
    with TransformGetItemToIndex():
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
    score_mod_other_buffer_grads = [
        actual_grad.copy_(grad) if isinstance(actual_grad, torch.Tensor) else None
        for actual_grad, grad in zip(actual_grad_score_mod_captured, grad_score_mod_captured)
    ]

    return (
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
        tuple(score_mod_other_buffer_grads),
    )
