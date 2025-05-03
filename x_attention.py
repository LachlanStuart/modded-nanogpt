"""This module implements the user facing API for x_attention in PyTorch."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
from x_attention_hop import XAttentionOp
from torch._higher_order_ops.utils import _set_compilation_env
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    _temp_remove_pre_dispatch_torch_function_mode,
)
from torch.nn.attention._utils import _supported_head_dim, _validate_sdpa_input
from torch.nn.attention.flex_attention import _identity, _create_empty_block_mask, BlockMask


__all__ = [
    "BlockMask",
    "x_attention",
]

_mask_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def _vmap_for_bhqkv(
    fn: Callable,
    prefix: Tuple[Optional[int], ...],
    suffix: Tuple[Optional[int], ...] = (),
    out_dims: Union[int, List[Optional[int]]] = 0,
    group_dim: bool = False,
):
    """Used to vmap mask_mods over 4-dimensional/5-dimension inputs.
    Mapping over the [b, hq, q_idx, kv_idx] or [b, hkv, g, q_idx, kv_idx] dimensions.

    Args:
        fn (callable): The function to vmap.
        prefix (tuple): The prefix of the vmap. Should be set to () for mask_mods.
        suffix (tuple): We need to add (0,) if gradOut is being mapped over,
                        and (None,) * len(other_buffers).
        out_dims (tuple): For forward cases, keep this as the default 0 since
                          we are only returning 1 output. For backwards, the joint
                          graph returns grads for B, H, Q_idx, KV_idx and other_buffers,
                          so we set this to (0, None, None, None, None) + (None,) * len(other_buffers).

    Returns:
        callable: The vmapped function.
    """
    # We vamp a function 4 times, broadcasting the [b, h, q_idx, kv_idx] dimensions
    dimensions: List[Tuple[None | int, None | int, None | int, None | int]] = []
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
        (None, 0, None, None),
    ]

    if group_dim:
        dimensions += [
            (None, 0, None, None),
        ]

    dimensions += [
        (0, None, None, None),
    ]

    for dims in dimensions:
        fn = torch.vmap(fn, in_dims=prefix + dims + suffix, out_dims=out_dims)  # type: ignore[arg-type]
    return fn


_LARGE_SPARSE_BLOCK_SIZE = 1 << 30


def create_mask(
    mask_mod: _mask_mod_signature,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
) -> Tensor:
    r"""This function creates a mask tensor from a mask_mod function.

    Args:
        mask_mod (_mask_mod_signature): Function to generate the mask.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    if B is None:
        B = 1
    if H is None:
        H = 1
    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)

    with TransformGetItemToIndex():
        mask_mod_vmap = _vmap_for_bhqkv(mask_mod, prefix=())
        mask = mask_mod_vmap(b, h, m, n)
        return mask


def _apply_kernel_options(query: Tensor, key: Tensor, value: Tensor, return_lse: bool, kernel_options):
    kernel_options = {} if kernel_options is None else dict(kernel_options)

    kernel_options.setdefault("PRESCALE_QK", False)
    kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
    kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
    # This forces all biases grad scatters to be done in the DQ iteration loop of the backwards
    kernel_options.setdefault("WRITE_DQ", True)

    # If forward kernel needs to return logsumexp is decided by this rule internally.
    assert "OUTPUT_LOGSUMEXP" not in kernel_options
    kernel_options["OUTPUT_LOGSUMEXP"] = True
    if not return_lse:
        # We used to check if q,k,v required grads but since captured buffers can require grad
        # we always write unless in no_grad
        output_logsumexp = torch.is_grad_enabled()
        kernel_options["OUTPUT_LOGSUMEXP"] = output_logsumexp
        any_inputs_on_cpu_device = query.device.type == "cpu" or key.device.type == "cpu" or value.device.type == "cpu"
        if any_inputs_on_cpu_device:
            # CPU with torch.compile now supports infernece, and will not return lse
            # TODO: support CPU for training and return lse
            kernel_options["OUTPUT_LOGSUMEXP"] = False

    return kernel_options


def _validate_embed_dim(query: Tensor, key: Tensor, value: Tensor):
    if query.size(-1) != key.size(-1):
        raise ValueError(
            f"Expect query and key/value to have the same embedding dimension "
            f"but got E={query.size(-1)} and E={key.size(-1)}."
        )
    # TODO this config segfaults with Triton without:
    # https://github.com/triton-lang/triton/pull/4540
    if not (_supported_head_dim(query.size(-1)) and _supported_head_dim(value.size(-1))):
        raise ValueError(
            f"NYI: Currently non power of 2 embedding dimension are not supported. "
            f"Got E={query.size(-1)} and Ev={value.size(-1)}."
        )


def _validate_device(query: Tensor, key: Tensor, value: Tensor):
    """TODO: Remove once non cuda/cpu devices support is added
    We only need to check query since we have already that q,k,v are on the same device
    """
    if query.device.type != "cuda" and query.device.type != "cpu":
        raise ValueError(
            f"XAttention is only supported on CUDA or CPU devices. Found input tensors on {query.device.type} device."
        )


def _validate_nestedness(query: Tensor, key: Tensor, value: Tensor):
    # Currently, inputs can only be all nested or no nested.
    if query.is_nested != key.is_nested or key.is_nested != value.is_nested:
        raise ValueError(
            "XAttention does not support mixed nested tensor / non-nested tensor inputs. "
            "Please file an issue requesting this if it is important to you."
        )

    if (
        (query.is_nested and query._lengths is not None)  # type: ignore[attr-defined]
        or (key.is_nested and key._lengths is not None)  # type: ignore[attr-defined]
        or (value.is_nested and value._lengths is not None)  # type: ignore[attr-defined]
    ):
        raise ValueError(
            "XAttention does not support nested tensors that are non-contiguous with holes. "
            "Please file an issue requesting this if it is important to you."
        )


def x_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    # Some basic input validation
    _validate_sdpa_input(query, key, value)
    _validate_embed_dim(query, key, value)
    _validate_device(query, key, value)
    _validate_nestedness(query, key, value)
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise NotImplementedError("NYI: query, key, and value must be 4D tensors")
    if (not enable_gqa) and query.size(-3) != key.size(-3):
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.size(-3)} and Hkv={key.size(-3)}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa:
        Hq = query.size(1)
        Hkv = key.size(1)
        if Hq % Hkv != 0:
            raise ValueError(
                f"Expect number of query heads to be a multiple of kv heads for GQA but got Hq={Hq} and Hkv={Hkv}."
            )
    if query.size(0) != key.size(0):
        if block_mask is None:
            raise ValueError(
                f"Expect query and key/value to have the same batch size, "
                f"or non-none block_mask, "
                f"but got block_mask=None, Bq={query.size(0)}, and Bkv={key.size(0)}."
            )

        if block_mask.kv_num_blocks.size(0) != query.size(0):
            raise ValueError(
                f"Expect query and key/value to have the same batch size, "
                f"or block_mask and query to have the same batch size, "
                f"but got Bq={query.size(0)}, Bkv={key.size(0)}, B_block_mask={block_mask.kv_num_blocks.size(0)}."
            )

    if block_mask is None:
        block_mask = _create_empty_block_mask(query, key)

    if block_mask.BLOCK_SIZE[0] == _LARGE_SPARSE_BLOCK_SIZE and block_mask.BLOCK_SIZE[1] == _LARGE_SPARSE_BLOCK_SIZE:
        # This corresponds to the case where we essentially have a "no-op" block mask.
        pass
    elif query.is_nested:
        if block_mask.shape[-2] != query._values.size(query._ragged_idx - 1):  # type: ignore[attr-defined]
            raise RuntimeError(
                f"block_mask of shape {block_mask.shape} is not compatible with nested tensor input "
                f"with total sequence length of {query._values.size(query._ragged_idx - 1)}"  # type: ignore[attr-defined]
            )
    else:
        block_mask_q_len = block_mask.shape[-2]
        block_mask_kv_len = block_mask.shape[-1]
        if query.size(-2) > block_mask_q_len or key.size(-2) > block_mask_kv_len:
            raise ValueError(
                f"block_mask was created for block_mask.shape={block_mask.shape} but got q_len={query.size(-2)} and kv_len={key.size(-2)}. "
                "As the block mask was created for a smaller length than you're using it for, you likely need to create a new block mask."
            )
        elif (query.size(-2) < block_mask_q_len and key.size(-2) <= block_mask_kv_len) or (
            query.size(-2) <= block_mask_q_len and key.size(-2) < block_mask_kv_len
        ):
            raise ValueError(
                f"block_mask was created for block_mask.shape={block_mask.shape} but got q_len={query.size(-2)} and kv_len={key.size(-2)}. "
                "As the block mask was created for a larger length than you're using it for, you can either 1. create a new block mask with the correct length, or 2. 'adjust' the existing block mask to the correct length by calling block_mask._adjust(q_len, kv_len). This essentially 'crops' the block mask to the upper left corner, which does not work for all mask_mods!"
            )
        assert query.size(-2) == block_mask_q_len
        assert key.size(-2) == block_mask_kv_len

    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    if query.device != block_mask.kv_num_blocks.device:  # type: ignore[union-attr]
        raise RuntimeError(
            f"Expect q/k/v and block_mask to be on the same device "
            f"but got {query.device} and {block_mask.kv_num_blocks.device}."  # type: ignore[union-attr]
        )

    kernel_options = _apply_kernel_options(
        query,
        key,
        value,
        return_lse,
        kernel_options,
    )

    if torch.compiler.is_dynamo_compiling():
        # mark head_dim and number of heads to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)

        out, lse = XAttentionOp.apply(
            query,
            key,
            value,
            block_mask.as_tuple(),
            scale,
            kernel_options,  # type: ignore[union-attr]
            (),
        )
        if return_lse:
            return out, lse * math.log(2)
        else:
            return out

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("x_attention requires dynamo support")

    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_mode,
    )

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass hop to it. So we wrap it in a dummy function.
    def _x_attention_hop_wrapper(*args, **kwargs):
        return XAttentionOp.apply(*args, **kwargs)

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                    if metadata_mode:
                        backend = make_eager_backend_with_torch_function_mode(metadata_mode)
                    else:
                        backend = "eager"
                    out, lse = torch.compile(_x_attention_hop_wrapper, backend=backend, fullgraph=True)(
                        query,
                        key,
                        value,
                        block_mask.as_tuple(),  # type: ignore[union-attr]
                        scale,
                        kernel_options,
                        (),
                    )
                    if return_lse:
                        return out, lse * math.log(2)
                    else:
                        return out
