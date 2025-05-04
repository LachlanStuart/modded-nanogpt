"""This module implements the user facing API for x_attention in PyTorch."""

from functools import partial
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, _ordered_to_dense
from lovely_tensors import monkey_patch

monkey_patch()

LSE_DTYPE = torch.float32


def aggregate_with_lse(
    outs: Tensor,  # shape (N, d)
    lses: Tensor,  # shape (N,)
) -> tuple[Tensor, Tensor]:  # shapes (d,), (,)
    max_log = lses.max()
    scaled_lses = (lses - max_log).exp()
    sum_lse = scaled_lses.sum()
    scales = (scaled_lses / sum_lse).type_as(outs)
    out = (outs * scales[:, None]).sum(dim=0)
    lse = sum_lse.log() + max_log

    # print(f"{outs=}")
    # print(f"{lses=}")
    # print(f"{max_log=}")
    # print(f"{scaled_lses=}")
    # print(f"{sum_lse=}")
    # print(f"{scales=}")
    # print(f"{out=}")
    # print(f"{lse=}")

    return out, lse


if __name__ == "__main__":
    outs = torch.randn(1000, 128)
    lses = torch.randn(1000)
    out, lse = aggregate_with_lse(outs, lses)


def mask_mod(b, h, q_idx, k_idx):
    return q_idx >= k_idx
    # return (q_idx[None] >= k_idx[None]).squeeze(0)


def process_full(
    query: Tensor,  # shape (Dk,)
    query_i: Tensor,  # shape ()
    key: Tensor,  # shape (Dk,)
    value: Tensor,  # shape (Dv,)
    key_i: Tensor,  # shape ()
) -> Tuple[Tensor, Tensor]:  # shapes (Dv,), (,)
    return value, (query * key).sum().float()


def process_partial(
    query: Tensor,  # shape (Dk,)
    query_i: Tensor,  # shape ()
    key: Tensor,  # shape (Dk,)
    value: Tensor,  # shape (Dv,)
    key_i: Tensor,  # shape ()
    mask_mod: Callable,
) -> Tuple[Tensor, Tensor]:  # shapes (Dv,), (,)
    def unmasked(query, key, value):
        score = torch.dot(query, key).float().exp()
        out = score * value
        lse = score.log()
        return out, lse

    def masked(query, key, value):
        return masked_val(query, key, value), masked_lse(query, key, value)

    def unmasked_val(query, key, value):
        return value.clone()

    def unmasked_lse(query, key, value):
        # WORKAROUND: torch.dot doesn't support being so nested...
        # score = torch.dot(query, key).float()
        return (query * key).sum().float()

    def masked_val(query, key, value):
        return value.new_zeros(V)

    def masked_lse(query, key, value):
        return value.new_full((), -torch.inf, dtype=torch.float32)

    V = value.shape[-1]
    mask = mask_mod(0, 0, query_i, key_i)
    # WORKAROUND: torch.cond doesn't seem to like the returned tuples here. No idea why. Other places work.
    # return torch.cond(mask, unmasked, masked, (query, key, value))
    return (
        # torch.cond(mask, unmasked_val, masked_val, (query, key, value)),
        torch.where(mask, unmasked_val(query, key, value), masked_val(query, key, value)),
        # torch.cond(mask, unmasked_lse, masked_lse, (query, key, value)),
        torch.where(mask, unmasked_lse(query, key, value), masked_lse(query, key, value)),
    )


if __name__ == "__main__":
    query = torch.randn(128)
    key = torch.randn(128)
    value = torch.randn(128)
    query_i = torch.tensor(0)
    key_i = torch.tensor(0)
    out, lse = torch.compile(process_full, fullgraph=True)(query, key, value)
    out, lse = torch.compile(process_partial, fullgraph=True)(query, query_i, key, value, key_i, mask_mod)
    print(out, lse)


def process_query_tile_full(
    query: Tensor,  # shape (Dk,)
    key: Tensor,  # shape (KV_LEN, Dk)
    value: Tensor,  # shape (KV_LEN, Dv)
) -> Tuple[Tensor, Tensor]:  # shapes (Dv,), (,)
    outs = value
    lses = (query * key).sum(dim=-1).float()

    out, lse = aggregate_with_lse(outs, lses)
    return out, lse


def process_query_tile_partial(
    query: Tensor,  # shape (Dk,)
    query_i: Tensor,  # shape ()
    key: Tensor,  # shape (KV_LEN, Dk)
    value: Tensor,  # shape (KV_LEN, Dv)
    key_i: Tensor,  # shape (KV_LEN,)
    mask_mod: Callable,
) -> Tuple[Tensor, Tensor]:  # shapes (Dv,), (,)
    process_partial_ = torch.vmap(process_partial, in_dims=(None, None, 0, 0, 0, None))
    outs, lses = process_partial_(query, query_i, key, value, key_i, mask_mod)
    out, lse = aggregate_with_lse(outs, lses)
    return out, lse


def process_tile(
    query_tile: Tensor,  # shape (Q_LEN, Dk)
    query_i: Tensor,  # shape (Q_LEN,)
    key_tile: Tensor,  # shape (KV_LEN, Dk)
    value_tile: Tensor,  # shape (KV_LEN, Dv)
    key_i: Tensor,  # shape (KV_LEN,)
    mask_val: Tensor,  # shape ()
    full_mask_val: Tensor,  # shape ()
    mask_mod: Callable,
) -> Tuple[Tensor, Tensor]:  # shapes (Q_LEN, Dv), (Q_LEN)
    def empty(query, query_i, key, value, key_i):
        out = value.new_zeros(query.shape[:1] + value.shape[1:])
        lse = value.new_full(query.shape[:1], -torch.inf, dtype=LSE_DTYPE)
        return out, lse

    def partial_tile(query, query_i, key, value, key_i):
        # query, query_i, key, value, key_i, mask_mod
        process_query_tile_ = torch.vmap(process_query_tile_partial, in_dims=(0, 0, None, None, None, None))
        return process_query_tile_(query, query_i, key, value, key_i, mask_mod)

    def full(query, query_i, key, value, key_i):
        # query, key, value
        process_query_tile_ = torch.vmap(process_query_tile_full, in_dims=(0, None, None))
        return process_query_tile_(query, key, value)

    def non_empty(query, query_i, key, value, key_i):
        # return torch.cond(full_mask_val, full, partial_tile, (query, query_i, key, value, key_i))
        partial_outs, partial_lse = partial_tile(query, query_i, key, value, key_i)
        full_outs, full_lse = full(query, query_i, key, value, key_i)
        return (
            torch.where(full_mask_val, full_outs, partial_outs),
            torch.where(full_mask_val, full_lse, partial_lse),
        )

    # return torch.cond(mask_val, non_empty, empty, (query_tile, query_i, key_tile, value_tile, key_i))
    empty_outs, empty_lse = empty(query_tile, query_i, key_tile, value_tile, key_i)
    non_empty_outs, non_empty_lse = non_empty(query_tile, query_i, key_tile, value_tile, key_i)
    return (
        torch.where(mask_val, non_empty_outs, empty_outs),
        torch.where(mask_val, non_empty_lse, empty_lse),
    )


if __name__ == "__main__":
    query = torch.randn(128, 48).cuda()
    key = torch.randn(128, 48).cuda()
    value = torch.randn(128, 56).cuda()
    query_i = torch.arange(128).cuda()
    key_i = torch.arange(128).cuda()
    mask_val = torch.tensor(False).cuda()
    full_mask_val = torch.tensor(False).cuda()
    out, lse = torch.compile(process_tile, fullgraph=True)(
        query, query_i, key, value, key_i, mask_val, full_mask_val, mask_mod
    )
    print(out)
    print(lse)


def process_query_tiles(
    query_tile: Tensor,  # shape (q_tile_size, Dk)
    query_i: Tensor,  # shape (q_tile_size,)
    key_tiles: Tensor,  # shape (k_tiles, k_tile_size, Dk)
    value_tiles: Tensor,  # shape (k_tiles, k_tile_size, Dv)
    key_i: Tensor,  # shape (k_tiles, k_tile_size)
    coarse_mask: Tensor,  # shape (k_tiles,)
    coarse_full_mask: Tensor,  # shape (k_tiles,)
    mask_mod: Callable,
) -> Tuple[Tensor, Tensor]:  # shapes (q_tile_size, Dv), (q_tile_size)
    # query, query_i, key, value, key_i, mask_val, full_mask_val, mask_mod
    process_tile_ = torch.vmap(process_tile, in_dims=(None, None, 0, 0, 0, 0, 0, None))
    outs, lses = process_tile_(
        query_tile, query_i, key_tiles, value_tiles, key_i, coarse_mask, coarse_full_mask, mask_mod
    )

    aggregate_with_lse_ = torch.vmap(aggregate_with_lse, in_dims=(1, 1))
    out, lse = aggregate_with_lse_(outs, lses)
    return out, lse


if __name__ == "__main__":
    query = torch.randn(24, 48).cuda()
    query_i = torch.arange(24).cuda()
    key = torch.randn(3, 24, 48).cuda()
    value = torch.randn(3, 24, 56).cuda()
    key_i = torch.arange(3 * 24).reshape(3, 24).cuda()
    coarse_mask = torch.randint(0, 2, (3,)).bool().cuda()
    coarse_full_mask = torch.randint(0, 2, (3,)).bool().cuda()
    out, lse = torch.compile(process_query_tiles, fullgraph=True)(
        query, query_i, key, value, key_i, coarse_mask, coarse_full_mask, mask_mod
    )
    print(out)
    print(lse)


class VmapAttentionMask(NamedTuple):
    partial_dense: Tensor
    full_dense: Tensor
    mask_mod: Callable
    q_tile_size: int
    k_tile_size: int

    @staticmethod
    def from_block_mask(block_mask: BlockMask) -> "VmapAttentionMask":
        partial_dense = _ordered_to_dense(block_mask.kv_num_blocks, block_mask.kv_indices).bool()
        if block_mask.full_kv_num_blocks is not None:
            full_dense = _ordered_to_dense(block_mask.full_kv_num_blocks, block_mask.full_kv_indices).bool()
            partial_dense = partial_dense | full_dense
        else:
            full_dense = torch.zeros_like(partial_dense)
        return VmapAttentionMask(
            partial_dense=partial_dense,
            full_dense=full_dense,
            mask_mod=block_mask.mask_mod,
            q_tile_size=block_mask.BLOCK_SIZE[1],
            k_tile_size=block_mask.BLOCK_SIZE[0],
        )

    def to(self, device: torch.device) -> "VmapAttentionMask":
        return VmapAttentionMask(
            partial_dense=self.partial_dense.to(device),
            full_dense=self.full_dense.to(device),
            mask_mod=self.mask_mod,
            q_tile_size=self.q_tile_size,
            k_tile_size=self.k_tile_size,
        )


def vmap_attention(
    query: Tensor,  # shape (B, H, Q_LEN, Dk)
    key: Tensor,  # shape (B, H, KV_LEN, Dk)
    value: Tensor,  # shape (B, H, KV_LEN, Dv)
    mask: VmapAttentionMask,
    scale: float,
) -> Tuple[Tensor, Tensor]:  # shapes (B, H, Q_LEN, Dv), (B, H, Q_LEN)
    query = query * scale
    assert query.shape[:2] == key.shape[:2] == value.shape[:2], f"{query.shape} != {key.shape} != {value.shape}"
    assert query.shape[0] == 1, f"{query.shape}"
    assert query.ndim == 4, f"{query.ndim}"
    assert query.shape[2] % mask.q_tile_size == 0, f"{query.shape[2]} % {mask.q_tile_size} != 0"
    assert query.shape[3] == key.shape[3], f"{query.shape[3]} != {key.shape[3]}"
    assert key.shape[2] % mask.k_tile_size == 0, f"{key.shape[2]} % {mask.k_tile_size} != 0"
    assert mask.partial_dense.ndim == 4, f"{mask.partial_dense.ndim=}"
    assert mask.partial_dense.shape[1] == 1, f"{mask.partial_dense.shape=}"
    assert mask.full_dense.ndim == 4, f"{mask.full_dense.ndim=}"

    partial_mask = mask.partial_dense[:, 0, :, :]
    full_mask = mask.full_dense[:, 0, :, :]

    q_tile_size = mask.q_tile_size
    k_tile_size = mask.k_tile_size
    B, H, Q_LEN, Dk = query.shape
    B, H, KV_LEN, Dv = value.shape
    # query, query_i, key, value, key_i, coarse_mask, coarse_full_mask, mask_mod
    process_query_tiles_ = torch.vmap(process_query_tiles, in_dims=(0, 0, None, None, None, 0, 0, None))
    process_tiles_by_head = torch.vmap(process_query_tiles_, in_dims=(0, None, 0, 0, None, None, None, None))
    process_tiles_by_batch = torch.vmap(process_tiles_by_head, in_dims=(0, None, 0, 0, None, 0, 0, None))
    q_tiles = Q_LEN // q_tile_size
    k_tiles = KV_LEN // k_tile_size
    query_i = torch.arange(Q_LEN, device=query.device).view(q_tiles, q_tile_size)
    key_i = torch.arange(KV_LEN, device=query.device).view(k_tiles, k_tile_size)
    query_tiles = query.view(B, H, q_tiles, q_tile_size, Dk)
    key_tiles = key.view(B, H, k_tiles, k_tile_size, Dk)
    value_tiles = value.view(B, H, k_tiles, k_tile_size, Dv)
    out, lse = process_tiles_by_batch(
        query_tiles, query_i, key_tiles, value_tiles, key_i, partial_mask, full_mask, mask.mask_mod
    )
    out = out.view(B, H, Q_LEN, Dv)
    lse = lse.view(B, H, Q_LEN)
    return out, lse


if __name__ == "__main__":
    query = torch.randn(1, 1, 96, 48).half().cuda()
    key = torch.randn(1, 1, 32, 48).half().cuda()
    value = torch.randn(1, 1, 32, 56).half().cuda()
    mask = VmapAttentionMask(
        partial_dense=torch.randint(0, 2, (1, 3, 2)).bool().cuda(),
        full_dense=torch.randint(0, 2, (1, 3, 2)).bool().cuda(),
        mask_mod=mask_mod,
        q_tile_size=32,
        k_tile_size=16,
    )
    out, lse = vmap_attention(query, key, value, mask)
    print(out)
    print(lse)


if __name__ == "__main__":
    query = torch.randn(1, 3, 16384, 128).half().cuda()
    key = torch.randn(1, 3, 16384, 128).half().cuda()
    value = torch.randn(1, 3, 16384, 128).half().cuda()
    coarse_mask = torch.randint(0, 1, (1, 128, 128)).bool().cuda()
    coarse_full_mask = torch.randint(0, 1, (1, 128, 128)).bool().cuda()
    out, lse = torch.compile(vmap_attention, fullgraph=True)(
        query, key, value, coarse_mask, coarse_full_mask, mask_mod, 128, 128
    )
    print(out)
    print(lse)
