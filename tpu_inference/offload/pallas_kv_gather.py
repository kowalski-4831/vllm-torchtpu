# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pallas DMA kernel for gathering KV cache blocks into a staging buffer.

Complements pallas_kv_scatter: while scatter writes N source blocks into
scattered positions of a destination cache, gather reads from scattered
positions of a source cache into N contiguous output blocks.

Used in the D2H offloading path to consolidate scattered KV-cache blocks
into a contiguous HBM staging buffer before a single bulk XLA TransferToHost.
A single staging copy() on a transfer stream has near-zero semaphore overhead
compared to calling CopyRawToHost once per block.

Design follows pallas_kv_scatter.py: PrefetchScalarGridSpec with
scalar-prefetched block IDs, HBM-only memory spaces, two-phase DMA
(start all → wait all) for maximum DMA parallelism within the kernel.

Usage (pure JAX)::

    from pallas_kv_gather import pallas_kv_gather

    cache = jnp.zeros((256, 16, 8, 128), dtype=jnp.bfloat16)
    ids   = jnp.array([3, 17, 42, 99], dtype=jnp.int32)
    gathered = pallas_kv_gather(cache, ids)  # shape (4, 16, 8, 128)

Usage (PyTorch via custom_jax_kernel bridge)::

    from torch_tpu._internal import pallas as tpu_pallas
    gather_fn = tpu_pallas.custom_jax_kernel(pallas_kv_gather)
    staging = gather_fn(kv_cache, block_ids_i32)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# TPU DMA requires the last dimension to be a multiple of 128 elements.
_DMA_ALIGNMENT = 128

# ---------------------------------------------------------------------------
# Pallas kernel body
# ---------------------------------------------------------------------------


def _gather_kernel(
        # --- scalar-prefetched arguments ---
        num_blocks_ref,  # int32[1]: number of blocks to gather
        block_ids_ref,  # int32[max_blocks]: source page indices in src_cache
        # --- HBM operands ---
    src_ref,  # HBM input: KV cache [total_pages, ...]
        # --- output ---
    dst_ref_out,  # HBM output: gathered blocks [max_blocks, ...]
        sem,  # DMA semaphore
):
    """Gather src_cache[block_ids[i]] → dst_ref_out[i] for i in [0, n).

    Two loops: first queues all DMAs, second waits.  This maximises DMA
    parallelism within the kernel.
    """
    n = num_blocks_ref[0]

    @pl.loop(0, n)
    def _start(i):
        src_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(src_idx, 1)],
            dst_ref_out.at[pl.ds(i, 1)],
            sem,
        ).start()

    @pl.loop(0, n)
    def _wait(i):
        src_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(src_idx, 1)],
            dst_ref_out.at[pl.ds(i, 1)],
            sem,
        ).wait()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pallas_kv_gather(
    src_cache: jax.Array,
    block_ids: jax.Array,
    *,
    num_blocks: int | None = None,
) -> jax.Array:
    """Gather blocks from *src_cache* at positions given by *block_ids*.

    Args:
        src_cache: KV cache array of shape ``[total_pages, page_size, ...]``.
        block_ids: 1-D ``int32`` array of length N (or padded to power-of-2).
            ``block_ids[i]`` is the source page index for ``output[i]``.
        num_blocks: Actual block count (≤ ``block_ids.shape[0]``).  If *None*,
            derived from ``block_ids.shape[0]``.  Useful when *block_ids* is
            padded to a fixed bucket size to avoid recompilation.

    Returns:
        Array of shape ``[block_ids.shape[0], page_size, ...]``.  The first
        ``num_blocks`` rows contain the gathered cache blocks; rows beyond
        ``num_blocks`` (padding) hold copies of the last real block.
    """
    if num_blocks is None:
        num_blocks = block_ids.shape[0]

    if src_cache.ndim < 2:
        raise ValueError(
            f"src_cache must be at least 2-D, got shape {src_cache.shape}")

    # --- 128-element alignment on last dim ---------------------------------
    orig_last = src_cache.shape[-1]
    needs_pad = orig_last % _DMA_ALIGNMENT != 0
    if needs_pad:
        pad_amount = _DMA_ALIGNMENT - (orig_last % _DMA_ALIGNMENT)
        pad_widths = [(0, 0)] * (src_cache.ndim - 1) + [(0, pad_amount)]
        src_cache = jnp.pad(src_cache, pad_widths)

    # --- scalar-prefetched metadata ----------------------------------------
    max_blocks = block_ids.shape[0]
    num_blocks_arr = jnp.array([num_blocks], dtype=jnp.int32)
    block_ids = block_ids.astype(jnp.int32)

    if block_ids.shape[0] < max_blocks:
        block_ids = jnp.pad(
            block_ids,
            (0, max_blocks - block_ids.shape[0]),
            constant_values=0,
        )

    block_shape = src_cache.shape[1:]
    dst_shape = (max_blocks, ) + block_shape

    # --- build pallas_call -------------------------------------------------
    result = pl.pallas_call(
        _gather_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1, ),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # src_cache
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(dst_shape, src_cache.dtype),
    )(num_blocks_arr, block_ids, src_cache)

    # --- strip alignment padding -------------------------------------------
    if needs_pad:
        result = result[..., :orig_last]

    return result
