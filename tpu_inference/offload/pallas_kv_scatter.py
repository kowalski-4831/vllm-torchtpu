# Copyright 2025 Google LLC
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
"""Pallas DMA kernel for in-place KV cache scatter on TPU.

Replaces the XLA-compiled ``index_put_`` which copies the entire KV cache
before scattering a handful of blocks.  This kernel uses TPU HBM-to-HBM DMA
to write only the target blocks, leaving the rest of the buffer untouched.

Design follows the patterns established in
https://github.com/vllm-project/tpu-inference/pull/2026:

* ``PrefetchScalarGridSpec`` with scalar-prefetched block IDs so that changing
  IDs at runtime does not trigger recompilation.
* ``input_output_aliases`` so the destination buffer is mutated in place (no
  allocation, no full copy).
* ``pl.loop`` for dynamic iteration over a runtime-variable number of blocks.
* HBM-only memory spaces — pure DMA engine, no VMEM round-trips.

Usage (pure JAX)::

    import jax.numpy as jnp
    from pallas_kv_scatter import pallas_kv_scatter

    dest = jnp.zeros((256, 16, 8, 128), dtype=jnp.bfloat16)
    src  = jnp.ones((8, 16, 8, 128),   dtype=jnp.bfloat16)
    ids  = jnp.array([3, 17, 42, 99, 130, 180, 200, 251], dtype=jnp.int32)
    dest = pallas_kv_scatter(dest, src, ids)

Usage (PyTorch via custom_jax_kernel bridge)::

    from torch_tpu._internal import pallas as tpu_pallas
    scatter_fn = tpu_pallas.custom_jax_kernel(pallas_kv_scatter)
    kv_cache = scatter_fn(kv_cache, src_blocks, block_ids_i32)
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


def _scatter_kernel(
        # --- scalar-prefetched arguments (loaded before kernel body runs) ---
        num_blocks_ref,  # int32[1]: number of blocks to scatter
        block_ids_ref,  # int32[max_blocks]: destination page indices
        # --- HBM operands (from in_specs) ---
    src_ref,  # HBM input: source blocks [max_blocks, ...]
        dest_ref,  # HBM input (aliased to output 0): KV cache [total_pages, ...]
        # --- output + scratch (from out_specs + scratch_shapes) ---
    dest_ref_out,  # HBM output (same buffer as dest_ref via input_output_aliases)
        sem,  # DMA semaphore
):
    """Scatter ``src_ref`` blocks into ``dest_ref`` at positions given by
    ``block_ids_ref``, using asynchronous HBM-to-HBM DMA.

    Two loops are used: the first queues all DMAs (start), the second waits
    for completion.  This maximises DMA parallelism within the kernel.
    """
    n = num_blocks_ref[0]

    # --- start all DMAs ---
    @pl.loop(0, n)
    def _start(i):
        dst_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(i, 1)],
            dest_ref.at[pl.ds(dst_idx, 1)],
            sem,
        ).start()

    # --- wait for all DMAs ---
    @pl.loop(0, n)
    def _wait(i):
        dst_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(i, 1)],
            dest_ref.at[pl.ds(dst_idx, 1)],
            sem,
        ).wait()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pallas_kv_scatter(
    dest_cache: jax.Array,
    src_blocks: jax.Array,
    block_ids: jax.Array,
    *,
    num_blocks: int | None = None,
) -> jax.Array:
    """In-place scatter of *src_blocks* into *dest_cache* using Pallas DMA.

    Args:
        dest_cache: KV cache array of shape ``[total_pages, page_size, ...]``.
        src_blocks: Source blocks, shape ``[N, page_size, ...]`` where the
            trailing dimensions must match *dest_cache*.
        block_ids: 1-D ``int32`` array of length ``N`` (or longer if padded).
            ``block_ids[i]`` is the destination page index for ``src_blocks[i]``.
        num_blocks: Optional explicit block count.  If *None*, derived from
            ``src_blocks.shape[0]``.  Useful when *block_ids* / *src_blocks*
            are padded to a fixed size to avoid recompilation.

    Returns:
        The updated *dest_cache* (same buffer when ``input_output_aliases``
        is honoured by the compiler).
    """
    if num_blocks is None:
        num_blocks = src_blocks.shape[0]

    # --- validate shapes ---------------------------------------------------
    if dest_cache.ndim < 2:
        raise ValueError(
            f"dest_cache must be at least 2-D, got shape {dest_cache.shape}")
    if src_blocks.shape[1:] != dest_cache.shape[1:]:
        raise ValueError(
            f"Trailing dimensions mismatch: src_blocks {src_blocks.shape[1:]} "
            f"vs dest_cache {dest_cache.shape[1:]}")

    # --- 128-element alignment on last dim ---------------------------------
    orig_last = dest_cache.shape[-1]
    needs_pad = orig_last % _DMA_ALIGNMENT != 0
    if needs_pad:
        pad_amount = _DMA_ALIGNMENT - (orig_last % _DMA_ALIGNMENT)
        pad_widths = [(0, 0)] * (dest_cache.ndim - 1) + [(0, pad_amount)]
        dest_cache = jnp.pad(dest_cache, pad_widths)
        src_blocks = jnp.pad(src_blocks, pad_widths)

    # --- scalar-prefetched metadata ----------------------------------------
    num_blocks_arr = jnp.array([num_blocks], dtype=jnp.int32)
    block_ids = block_ids.astype(jnp.int32)

    # The kernel operates on the *padded* max block count from src_blocks so
    # that the traced shape is static; the ``pl.loop`` bound (num_blocks_arr)
    # gates the actual iteration count at runtime.
    max_blocks = src_blocks.shape[0]

    # Pad block_ids to max_blocks if shorter (values beyond num_blocks are
    # never read thanks to the loop bound).
    if block_ids.shape[0] < max_blocks:
        block_ids = jnp.pad(
            block_ids,
            (0, max_blocks - block_ids.shape[0]),
            constant_values=0,
        )

    # --- build pallas_call -------------------------------------------------
    dest_result = pl.pallas_call(
        _scatter_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1, ),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # src_blocks
                pl.BlockSpec(memory_space=pltpu.HBM),  # dest_cache (aliased)
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(dest_cache.shape, dest_cache.dtype),
        input_output_aliases={3: 0},  # input arg idx 3 (dest_cache) → output 0
        # NOTE: has_side_effects=True is omitted because the torch_tpu SPMD
        # partitioner requires explicit sharding on side-effecting ops.  The
        # kernel result is consumed, so XLA will not eliminate it.
    )(num_blocks_arr, block_ids, src_blocks, dest_cache)

    # --- strip alignment padding -------------------------------------------
    if needs_pad:
        dest_result = dest_result[..., :orig_last]

    return dest_result
