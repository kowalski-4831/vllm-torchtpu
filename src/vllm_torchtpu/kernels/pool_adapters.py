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
"""Typed access to byte-regions of the unified KV pool.

The pool is the stock attention-shaped KV cache
`(num_blocks, block_size, num_kv_heads * 2, head_size)` in the KV dtype —
the one tensor per physical buffer that every sharing layer receives.
Attention consumes it natively. Consumers that pack other element types
into a block's byte-region (the GDN f32 ssm state, the conv slot) go
through these adapters: per-request gather/scatter of a token-range,
reinterpreted in VMEM via Mosaic `ref.bitcast` when the element type
differs — the only place a cross-dtype view is free on TPU (an XLA-level
bitcast of a pool slice forces a pool-sized layout copy per step).

Regions are token-ranges `[tok0, tok0 + ntok)` within a block with
`tok0 % ntok == 0`, so the block spec covers exactly the region and one
grid step moves only the region's bytes. Scatters alias the pool in place
(`input_output_aliases`); the unwritten complement is preserved via the
HBM alias, with no read-modify-write.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _pool_geometry(pool):
    """(block_size, payload dims between token and lane, lanes)."""
    assert pool.ndim >= 3, pool.shape
    return pool.shape[1], tuple(pool.shape[2:-1]), pool.shape[-1]


def _bitcast_payload(payload: tuple[int, ...], pool_dtype, out_dtype):
    """Payload dims of a region block after an in-kernel bitcast.

    Mosaic ``ref.bitcast`` rescales the second-minor dim by the
    element-size ratio: the last payload dim, or — for a 3-D pool
    ``(nb, rows, lanes)`` whose block ref is already 2-D — the token/row
    dim itself (handled by the callers via ``_out_rows``).
    """
    elem_in = jnp.dtype(pool_dtype).itemsize
    elem_out = jnp.dtype(out_dtype).itemsize
    if not payload:
        return payload
    last = payload[-1] * elem_in
    assert last % elem_out == 0, (payload, pool_dtype, out_dtype)
    return payload[:-1] + (last // elem_out, )


def _rescale_rows(rows: int, pool_dtype, out_dtype) -> int:
    elem_in = jnp.dtype(pool_dtype).itemsize
    elem_out = jnp.dtype(out_dtype).itemsize
    scaled = rows * elem_in
    assert scaled % elem_out == 0, (rows, pool_dtype, out_dtype)
    return scaled // elem_out


def _out_rows(payload: tuple[int, ...], ntok: int) -> int:
    rows = ntok
    for dim in payload:
        rows *= dim
    return rows


def gather_region(pool,
                  state_indices,
                  *,
                  tok0: int,
                  ntok: int,
                  out_dtype,
                  out_lanes: int | None = None):
    """Per-request typed read of a pool token-range.

    pool: (num_blocks, block_size, heads2, lanes) KV dtype.
    state_indices: (num_reqs,) int32 block ids.
    returns: (num_reqs, rows, out_lanes or lanes) out_dtype. With
    ``out_lanes`` (a divisor of the pool's lane count) the kernel splits
    lanes in place, so callers reshaping to a narrower-lane state shape
    pay no XLA lane-crossing relayout; the remaining outside reshape is
    lane-preserving (free).
    """
    na = state_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    assert tok0 % ntok == 0 and tok0 + ntok <= block_size
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(out_dtype)
    out_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, out_dtype))
    out_rows = _out_rows(out_payload, ntok)
    if not payload and not same_dtype:
        # 3-D pool: the block ref is 2-D and the bitcast rescales the
        # token/row dim itself (the old validated sublane convention).
        out_rows = _rescale_rows(ntok, pool.dtype, out_dtype)
    lane_split = 1
    if out_lanes is not None and out_lanes != lanes:
        assert lanes % out_lanes == 0, (lanes, out_lanes)
        lane_split = lanes // out_lanes
        out_rows *= lane_split
    o_lanes = lanes // lane_split
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, pool_ref, o_ref):
        block = pool_ref.at[0]
        if not same_dtype:
            block = block.bitcast(jnp.dtype(out_dtype))
        arr = block[...].reshape(out_rows // lane_split, lanes)
        if lane_split > 1:
            # In-kernel lane split: 128-aligned lane slices stacked on a
            # new sublane axis preserve the row-major element order.
            arr = jnp.stack([
                arr[:, i * o_lanes:(i + 1) * o_lanes]
                for i in range(lane_split)
            ],
                            axis=1).reshape(out_rows, o_lanes)
        o_ref[...] = arr[None]

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((1, ntok) + payload + (lanes, ), lambda i, s:
                             (s[i], tok0 // ntok) + pad)
            ],
            out_specs=pl.BlockSpec((1, out_rows, o_lanes), lambda i, s:
                                   (i, 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((na, out_rows, o_lanes), out_dtype),
    )(state_indices, pool)


def gather_blocks(pool,
                  mgr_indices,
                  *,
                  split: int,
                  kb0: int,
                  nblocks: int,
                  out_dtype,
                  out_lanes: int | None = None):
    """Typed read of `nblocks` whole kernel blocks inside each request's
    manager block, one grid step and ONE DMA window per request.

    The pool is born at kernel granularity; a manager block is `split`
    consecutive kernel blocks (upstream map_to_kernel_blocks), so the
    window (split, block_size, ...) indexed by the manager id is always
    aligned (element offset = mgr * split). Reading the whole manager
    window costs a little extra DMA but restores the single-window-per-
    request shape whose per-window setup dominates the region cost.
    """
    assert 0 <= kb0 and kb0 + nblocks <= split, (kb0, nblocks, split)
    na = mgr_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(out_dtype)
    out_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, out_dtype))
    rows_pb = _out_rows(out_payload, block_size)
    if not payload and not same_dtype:
        rows_pb = _rescale_rows(block_size, pool.dtype, out_dtype)
    lane_split = 1
    if out_lanes is not None and out_lanes != lanes:
        assert lanes % out_lanes == 0, (lanes, out_lanes)
        lane_split = lanes // out_lanes
        rows_pb *= lane_split
    o_lanes = lanes // lane_split
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, pool_ref, o_ref):
        for j in range(nblocks):
            block = pool_ref.at[kb0 + j]
            if not same_dtype:
                block = block.bitcast(jnp.dtype(out_dtype))
            arr = block[...].reshape(rows_pb // lane_split, lanes)
            if lane_split > 1:
                arr = jnp.stack([
                    arr[:, k * o_lanes:(k + 1) * o_lanes]
                    for k in range(lane_split)
                ],
                                axis=1).reshape(rows_pb, o_lanes)
            o_ref[0, j * rows_pb:(j + 1) * rows_pb, :] = arr

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((split, block_size) + payload + (lanes, ),
                             lambda i, s: (s[i], 0) + pad)
            ],
            out_specs=pl.BlockSpec((1, nblocks * rows_pb, o_lanes),
                                   lambda i, s: (i, 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((na, nblocks * rows_pb, o_lanes),
                                       out_dtype),
    )(mgr_indices, pool)


def scatter_blocks(pool, vals, mgr_indices, *, split: int, kb0: int,
                   nblocks: int):
    """Typed write of `nblocks` whole kernel blocks inside each request's
    manager block (in place), one grid step per request; vals is the
    gather_blocks shape. The aliased output window covers the whole
    manager block, so the non-region kernel blocks are copied through
    from the aliased input to satisfy the full-write rule.
    """
    assert 0 <= kb0 and kb0 + nblocks <= split, (kb0, nblocks, split)
    na = mgr_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(vals.dtype)
    val_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, vals.dtype))
    rows_pb = _out_rows(val_payload, block_size)
    if not payload and not same_dtype:
        rows_pb = _rescale_rows(block_size, pool.dtype, vals.dtype)
    v_lanes = vals.shape[-1]
    lane_split = 1
    if v_lanes != lanes:
        assert lanes % v_lanes == 0, (lanes, v_lanes)
        lane_split = lanes // v_lanes
    v_rows_pb = rows_pb * lane_split
    assert vals.shape == (na, nblocks * v_rows_pb,
                          v_lanes), (vals.shape, nblocks, v_rows_pb, v_lanes)
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, val_ref, pool_in_ref, pool_out_ref):
        for j in range(split):
            if kb0 <= j < kb0 + nblocks:
                arr = val_ref[0, (j - kb0) * v_rows_pb:(j - kb0 + 1) *
                              v_rows_pb, :]
                if lane_split > 1:
                    arr = arr.reshape(rows_pb, lane_split, v_lanes)
                    arr = jnp.concatenate(
                        [arr[:, k, :] for k in range(lane_split)], axis=-1)
                block = pool_out_ref.at[j]
                if not same_dtype:
                    block = block.bitcast(jnp.dtype(vals.dtype))
                if not payload and not same_dtype:
                    block[...] = arr.reshape(rows_pb, lanes)
                else:
                    block[...] = arr.reshape((block_size, ) + val_payload +
                                             (lanes, ))
            else:
                # full-write rule: pass the untouched kernel blocks through
                pool_out_ref.at[j][...] = pool_in_ref.at[j][...]

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((1, nblocks * v_rows_pb, v_lanes), lambda i, s:
                             (i, 0, 0)),
                pl.BlockSpec((split, block_size) + payload + (lanes, ),
                             lambda i, s: (s[i], 0) + pad),
            ],
            out_specs=pl.BlockSpec((split, block_size) + payload + (lanes, ),
                                   lambda i, s: (s[i], 0) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(mgr_indices, vals, pool)


def scatter_region(pool, vals, state_indices, *, tok0: int, ntok: int):
    """Per-request typed write of a pool token-range (in place).

    pool: (num_blocks, block_size, heads2, lanes) KV dtype (aliased;
    unwritten bytes preserved).
    vals: (num_reqs, rows, lanes) — the gather_region shape.
    returns: the updated pool.
    """
    na = state_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    assert tok0 % ntok == 0 and tok0 + ntok <= block_size
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(vals.dtype)
    val_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, vals.dtype))
    out_rows = _out_rows(val_payload, ntok)
    if not payload and not same_dtype:
        out_rows = _rescale_rows(ntok, pool.dtype, vals.dtype)
    v_lanes = vals.shape[-1]
    lane_split = 1
    if v_lanes != lanes:
        # Narrow-lane vals (the gather out_lanes shape): merge lanes back
        # in-kernel; 128-aligned lane concat preserves row-major order.
        assert lanes % v_lanes == 0, (lanes, v_lanes)
        lane_split = lanes // v_lanes
    v_rows = out_rows * lane_split
    assert vals.shape == (na, v_rows, v_lanes), (vals.shape, v_rows, v_lanes)
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, val_ref, pool_in_ref, pool_out_ref):
        # The output block covers exactly the region and is fully written;
        # the complement is preserved via the HBM alias.
        block = pool_out_ref.at[0]
        if not same_dtype:
            block = block.bitcast(jnp.dtype(vals.dtype))
        arr = val_ref[0]
        if lane_split > 1:
            arr = arr.reshape(out_rows, lane_split, v_lanes)
            arr = jnp.concatenate([arr[:, i, :] for i in range(lane_split)],
                                  axis=-1)
        if not payload and not same_dtype:
            block[...] = arr.reshape(out_rows, lanes)
        else:
            block[...] = arr.reshape((ntok, ) + val_payload + (lanes, ))

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((1, v_rows, v_lanes), lambda i, s: (i, 0, 0)),
                # Aliased with the output; never read, so leave it in HBM.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((1, ntok) + payload + (lanes, ),
                                   lambda i, s: (s[i], tok0 // ntok) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(state_indices, vals, pool)


def copy_blocks(pool, src_indices, dst_indices):
    """Whole-block copies within the pool: pool[dst[i]] = pool[src[i]].

    Seeds a request's new mamba state block from its previous one whenever
    the block-table-derived state block advances (chunked-prefill boundary,
    decode crossing, or prefix-cache resume). Pairs with src == dst are
    no-ops used as padding.
    """
    na = src_indices.shape[0]
    block_shape = (1, ) + pool.shape[1:]
    pad = (0, ) * (pool.ndim - 1)

    def _copy_kernel(src_ref, dst_ref, pool_in_ref, pool_out_ref):
        pool_out_ref[...] = pool_in_ref[...]

    return pl.pallas_call(
        _copy_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec(block_shape, lambda i, s, d: (s[i], ) + pad)
            ],
            out_specs=pl.BlockSpec(block_shape, lambda i, s, d:
                                   (d[i], ) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(src_indices, dst_indices, pool)
