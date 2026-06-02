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
"""TPU ↔ CPU KV cache offloading handler.

This module plugs into vllm's `OffloadingConnector` framework to give TPU
inference a large CPU-resident host pool for KV-cache blocks evicted from
HBM. Cache hits on the host pool are streamed back into HBM via Pallas
DMA before attention reads them.

Architecture
------------
- `TPUCPUOffloadingSpec` subclasses `vllm.v1.kv_offload.cpu.spec.
  CPUOffloadingSpec`; it overrides `get_handlers()` to wire up Pallas
  gather/scatter kernels and exposes
  `estimate_hbm_reserve_bytes(vllm_config) -> int` so the worker can
  reserve HBM for the H2D staging buffer before profile_run.
- `CpuTpuOffloadingHandlers` holds one D2H and one H2D
  `SingleDirectionOffloadingHandler` instance; the spec yields them per
  the OffloadingConnector contract.
- Each handler owns a single background DMA worker thread plus (H2D only)
  a depth-1 device-side staging buffer; transfers serialize through them.
- Oversized H2D loads (more than `KV_H2D_POOL_MAX_BLOCKS` blocks) are
  split into max-buffer-sized chunks; each chunk consumes one engine
  step's worth of latency, the depth-1 invariant is preserved.

Usage
-----
Pass `--kv-transfer-config` to `vllm serve`:

    vllm serve <model> \\
        --tensor-parallel-size=8 \\
        --kv-offloading-size 400 \\
        --kv-transfer-config '{
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TPUCPUOffloadingSpec",
                "spec_module_path": "tpu_inference.offload.cpu_tpu"
            }
        }'

`--kv-offloading-size N` reserves N GiB of host RAM for the pool. Sizing
guidance: pool large enough to fit the working set of repeated prefixes,
but small enough that GPU_MEM_UTIL still leaves usable HBM after the H2D
staging buffer reserve (see `estimate_hbm_reserve_bytes`).

A working launcher script lives at
`examples/kv-offload/run_kv_offload_single_host.sh`.

Environment knobs (all optional)
--------------------------------
- `KV_OFFLOAD_LAZY_STORE` (default "0")
    "1" = only offload blocks at the HBM eviction frontier (recommended
    for prefix-cache-heavy workloads).
    "0" = legacy eager double-write on every newly computed block.
- `KV_H2D_POOL_MAX_BLOCKS` (default 2048)
    Caps the H2D staging buffer's `(max_padded, …)` first dim. Rounded
    to the next power of 2. Larger = fewer chunked loads on long-prefix
    workloads but more HBM reserved up front.
- `MAX_H2D_PREWARM_BLOCKS` (default 2048)
    Caps the number of power-of-2 shapes the H2D Pallas scatter HLO is
    pre-compiled for at engine init.
- `MAX_D2H_PREWARM_BLOCKS` (default 64)
    Same, for D2H Pallas gather.
- `TPU_PREMAPPED_BUFFER_SIZE` (libtpu env)
    Bumping to 8 GiB is required for `--kv-offloading-size` above ~350
    GiB; otherwise the lazy `copy_` falls back to slower per-call pinning.
"""
from __future__ import annotations

import os
import queue as _queue
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch_tpu._internal import pallas as tpu_pallas
from torch_tpu._internal.sync import synchronize as _tpu_sync
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.mediums import (BlockIDsLoadStoreSpec,
                                        CPULoadStoreSpec, GPULoadStoreSpec)
from vllm.v1.kv_offload.spec import CanonicalKVCaches
from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids
from vllm.v1.kv_offload.worker.worker import (OffloadingHandler,
                                              TransferResult, TransferSpec)

from tpu_inference.logger import init_logger
from tpu_inference.offload.pallas_kv_gather import pallas_kv_gather
from tpu_inference.offload.pallas_kv_scatter import pallas_kv_scatter

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level Pallas kernel bridges (compiled once per process).
# ---------------------------------------------------------------------------
_scatter_fn = tpu_pallas.custom_jax_kernel(pallas_kv_scatter)
_gather_fn = tpu_pallas.custom_jax_kernel(pallas_kv_gather)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_power_of_2(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (padded_ids, original_len) padded to the next power of two.

    Repeated final element fills the padding so scattered writes are
    idempotent.  The caller must use `original_len` to trim results when
    needed.
    """
    n = len(ids)
    if n == 0:
        return ids, 0
    p = 1
    while p < n:
        p <<= 1
    if p == n:
        return ids, n
    padded = np.empty(p, dtype=ids.dtype)
    padded[:n] = ids
    padded[n:] = ids[n - 1]
    return padded, n


# ---------------------------------------------------------------------------
# Background DMA worker
# ---------------------------------------------------------------------------


class _DmaWorker:
    """Single background thread draining a FIFO work queue.

    One instance per direction (D2H / H2D). Each `submit(fn, *args)` enqueues
    work and returns a `threading.Event` that is set when `fn` returns. The
    worker blocks on `queue.get()` when idle, and the FIFO queue guarantees
    serial execution — combined with `event.synchronize()` inside the task,
    this guarantees only one DMA per direction is in flight at any moment.
    """

    def __init__(self, name: str) -> None:
        self._queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run,
                                        name=name,
                                        daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> threading.Event:
        done = threading.Event()
        self._queue.put((fn, args, done))
        return done

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, args, done = item
            try:
                fn(*args)
            except Exception:
                logger.exception("[kv-offload] DMA worker task failed")
            finally:
                done.set()

    def shutdown(self) -> None:
        self._queue.put(None)


# ---------------------------------------------------------------------------
# Transfer record
# ---------------------------------------------------------------------------


@dataclass
class Transfer:
    job_id: int
    num_bytes: int
    n: int
    # threading.Event set by _DmaWorker when the DMA task returns. Because
    # the task ends with `_tpu_sync(wait=True)` (D2H) or a blocking
    # `d.copy_(h)` (H2D), dma_done.is_set() implies the DMA is physically
    # complete on-device. For H2D, dma_done starts as None until
    # `_start_h2d_dma` kicks the worker — deferred submission keeps only one
    # H2D DMA in flight at any time, so the single _h2d_device_buffer is
    # never overwritten while the previous transfer's scatter is still
    # reading it.
    dma_done: threading.Event | None
    # Pinned CPU staging used as the DMA endpoint in BOTH directions.
    # D2H: filled by the worker's blocking copy_; main-thread scatter into
    #      cpu_pool happens after dma_done.is_set() flips True.
    # H2D: filled by main-thread index_select(cpu_pool, padded_src_ids);
    #      consumed by the worker's H2D copy_ into device_buffer.
    host_buffer: list | None = None
    # H2D: device staging (slice of the single _h2d_device_buffer) that
    # holds the freshly DMA'd blocks until the Pallas scatter copies them
    # into kv_cache. D2H: holds the Pallas gather output until the worker's
    # D2H copy_ drains it; kept alive so XLA's gather IR isn't dropped early.
    # For H2D, set lazily by `_start_h2d_dma` (None at register time).
    device_buffer: list | None = None
    # D2H: numpy int64 cpu_pool scatter indices, length n.
    # H2D: numpy int64 KV-cache scatter indices (full, unpadded). Sliced
    # per-chunk by `_start_h2d_dma`.
    dst_ids: np.ndarray | None = None
    # H2D: numpy int64 host pool gather indices (full, unpadded). Sliced
    # per-chunk by `_start_h2d_dma`.
    src_ids: np.ndarray | None = None
    # H2D: total number of chunks the transfer is split into. = 1 for
    # transfers that fit the single device buffer; > 1 for oversized loads
    # that are streamed through the buffer one chunk per get_finished()
    # round-trip. Set at transfer_async time, never mutated thereafter.
    chunks_total: int = 0
    # H2D: number of chunks already dispatched-and-scattered. Incremented
    # in get_finished() after each chunk's Pallas scatter syncs.
    chunks_done: int = 0
    # H2D: current chunk's padded dst_ids on device, used by get_finished()
    # for the Pallas scatter. Rebuilt for each chunk in `_start_h2d_dma`.
    dst_ids_i32: object | None = None
    # Set by the DMA worker task if it raises. get_finished() reads this to
    # report success=False.
    error: BaseException | None = None


# ---------------------------------------------------------------------------
# Single-direction handler
# ---------------------------------------------------------------------------


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """Handles KV-cache transfers for one direction (D2H or H2D) on TPU."""

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        assert len(src_tensors) == len(dst_tensors)

        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors

        min_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor = src_block_size_factor // min_factor
        self.dst_block_size_factor = dst_block_size_factor // min_factor

        self.block_size_in_bytes = [
            t.element_size() * t.stride(0) * min_factor for t in src_tensors
        ]
        self.total_block_size_in_bytes = sum(self.block_size_in_bytes)

        self.tpu_to_cpu: bool = src_tensors[0].device.type != "cpu"
        self.transfer_type = ("GPU", "CPU") if self.tpu_to_cpu else ("CPU",
                                                                     "GPU")
        self._device: torch.device = (src_tensors[0].device if self.tpu_to_cpu
                                      else dst_tensors[0].device)

        self._transfer_map: dict[int, Transfer] = {}
        self._transfers: deque[Transfer] = deque()

        # Lightweight diagnostics — printed sparingly to spot stalls.

        # Single background DMA worker for this direction. The FIFO queue
        # plus the worker task's terminating `_tpu_sync(..., wait=True)`
        # together guarantee that only one DMA per direction is in flight
        # at any instant. Thread isolation replaces what a dedicated TPU
        # stream + event.synchronize() would do on CUDA — no explicit
        # stream context is needed.
        worker_name = ("kv-offload-d2h-worker"
                       if self.tpu_to_cpu else "kv-offload-h2d-worker")
        self._dma_worker = _DmaWorker(name=worker_name)

        # H2D device-staging buffer. Single pre-allocated buffer at the max
        # padded shape; each H2D transfer takes a view [:n_padded].
        self._h2d_device_buffer: list[torch.Tensor] | None = None
        self._h2d_max_padded: int = 0
        if not self.tpu_to_cpu:
            ref = self.dst_tensors
            max_pool_blocks = int(
                os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "2048"))
            max_n = min(ref[0].shape[0], max_pool_blocks)
            max_padded = 1
            while max_padded < max_n:
                max_padded <<= 1
            self._h2d_max_padded = max_padded
            self._h2d_device_buffer = [
                torch.zeros(
                    (max_padded, ) + t.shape[1:],
                    dtype=t.dtype,
                    device=self._device,
                ) for t in ref
            ]
            # Force materialization so subsequent copy_ calls don't go
            # through a lazy-allocation path that could stall PJRT.
            _tpu_sync(self._h2d_device_buffer, wait=True)
            logger.info(
                "[kv-offload] H2D device buffer: %d blocks (max_padded)",
                max_padded,
            )

        # Pre-compile Pallas kernels for all power-of-2 shapes to avoid
        # Python JAX tracing (GIL-holding) during inference.
        self._prewarm_kernels()

    # -- H2D deferred-submission management ----------------------------------

    def _start_h2d_dma(self, t: Transfer) -> None:
        """Kick the worker thread on the H2D DMA for transfer `t`.

        Builds the worker closure that:
          1. Allocates host_buffer and runs index_select(cpu_pool, src_ids)
             — i.e., the host-side gather that *was* on the main thread.
          2. Per-layer blocking copy_ from host_buffer → device_buffer.
        Both steps run inside the worker so transfer_async stays fast and
        only one H2D's host_buffer is ever alive (in-flight=1).

        Caller (always `_ensure_h2d_in_flight`) must guarantee no other H2D
        is in flight at the moment of call — i.e., the prior transfer's
        scatter completed synchronously in get_finished.

        For oversized transfers (chunks_total > 1), dispatches just the
        chunk indexed by `t.chunks_done`. get_finished() resets `dma_done`
        to None after each chunk's scatter so subsequent calls re-enter
        here for the next chunk.
        """
        chunk_size = self._h2d_max_padded
        start = t.chunks_done * chunk_size
        end = min(start + chunk_size, len(t.src_ids))
        src_chunk = t.src_ids[start:end]
        dst_chunk = t.dst_ids[start:end]

        # Per-chunk padding: non-last chunks already match a power-of-2
        # (chunk_size is rounded to one); only the last chunk of an
        # oversized transfer may need padding.
        src_chunk_padded, _ = _pad_to_power_of_2(src_chunk)
        dst_chunk_padded, _ = _pad_to_power_of_2(dst_chunk)
        n_chunk_padded = len(src_chunk_padded)

        t.dst_ids_i32 = (torch.from_numpy(dst_chunk_padded.astype(
            np.int32)).to(torch.int32).to(self._device))

        device_buffer = [
            full[:n_chunk_padded] for full in self._h2d_device_buffer
        ]
        src_tensors = self.src_tensors  # cpu_pool, captured for the closure

        def _h2d_task(devs: list, src_tensors: list, src_ids_padded):
            try:
                src_ids_torch = torch.from_numpy(src_ids_padded).to(
                    torch.int64)
                n_padded = len(src_ids_padded)
                hosts = []
                for cpu_pool in src_tensors:
                    h = torch.empty(
                        (n_padded, ) + cpu_pool.shape[1:],
                        dtype=cpu_pool.dtype,
                    )
                    torch.index_select(cpu_pool, 0, src_ids_torch, out=h)
                    hosts.append(h)
                # Per-layer BLOCKING copy_. Hits the prewarmed single-layer
                # HLO; the combined N-layer compile was observed to trigger
                # a SparseCore UserFatal, so we keep the per-layer dispatch.
                for d, h in zip(devs, hosts):
                    d.copy_(h)
            except BaseException as e:
                t.error = e
                raise

        t.device_buffer = device_buffer
        t.dma_done = self._dma_worker.submit(_h2d_task, device_buffer,
                                             src_tensors, src_chunk_padded)

    def _ensure_h2d_in_flight(self) -> None:
        """Start the H2D DMA for the next pending transfer iff nothing is
        currently in-flight.

        Scans `self._transfers` in order. If the head transfer already has
        `dma_done` set on it, an H2D is already in flight and this is a
        no-op. Otherwise kick that transfer.
        """
        for t in self._transfers:
            if t.dma_done is not None:
                return  # already in-flight
            self._start_h2d_dma(t)
            return

    # -- Diagnostics ----------------------------------------------------------

    # -- Pallas kernel prewarm ------------------------------------------------

    @property
    def prewarm_shapes(self) -> list[int]:
        """Power-of-2 block-count shapes for H2D scatter prewarm.

        Capped at MAX_H2D_PREWARM_BLOCKS (default 2048) because the combined
        64-layer scatter HLO compilation time grows super-linearly with shape:
        p=2048 ~13s, p=4096 ~35s, p=8192 >60s (exceeds shm_broadcast timeout).
        """
        if self.tpu_to_cpu:
            return []
        max_prewarm = int(os.environ.get("MAX_H2D_PREWARM_BLOCKS", "2048"))
        max_n = min(self.dst_tensors[0].shape[0], max_prewarm)
        max_shape = 1
        while max_shape < max_n:
            max_shape <<= 1
        shapes, p = [], 1
        while p <= max_shape:
            shapes.append(p)
            p <<= 1
        return shapes

    def prewarm_shape(self, p: int) -> None:
        """Pre-compile H2D Pallas scatter, d.copy_(h), and id .to(device).

        Three compile artifacts per shape:
          1. Pallas scatter (tt_jit_custom_kernel)
          2. d.copy_(h) (tt_jit_copy__copy_from_as_strided_inverse)
          3. ids.to(self._device) (tt_jit_to_copy_copy__copy_from_as_strided)
        """
        assert not self.tpu_to_cpu, "prewarm_shape is for H2D only"
        dummy_ids = torch.zeros(p, dtype=torch.int32, device=self._device)
        dummy_devs = [
            torch.empty((p, ) + kv_l.shape[1:],
                        dtype=kv_l.dtype,
                        device=self._device) for kv_l in self.dst_tensors
        ]
        # (1) Pallas scatter prewarm.
        dummy_kv = [torch.zeros_like(kv_l) for kv_l in self.dst_tensors]
        for kv_d, d in zip(dummy_kv, dummy_devs):
            kv_d[:] = _scatter_fn(kv_d, d, dummy_ids)
        _tpu_sync(dummy_kv, wait=True)
        # (2) H2D copy_ prewarm — use an actual SLICE of _h2d_device_buffer
        #     so the storage / stride / view pattern matches runtime exactly.
        if self._h2d_device_buffer and p <= self._h2d_max_padded:
            slice_d = self._h2d_device_buffer[0][:p]
            slice_h = torch.empty(slice_d.shape, dtype=slice_d.dtype)
            slice_d.copy_(slice_h)
            _tpu_sync([slice_d], wait=True)
            del slice_d, slice_h
        # (3) id .to(self._device) prewarm — matches runtime line ~616.
        _ = (torch.from_numpy(np.zeros(p, dtype=np.int32)).to(torch.int32).to(
            self._device))
        del dummy_devs, dummy_kv
        logger.debug("[kv-offload] Pre-warmed H2D for shape p=%d", p)

    def _prewarm_kernels(self) -> None:
        """Pre-compile D2H gather Pallas + h.copy_(d) for power-of-2 shapes.

        H2D scatter is pre-warmed separately via prewarm_shape() because each
        shape needs its own 60-second shm_broadcast window.

        ROI-targeted copy_ prewarm: single-layer dummy tensors only — each
        layer shares the same per-layer shape, so one prewarm per p covers
        every layer. Runtime D2H hits exactly one unique
        (shape, dtype, stride) per p across all TP workers.
        """
        if not self.tpu_to_cpu:
            return  # H2D prewarm is deferred to prewarm_shape()

        # Cap at MAX_D2H_PREWARM_BLOCKS (default 64). Runtime D2H tends to
        # only hit small p; unbounded prewarm up to the full TPU KV cache
        # size caused HBM bloat / OOM.
        max_prewarm = int(os.environ.get("MAX_D2H_PREWARM_BLOCKS", "64"))
        max_n = min(self.src_tensors[0].shape[0], max_prewarm)
        max_shape = 1
        while max_shape < max_n:
            max_shape <<= 1

        count = 0
        p = 1
        while p <= max_shape:
            dummy_ids = torch.zeros(p, dtype=torch.int32, device=self._device)
            # (1) Pallas gather prewarm — all layers (one HLO per p).
            all_gather = [
                _gather_fn(kv_l, dummy_ids) for kv_l in self.src_tensors
            ]
            # (2) D2H copy_ prewarm — single layer only. copy_ will block
            #     until the gather materializes, so no explicit sync needed.
            single_d = all_gather[0]
            single_h = torch.empty(single_d.shape, dtype=single_d.dtype)
            single_h.copy_(single_d)
            # (3) id .to(self._device) prewarm — matches runtime line ~532.
            _ = (torch.from_numpy(np.zeros(p, dtype=np.int32)).to(
                torch.int32).to(self._device))
            # Drop refs so the gather HBM is freed before next iteration.
            del all_gather, single_d, single_h
            count += 1
            p <<= 1

        logger.debug(
            "[kv-offload] Pre-warmed D2H gather+copy for %d power-of-2 "
            "shapes (max_shape=%d, max_n=%d)",
            count,
            max_shape,
            max_n,
        )

    # -- OffloadingHandler interface -----------------------------------------

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_count = src_blocks.size * self.src_block_size_factor
        dst_sub_count = dst_blocks.size * self.dst_block_size_factor
        src_skip = -dst_blocks.size % self.src_block_size_factor
        assert dst_sub_count == src_sub_count - src_skip

        src_expanded = np.empty(src_sub_count, dtype=np.int64)
        dst_expanded = np.empty(dst_sub_count, dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_expanded,
            skip_count=src_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_expanded)
        src_ids = src_expanded[src_skip:]
        dst_ids = dst_expanded
        n = len(dst_ids)

        if self.tpu_to_cpu:
            # ── D2H: Pallas gather (main) → lazy copy_ + sync (worker thread)
            #         → main-thread scatter into cpu_pool in get_finished()
            #         once dma_done.is_set(). ──
            padded_src_ids, _ = _pad_to_power_of_2(src_ids)
            src_ids_i32 = (torch.from_numpy(padded_src_ids.astype(
                np.int32)).to(torch.int32).to(self._device))

            # (1) Compute stream (main thread): Pallas gather + flush IR.
            # Stays on the main thread because moving it would force XLA
            # to coordinate kv_l (live KV cache) reads across threads.
            device_buffer = [
                _gather_fn(kv_l, src_ids_i32) for kv_l in self.src_tensors
            ]
            _tpu_sync(device_buffer, wait=False)

            # (2) Unpinned host staging matching the gathered device shape.
            # The lazy D2H copy_ below stages through libtpu's premap pool
            # (TPU_PREMAPPED_BUFFER_SIZE), so the dst doesn't need to be
            # pinned ourselves — saves the per-call pin-allocation cost.
            host_buffer = [
                torch.empty(db.shape, dtype=db.dtype) for db in device_buffer
            ]

            # (3) Submit the actual DMA work to the per-direction worker.
            # The task does device -> host copy AND the cpu_pool host
            # scatter. dma_done is signaled only after both are complete.
            cpu_pool_tensors = self.dst_tensors
            dst_ids_for_scatter = dst_ids[:n]
            n_for_scatter = n

            # Build the Transfer first so the worker task closure can stash
            # any exception on it; success=False is then reported from
            # get_finished() instead of silently masking a failed DMA.
            t = Transfer(
                job_id=job_id,
                num_bytes=n * self.total_block_size_in_bytes,
                n=n,
                dma_done=None,
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                dst_ids=dst_ids[:n],
            )

            def _d2h_task(devs: list, hosts: list):
                try:
                    for d, h in zip(devs, hosts):
                        h.copy_(d)
                    for h, dst in zip(hosts, cpu_pool_tensors):
                        dst[dst_ids_for_scatter] = h[:n_for_scatter]
                except BaseException as e:
                    t.error = e
                    raise

            t.dma_done = self._dma_worker.submit(_d2h_task, device_buffer,
                                                 host_buffer)

        else:
            # H2D: host index_select (main) → lazy copy_ + sync (worker)
            #      → Pallas scatter (main) in get_finished().
            # Split into max_padded-sized chunks so the depth=1 device
            # buffer is never overflowed. Oversized transfers stream one
            # chunk per get_finished() round-trip; chunk splitting +
            # padding happens lazily in `_start_h2d_dma`. Each chunk's
            # Pallas scatter completes before the next chunk's DMA starts,
            # preserving the depth=1 buffer-reuse invariant.
            chunk_size = self._h2d_max_padded
            chunks_total = max(1,
                               (len(src_ids) + chunk_size - 1) // chunk_size)
            if chunks_total > 1:
                logger.info_once(
                    "[kv-offload] H2D transfer of %d blocks exceeds device "
                    "buffer (%d); will be loaded as %d chunks across that "
                    "many engine steps.", len(src_ids), chunk_size,
                    chunks_total)

            t = Transfer(
                job_id=job_id,
                num_bytes=n * self.total_block_size_in_bytes,
                n=n,
                dma_done=None,
                host_buffer=None,  # populated by _h2d_task
                device_buffer=None,
                src_ids=src_ids,
                dst_ids=dst_ids,
                chunks_total=chunks_total,
                chunks_done=0,
            )

        self._transfer_map[job_id] = t
        self._transfers.append(t)
        if not self.tpu_to_cpu:
            self._ensure_h2d_in_flight()
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        if self._transfers:
            pass
        while self._transfers:
            t = self._transfers[0]

            # For H2D, dma_done is None until `_start_h2d_dma` kicks the
            # worker (deferred submission). If we're at the head and no DMA
            # has been started yet, kick it now and bail until next poll —
            # we don't want to block the main thread waiting for the DMA.
            if t.dma_done is None:
                if not self.tpu_to_cpu:
                    self._ensure_h2d_in_flight()
                break
            # dma_done is set by _DmaWorker after the task's terminating
            # _tpu_sync(wait=True) (D2H) or blocking copy_ (H2D) returns —
            # i.e., the DMA is physically complete on-device.
            if not t.dma_done.is_set():
                break

            if self.tpu_to_cpu:
                # D2H: dma_done.is_set() (verified above) means the worker
                # task ran both the device -> host copy AND the cpu_pool host
                # scatter. The main thread only has to drop buffer refs.
                self._transfers.popleft()
                t.host_buffer = None
                t.device_buffer = None
            else:
                # H2D: dma_done.is_set() (verified above) means the worker
                # task ran the host->device copy_ to completion, so the
                # current chunk's bytes are in device_buffer on-device.
                # Dispatch the Pallas scatter and wait for it synchronously
                # — this is what lets the single _h2d_device_buffer be
                # safely reused by the next chunk (or next transfer).
                # Combined all-layer scatter HLO is prewarmed in
                # prewarm_shape().
                if t.error is None:
                    for kv_cache_l, dev_buf in zip(self.dst_tensors,
                                                   t.device_buffer):
                        kv_cache_l[:] = _scatter_fn(kv_cache_l, dev_buf,
                                                    t.dst_ids_i32)
                    # wait=True: block the main thread until the scatter
                    # HLO has finished reading device_buffer. Required for
                    # the depth=1 buffer-reuse invariant.
                    _tpu_sync(list(self.dst_tensors), wait=True)
                t.device_buffer = None
                t.host_buffer = None
                t.chunks_done += 1
                # More chunks pending → leave the transfer at the head and
                # kick its next chunk via _ensure_h2d_in_flight. The next
                # get_finished() call will scatter that chunk. Skips the
                # TransferResult / pop until the final chunk lands so the
                # scheduler doesn't see the load as complete prematurely.
                if t.error is None and t.chunks_done < t.chunks_total:
                    t.dma_done = None
                    self._ensure_h2d_in_flight()
                    break
                self._transfers.popleft()
                # Scatter completed (or transfer errored) → the single device
                # buffer is now safe to reuse. Kick the next pending H2D
                # so the worker thread overlaps with main-thread work.
                self._ensure_h2d_in_flight()

            results.append(
                TransferResult(
                    job_id=t.job_id,
                    success=(t.error is None),
                    transfer_size=t.num_bytes,
                    # TODO: elapsed_time() not yet implemented on TpuEvent.
                    transfer_time=1e-9,
                    transfer_type=self.transfer_type,
                ))
            del self._transfer_map[t.job_id]
        if results:
            pass
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Block until the named transfers have completed.

        Mirrors the CUDA reference impl in
        vllm/v1/kv_offload/cpu/gpu_worker.py:wait — purely a flush
        primitive. Internal-state cleanup (popping from `_transfers`,
        deleting from `_transfer_map`, building TransferResults) is left
        to the next `get_finished()` call. Removing transfers here would
        prevent `get_finished()` from emitting their TransferResults,
        which the scheduler relies on to mark jobs complete.
        """
        for jid in job_ids:
            t = self._transfer_map.get(jid)
            if t is None:
                continue
            if self.tpu_to_cpu:
                # Worker task ends with the cpu_pool host scatter, so
                # dma_done.is_set() implies the D2H is fully landed.
                t.dma_done.wait()
            else:
                # H2D skips non-ready transfers rather than waiting; the
                # OffloadingConnector only ever calls wait() for D2H flush.
                raise AssertionError(
                    f"wait() called on H2D handler for job={jid}; "
                    "OffloadingConnector should never invoke this path.")


# ---------------------------------------------------------------------------
# Factory: allocates CPU pinned tensors, creates bidirectional handlers
# ---------------------------------------------------------------------------


class CpuTpuOffloadingHandlers:
    """Mirrors CpuGpuOffloadingHandlers for TPU.

    TPU KV-cache tensors always have block dimension at index 0:
        shape = (num_kernel_blocks, block_size, num_kv_heads, head_size)
    There is no split_k_and_v variant as in the GPU paged-attention backend.
    """

    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        kv_caches: "CanonicalKVCaches",
        kernel_block_size: int,
        kv_dtype: torch.dtype,
        per_block_shape: tuple[int, ...],
    ):
        """
        Args:
          gpu_block_size: scheduler block size (vllm config block_size).
          cpu_block_size: offloaded block size (gpu_block_size * factor).
          num_cpu_blocks: number of CPU-side scheduler blocks.
          kv_caches: vllm 0.19 CanonicalKVCaches — each .tensor is a 2D int8
              view (num_blocks, page_size_bytes) sharing storage with the
              model's kv_cache.
          kernel_block_size: physical block_size in the original 5D Pallas
              layout (typically 16). gpu_block_size must be a multiple of it.
          kv_dtype: original dtype of the kv_cache (bf16, fp8, ...).
          per_block_shape: per-block trailing dims in the 5D Pallas layout
              (e.g. (block_size, num_kv_heads_x2 // kv_packing, kv_packing,
              padded_head_size)) — everything after the leading num_blocks
              dim.
        """
        assert kv_caches.tensors
        assert cpu_block_size % gpu_block_size == 0
        assert gpu_block_size % kernel_block_size == 0

        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

        tpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []

        logger.debug(
            "[kv-offload] Allocating %d unpinned CPU pool tensors "
            "(num_cpu_blocks=%d, gpu_factor=%d, cpu_factor=%d)",
            len(kv_caches.tensors),
            num_cpu_blocks,
            gpu_block_size_factor,
            cpu_block_size_factor,
        )
        total_bytes = 0
        for kv_cache_tensor in kv_caches.tensors:
            int8_view = kv_cache_tensor.tensor
            num_blocks = int8_view.shape[0]
            # Rebuild the original 5D Pallas view from the same storage.
            # vllm canonicalized to (num_blocks, page_size_bytes) int8 via
            # .set_(storage).view(num_blocks, page_size_bytes); we round-trip
            # to the kv_dtype-typed 5D shape so Pallas gather/scatter can
            # operate on it directly.
            tpu_tensor = (torch.empty(
                0, dtype=kv_dtype, device=int8_view.device).set_(
                    int8_view.untyped_storage()).view((num_blocks, ) +
                                                      tuple(per_block_shape)))
            cpu_shape = (num_cpu_kernel_blocks, ) + tuple(per_block_shape)
            # cpu_pool is intentionally NOT pinned. The DMA-capable host
            # staging buffer is allocated per-transfer (pinned) in
            # register_store/register_load, and the host
            # gather/scatter step copies between cpu_pool and that pinned
            # staging buffer. Pinning the full cpu_pool would lock hundreds
            # of GB and starve the libtpu premap region.
            cpu_tensor = torch.zeros(
                cpu_shape,
                dtype=kv_dtype,
                device="cpu",
            )
            total_bytes += cpu_tensor.element_size() * cpu_tensor.numel()
            tpu_tensors.append(tpu_tensor)
            cpu_tensors.append(cpu_tensor)

        logger.debug(
            "[kv-offload] Allocated %.2f GiB unpinned CPU pool",
            total_bytes / (1 << 30),
        )

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=tpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
        )
        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=tpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
        )


# ---------------------------------------------------------------------------
# Spec: subclasses CPUOffloadingSpec, overrides get_handlers() for TPU
# ---------------------------------------------------------------------------


class TPUCPUOffloadingSpec(CPUOffloadingSpec):
    """CPU offloading spec for TPU.

    Reuses CPUOffloadingSpec.__init__ (block-size calculation, eviction-policy
    config) and get_manager() (LRU/ARC setup).  Only get_handlers() is
    overridden to create TPU-specific transfer handlers.
    """

    def __init__(self, vllm_config: VllmConfig,
                 kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        self._tpu_handlers: CpuTpuOffloadingHandlers | None = None

    @property
    def prewarm_shapes(self) -> list[int]:
        """H2D Pallas-scatter shapes the runner should pre-compile.

        Delegates to the H2D handler, which is the actual owner of the
        Pallas kernels. Returns [] if get_handlers() hasn't run yet —
        TpuMultiprocExecutor calls this AFTER register_kv_caches, so the
        handler is set by then; the guard is defensive.
        """
        if self._tpu_handlers is None:
            return []
        return self._tpu_handlers.cpu_to_gpu_handler.prewarm_shapes

    def prewarm_shape(self, p: int) -> None:
        """Pre-compile H2D Pallas scatter for block-count `p`. Delegates
        to the H2D handler; no-op before get_handlers() has run."""
        if self._tpu_handlers is None:
            return
        self._tpu_handlers.cpu_to_gpu_handler.prewarm_shape(p)

    @classmethod
    def estimate_hbm_reserve_bytes(cls, vllm_config: VllmConfig) -> int:
        """Upper bound on HBM the H2D staging buffer will consume.

        Called by tpu_worker before profile_run to shrink the KV-cache
        budget so vllm leaves room for the SingleDirectionOffloadingHandler's
        _h2d_device_buffer, which is allocated lazily after
        determine_available_memory() returns.

        Only TPUCPUOffloadingSpec needs this reserve; other connectors
        (e.g., TPUConnector for P/D disagg) don't allocate an HBM staging
        buffer, so the worker must NOT shrink their budgets. Living on the
        spec class makes the gating explicit at the call site.
        """
        max_pool_blocks = int(os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "2048"))
        max_padded = 1
        while max_padded < max_pool_blocks:
            max_padded <<= 1

        from tpu_inference.layers.vllm.attention import (
            TPU_STR_DTYPE_TO_TORCH_DTYPE, PallasAttentionBackend)

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        # _resolve_kv_cache_dtype is strict on "auto", so resolve it here
        # the same way tpu_worker / tpu_runner already do.
        if cache_config.cache_dtype == "auto":
            model_dtype = model_config.dtype
            kv_dtype = (TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
                        if isinstance(model_dtype, str) else model_dtype)
        else:
            kv_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        per_block_bytes = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            cache_config.block_size,
            model_config.get_num_kv_heads(parallel_config),
            model_config.get_head_size(),
            kv_dtype,
        )
        # Upper bound: vLLM may group multiple layers into one canonical
        # tensor, in which case the actual H2D buffer is smaller. Safe
        # direction (overestimate leaves a few GiB unused).
        num_layers = model_config.get_num_layers(parallel_config)
        return num_layers * max_padded * per_block_bytes

    def get_handlers(
        self,
        kv_caches: "CanonicalKVCaches",
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec],
                        OffloadingHandler]]:
        if self._tpu_handlers is None:
            from vllm.v1.kv_cache_interface import AttentionSpec

            from tpu_inference.layers.vllm.attention import \
                PallasAttentionBackend

            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor

            # All kv_cache_groups must share the same attention spec so the
            # rebuilt 5D view is consistent across canonical tensors.
            kv_specs = [
                g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
                if isinstance(g.kv_cache_spec, AttentionSpec)
            ]
            assert kv_specs, "TPU offloading requires at least one AttentionSpec group"
            spec0 = kv_specs[0]
            for s in kv_specs[1:]:
                assert (
                    s.block_size == spec0.block_size
                    and s.num_kv_heads == spec0.num_kv_heads
                    and s.head_size == spec0.head_size
                    and s.dtype == spec0.dtype
                ), ("TPU offloading assumes a single attention spec across "
                    "all KV cache groups")

            full_5d_shape = PallasAttentionBackend.get_kv_cache_shape(
                num_blocks=1,
                block_size=spec0.block_size,
                num_kv_heads=spec0.num_kv_heads,
                head_size=spec0.head_size,
                cache_dtype_str=spec0.dtype,
            )
            per_block_shape = tuple(full_5d_shape[1:])

            self._tpu_handlers = CpuTpuOffloadingHandlers(
                gpu_block_size=gpu_block_size,
                cpu_block_size=offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                kv_caches=kv_caches,
                kernel_block_size=spec0.block_size,
                kv_dtype=spec0.dtype,
                per_block_shape=per_block_shape,
            )

        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._tpu_handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._tpu_handlers.cpu_to_gpu_handler,
        )
