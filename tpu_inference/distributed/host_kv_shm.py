# SPDX-License-Identifier: Apache-2.0
"""Shared-memory KV staging pool for the TPUConnector TP>1 path.

Layout (single SharedMemory block per host):

    slot 0: [rank 0 layers back-to-back] [rank 1 layers ...] ... [rank N-1 ...]
    slot 1: ...
    ...

Each (slot, rank, layer) is a contiguous region of max_per_layer_bytes; the
actual number of valid blocks per request varies but is always <= max_blocks.
The wire-format header carries the actual block count so only the meaningful
bytes are transferred over the network; padding stays resident.

Free-list is held on rank-0 only. Other ranks compute offsets from
(slot_idx, tp_rank, layer_idx) and never touch the free-list state.
"""

import queue
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import torch

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


@dataclass(frozen=True)
class PoolSpec:
    """Describes the per-host pool. Must match byte-for-byte across ranks."""
    num_slots: int
    tp_size: int
    num_layers: int
    max_blocks: int
    # Shape of a single layer's shard for one rank at max_blocks.
    # E.g. [max_blocks, block_size, kv_heads_per_rank, 2, head_dim].
    layer_shard_shape: tuple
    dtype: torch.dtype

    @property
    def per_layer_bytes(self) -> int:
        n = 1
        for d in self.layer_shard_shape:
            n *= d
        return n * _dtype_bytes(self.dtype)

    @property
    def per_rank_bytes(self) -> int:
        return self.num_layers * self.per_layer_bytes

    @property
    def per_slot_bytes(self) -> int:
        return self.tp_size * self.per_rank_bytes

    @property
    def total_bytes(self) -> int:
        return self.num_slots * self.per_slot_bytes


class HostKVShmPool:
    """Per-host shared-memory pool. Rank 0 `create()`s, others `attach()`."""

    def __init__(self, spec: PoolSpec, shm: shared_memory.SharedMemory,
                 owner: bool):
        self.spec = spec
        self._shm = shm
        self._owner = owner
        # Only rank 0 uses the free list.
        self._free: Optional[queue.Queue] = None
        self._lock: Optional[threading.Lock] = None
        if owner:
            self._free = queue.Queue(maxsize=spec.num_slots)
            for i in range(spec.num_slots):
                self._free.put(i)
            self._lock = threading.Lock()

    @classmethod
    def create(cls, spec: PoolSpec, name: str) -> "HostKVShmPool":
        size = spec.total_bytes
        # Unlink any stale block from a previous crashed run. We only do this
        # on create (rank 0); other ranks must never unlink.
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            logger.warning("HostKVShmPool --> unlinked stale block %s", name)
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        logger.info(
            "HostKVShmPool --> created name=%s size=%.2fGB "
            "(num_slots=%d, tp_size=%d, num_layers=%d, per_slot=%.2fMB)", name,
            size / (1024**3), spec.num_slots, spec.tp_size, spec.num_layers,
            spec.per_slot_bytes / (1024**2))
        return cls(spec, shm, owner=True)

    @classmethod
    def attach(cls, spec: PoolSpec, name: str) -> "HostKVShmPool":
        shm = shared_memory.SharedMemory(name=name, create=False)
        if shm.size < spec.total_bytes:
            raise RuntimeError(
                f"HostKVShmPool attach: shm size {shm.size} < expected "
                f"{spec.total_bytes}. Spec mismatch between ranks?")
        logger.info("HostKVShmPool --> attached name=%s size=%.2fGB", name,
                    shm.size / (1024**3))
        return cls(spec, shm, owner=False)

    # ---- Free-list (rank 0 only) ---------------------------------------
    def acquire_slot(self, timeout: Optional[float] = None) -> int:
        assert self._owner, "acquire_slot: only rank 0 manages the free list"
        return self._free.get(timeout=timeout)

    def release_slot(self, slot_idx: int) -> None:
        assert self._owner, "release_slot: only rank 0 manages the free list"
        # Idempotent: tolerate double-release which can happen if a timeout
        # races a pull-done notification.
        with self._lock:
            if slot_idx in self._free.queue:
                return
            self._free.put_nowait(slot_idx)

    # ---- Tensor views ---------------------------------------------------
    def layer_view(self, slot_idx: int, rank: int, layer_idx: int,
                   num_blocks: int) -> torch.Tensor:
        """Return a CPU tensor view over the (slot, rank, layer) region.

        Shape is (num_blocks,) + layer_shard_shape[1:], so callers can copy
        the exact request-sized slice in/out without touching the padding.
        """
        spec = self.spec
        offset = (slot_idx * spec.per_slot_bytes + rank * spec.per_rank_bytes +
                  layer_idx * spec.per_layer_bytes)
        count = num_blocks
        for d in spec.layer_shard_shape[1:]:
            count *= d
        # frombuffer returns a flat tensor; reshape to the layer shard shape
        # with the actual block count on dim 0.
        flat = torch.frombuffer(self._shm.buf,
                                dtype=spec.dtype,
                                count=count,
                                offset=offset)
        view_shape = (num_blocks, ) + tuple(spec.layer_shard_shape[1:])
        return flat.view(view_shape)

    def _used_per_layer(self, num_blocks: int) -> int:
        """Bytes used by one layer's shard at the actual block count."""
        bytes_per_block = _dtype_bytes(self.spec.dtype)
        for d in self.spec.layer_shard_shape[1:]:
            bytes_per_block *= d
        return bytes_per_block * num_blocks

    def pack_rank(self, slot_idx: int, rank: int, num_blocks: int) -> bytes:
        """Copy the `num_blocks`-sized shard of each layer into a single
        contiguous bytes buffer (strips per-layer padding). Suitable as a ZMQ
        payload frame."""
        spec = self.spec
        rank_start = (slot_idx * spec.per_slot_bytes +
                      rank * spec.per_rank_bytes)
        used = self._used_per_layer(num_blocks)
        out = bytearray(used * spec.num_layers)
        pos = 0
        for li in range(spec.num_layers):
            layer_off = rank_start + li * spec.per_layer_bytes
            out[pos:pos + used] = self._shm.buf[layer_off:layer_off + used]
            pos += used
        return bytes(out)

    def rank_layer_views(self, slot_idx: int, rank: int,
                         num_blocks: int) -> list[memoryview]:
        """Return one memoryview per layer over the (slot, rank, layer)
        'used' slice of shm. No copy -- ZeroMQ can send these directly
        with copy=False, which turns the 4-copy path
        (shm->bytearray->bytes->zmq->kernel) into a single scatter-gather
        sendmsg from shm straight to the NIC buffer. The returned views
        alias live shm and must not outlive the current slot."""
        spec = self.spec
        rank_start = (slot_idx * spec.per_slot_bytes +
                      rank * spec.per_rank_bytes)
        used = self._used_per_layer(num_blocks)
        base = memoryview(self._shm.buf)
        out: list[memoryview] = []
        for li in range(spec.num_layers):
            layer_off = rank_start + li * spec.per_layer_bytes
            out.append(base[layer_off:layer_off + used])
        return out

    def unpack_rank_layers(self, slot_idx: int, rank: int, num_blocks: int,
                           layer_buffers: list) -> None:
        """Inverse of rank_layer_views. `layer_buffers` is a list of
        buffer-protocol objects (zmq.Frame.buffer, memoryview, bytes)
        sized per-layer `used` bytes, one per layer. Each is memcpy'd into
        its shm slot -- one copy on the consumer instead of two."""
        spec = self.spec
        if len(layer_buffers) != spec.num_layers:
            raise RuntimeError(
                f"unpack_rank_layers: got {len(layer_buffers)} layers, "
                f"expected {spec.num_layers}")
        rank_start = (slot_idx * spec.per_slot_bytes +
                      rank * spec.per_rank_bytes)
        used = self._used_per_layer(num_blocks)
        for li, buf in enumerate(layer_buffers):
            if len(buf) != used:
                raise RuntimeError(
                    f"unpack_rank_layers: layer {li} size {len(buf)} != "
                    f"expected {used} (num_blocks={num_blocks})")
            layer_off = rank_start + li * spec.per_layer_bytes
            self._shm.buf[layer_off:layer_off + used] = buf

    def unpack_rank(self, slot_idx: int, rank: int, num_blocks: int,
                    payload) -> None:
        """Inverse of pack_rank: scatter a received contiguous payload into
        the per-layer regions of this slot/rank."""
        spec = self.spec
        rank_start = (slot_idx * spec.per_slot_bytes +
                      rank * spec.per_rank_bytes)
        used = self._used_per_layer(num_blocks)
        expected = used * spec.num_layers
        if len(payload) != expected:
            raise RuntimeError(
                f"unpack_rank: payload size {len(payload)} != expected "
                f"{expected} (num_blocks={num_blocks})")
        mv = memoryview(payload)
        for li in range(spec.num_layers):
            layer_off = rank_start + li * spec.per_layer_bytes
            self._shm.buf[layer_off:layer_off +
                          used] = (mv[li * used:(li + 1) * used])

    # ---- Lifecycle ------------------------------------------------------
    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            logger.exception("HostKVShmPool close: error closing shm %s",
                             self._shm.name)
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
