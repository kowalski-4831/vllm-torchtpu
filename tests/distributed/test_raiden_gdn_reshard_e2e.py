# SPDX-License-Identifier: Apache-2.0
"""GDN (mamba) layer resharding plan generation, admission, and transfer
with raiden.
"""

from __future__ import annotations

import ctypes

import pytest

from vllm_torchtpu.distributed.kv_transfer.v2.common import (TAG_GDN_CONV,
                                                             TAG_GDN_SSM)
from vllm_torchtpu.distributed.kv_transfer.v2.mamba_raiden_plan import (
    GdnPushEntry, GdnShardGeometry, build_gdn_reshard_entries,
    validate_entries_against_geometry, validate_geometry_against_manifest)
from vllm_torchtpu.distributed.kv_transfer.v2.raiden_pool_manifest import (
    GdnHeadGeometry, build_qwen35_pool_manifest)

from .tpu_connector_v2_test_utils import FakeTensor

kv_cache_manager = pytest.importorskip(
    "tpu_raiden.api.torch.kv_cache_manager",
    reason="tpu_raiden host extension is not on PYTHONPATH")
raiden_service_pb2 = pytest.importorskip("tpu_raiden.rpc.raiden_service_pb2")

NUM_GDN_LAYERS = 2
NUM_BLOCKS = 4
SRC_BLOCK_ID = 2
DST_BLOCK_ID = 3
UUID_BASE = 0x6D35
POISON = 0xAA

SEGMENT_CODES = {"q": 1, "k": 2, "v": 3, "ssm": 4}


def qwen35_geometry(tp_size: int) -> GdnShardGeometry:
    """Qwen3.5-397B-A17B GDN geometry."""
    return GdnShardGeometry(
        tp_size=tp_size,
        taps=3,
        total_key_heads=16,
        total_value_heads=64,
        key_head_dim=128,
        value_head_dim=128,
        conv_itemsize=2,
        ssm_itemsize=4,
    )


def _pattern(layer: int, segment: str, global_head: int, tap: int,
             byte_index: int) -> int:
    return (layer * 31 + SEGMENT_CODES[segment] * 17 + global_head * 7 +
            tap * 3 + byte_index) % 251


# One head's payload is the byte sequence (base + i) % 251 — a cyclic slice
# of 0..250, precomputed so multi-MB ssm blocks fill fast.
_PATTERN_CYCLE = bytes(range(251)) * (65536 // 251 + 2)


def _head_payload(layer: int, segment: str, global_head: int, tap: int,
                  nbytes: int) -> bytes:
    start = _pattern(layer, segment, global_head, tap, 0)
    return _PATTERN_CYCLE[start:start + nbytes]


def _conv_block_bytes(geometry: GdnShardGeometry, rank: int,
                      layer: int) -> bytes:
    """One rank's conv block, rows of [q | k | v] built from global heads."""
    payload = bytearray()
    for tap in range(geometry.taps):
        for segment, heads, head_bytes in (
            ("q", geometry.key_range(rank), geometry.key_head_bytes),
            ("k", geometry.key_range(rank), geometry.key_head_bytes),
            ("v", geometry.value_range(rank), geometry.value_head_bytes),
        ):
            for global_head in heads:
                payload.extend(
                    _head_payload(layer, segment, global_head, tap,
                                  head_bytes))
    assert len(payload) == geometry.conv_live_bytes
    return bytes(payload)


def _ssm_block_bytes(geometry: GdnShardGeometry, rank: int,
                     layer: int) -> bytes:
    payload = bytearray()
    for global_head in geometry.value_range(rank):
        payload.extend(
            _head_payload(layer, "ssm", global_head, 0,
                          geometry.ssm_head_bytes))
    assert len(payload) == geometry.ssm_live_bytes
    return bytes(payload)


def _new_host_manager(*, num_layers: int, slice_bytes: int, node_id: int,
                      host_blocks: int):
    return kv_cache_manager.KVCacheManager.create_host_only_for_testing(
        num_layers=num_layers,
        num_shards=1,
        slice_byte_size=slice_bytes,
        node_id=node_id,
        local_port=0,
        host_blocks=host_blocks,
        parallelism=2,
    )


def _new_request(*,
                 uuid: int,
                 is_sender: bool,
                 req_id: str,
                 pool_dtype_tags=None):
    request = raiden_service_pb2.StartTransferRequest()
    request.uuid = int(uuid)
    request.is_sender = bool(is_sender)
    request.dst_mem_type = raiden_service_pb2.MEMORY_TYPE_DRAM
    request.use_block_chunks = True
    request.req_id = str(req_id)
    for tag in pool_dtype_tags or ():
        request.pool_dtype_tags.append(str(tag))
    return request


def _append_entry(schedule, entry: GdnPushEntry, *, dst_peer: str) -> None:
    proto_entry = schedule.entries.add()
    proto_entry.dst_peer = str(dst_peer)
    proto_entry.dst_shard_idx = 0
    proto_entry.src_block_id = SRC_BLOCK_ID
    proto_entry.dst_block_id = DST_BLOCK_ID
    proto_entry.src_offset_bytes = int(entry.src_offset_bytes)
    proto_entry.dst_offset_bytes = int(entry.dst_offset_bytes)
    proto_entry.size_bytes = int(entry.size_bytes)
    proto_entry.src_stride_bytes = int(entry.src_stride_bytes)
    proto_entry.dst_stride_bytes = int(entry.dst_stride_bytes)
    proto_entry.count = int(entry.count)


def _sender_request(*,
                    uuid,
                    tag,
                    entries,
                    src_rank,
                    dst_peers,
                    req_id,
                    pool_dtype_tags=None):
    """The plan one producer rank registers with ``is_sender=True``; the
    schedule key is its local shard index (0)."""
    request = _new_request(uuid=uuid,
                           is_sender=True,
                           req_id=req_id,
                           pool_dtype_tags=pool_dtype_tags)
    schedule = request.shard_push_schedules[0]
    for entry in entries:
        if entry.tag != tag or entry.src_rank != int(src_rank):
            continue
        _append_entry(schedule, entry, dst_peer=dst_peers[entry.dst_rank])
    return request


def _receiver_request(*,
                      uuid,
                      tag,
                      entries,
                      dst_rank,
                      dst_peers,
                      req_id,
                      pool_dtype_tags=None):
    """The plan one decode rank registers with ``is_sender=False``; schedule
    keys are the sender node ids — the admission step."""
    request = _new_request(uuid=uuid,
                           is_sender=False,
                           req_id=req_id,
                           pool_dtype_tags=pool_dtype_tags)
    for entry in entries:
        if entry.tag != tag or entry.dst_rank != int(dst_rank):
            continue
        schedule = request.shard_push_schedules[int(entry.src_rank)]
        _append_entry(schedule, entry, dst_peer=dst_peers[entry.dst_rank])
    return request


def _push_all(src_managers, entries, tag, uuid, dst_peers, *, layer_idx):
    """Every source rank pushes its share of the plan to every peer it
    feeds."""
    for src_rank, manager in enumerate(src_managers):
        peers = {
            dst_peers[entry.dst_rank]
            for entry in entries
            if entry.tag == tag and entry.src_rank == src_rank
        }
        for peer in sorted(peers):
            manager.push_registered_plan(uuid,
                                         peer, [SRC_BLOCK_ID], [DST_BLOCK_ID],
                                         layer_idx=layer_idx,
                                         parallelism=1)


def _register_plans(src_managers,
                    dst_managers,
                    entries,
                    tag,
                    uuid,
                    dst_peers,
                    *,
                    pool_dtype_tags=None):
    for src_rank, manager in enumerate(src_managers):
        manager.register_active_plan(
            uuid,
            _sender_request(uuid=uuid,
                            tag=tag,
                            entries=entries,
                            src_rank=src_rank,
                            dst_peers=dst_peers,
                            req_id=f"gdn-{tag}",
                            pool_dtype_tags=pool_dtype_tags),
            is_sender=True,
        )
    for dst_rank, manager in enumerate(dst_managers):
        manager.register_active_plan(
            uuid,
            _receiver_request(uuid=uuid,
                              tag=tag,
                              entries=entries,
                              dst_rank=dst_rank,
                              dst_peers=dst_peers,
                              req_id=f"gdn-{tag}",
                              pool_dtype_tags=pool_dtype_tags),
            is_sender=False,
        )


def _unregister_plans(managers, uuid):
    for manager in managers:
        try:
            manager.unregister_active_plan(uuid)
        except Exception:
            pass


class _FakeGroup:

    def __init__(self, layer_names):
        self.layer_names = tuple(layer_names)


def _gdn_manifest(geometry: GdnShardGeometry):
    """Builds the canonical Qwen3.5 pool manifest over fake typed GDN state
    tensors — one (conv, ssm) pair per layer."""
    named = {}
    groups = []
    for layer in range(NUM_GDN_LAYERS):
        name = f"model.layers.{layer}.linear_attn"
        conv = FakeTensor((NUM_BLOCKS, geometry.taps,
                           geometry.conv_row_bytes // geometry.conv_itemsize),
                          element_size=geometry.conv_itemsize,
                          dtype="torch.bfloat16")
        ssm = FakeTensor((NUM_BLOCKS, geometry.local_value_heads,
                          geometry.key_head_dim, geometry.value_head_dim),
                         element_size=geometry.ssm_itemsize,
                         dtype="torch.float32")
        named[name] = [conv, ssm]
        groups.append(_FakeGroup([name]))
    return build_qwen35_pool_manifest(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=(),
        gdn_geometry=GdnHeadGeometry(
            local_key_heads=geometry.local_key_heads,
            local_value_heads=geometry.local_value_heads,
            key_head_dim=geometry.key_head_dim,
            value_head_dim=geometry.value_head_dim,
        ),
    )


def _write_pool_block(manager, pool_idx: int, block_id: int,
                      payload: bytes) -> None:
    ref = manager.get_block_ref(pool_idx, block_id)
    assert len(payload) <= int(ref["block_stride_bytes"])
    ctypes.memmove(int(ref["ptr"]), bytes(payload), len(payload))


def _read_pool_block(manager, pool_idx: int, block_id: int,
                     nbytes: int) -> bytes:
    ref = manager.get_block_ref(pool_idx, block_id)
    return ctypes.string_at(int(ref["ptr"]), nbytes)


class TestExplicitPoolAdmission:

    def _make_managers(self, src: GdnShardGeometry, dst: GdnShardGeometry):
        managers = []
        for geometry, node_base in ((src, 0), (dst, 100)):
            manifest = _gdn_manifest(geometry)
            validate_geometry_against_manifest(geometry,
                                               manifest.geometry_by_tag())
            slice_bytes = max(pool.block_stride_bytes
                              for pool in manifest.pools)
            side = []
            for rank in range(geometry.tp_size):
                manager = _new_host_manager(num_layers=len(manifest.storages),
                                            slice_bytes=slice_bytes,
                                            node_id=node_base + rank,
                                            host_blocks=NUM_BLOCKS)
                manager.register_pools(manifest.pool_dicts())
                side.append(manager)
            managers.append(side)
        return managers[0], managers[1]

    def _pool_dtype_tags(self) -> list[str]:
        return ["bfloat16", "float32"] * NUM_GDN_LAYERS

    def _fill_sides(self, src_managers, dst_managers, src, dst):
        """Pattern-fill the source state block; poison every destination
        pool block."""
        conv_ids = src_managers[0].pool_ids_with_tag(TAG_GDN_CONV)
        ssm_ids = src_managers[0].pool_ids_with_tag(TAG_GDN_SSM)
        for rank, manager in enumerate(src_managers):
            for layer, (conv_pool,
                        ssm_pool) in enumerate(zip(conv_ids, ssm_ids)):
                _write_pool_block(manager, conv_pool, SRC_BLOCK_ID,
                                  _conv_block_bytes(src, rank, layer))
                _write_pool_block(manager, ssm_pool, SRC_BLOCK_ID,
                                  _ssm_block_bytes(src, rank, layer))
        for manager in dst_managers:
            for pool_ids, live in ((conv_ids, dst.conv_live_bytes),
                                   (ssm_ids, dst.ssm_live_bytes)):
                for pool_idx in pool_ids:
                    for block in range(NUM_BLOCKS):
                        _write_pool_block(manager, pool_idx, block,
                                          bytes([POISON]) * live)

    @pytest.mark.parametrize("src_tp,dst_tp", [(8, 2), (8, 1), (2, 8)])
    def test_admit_plan_reshard_state_byte_exact(self, src_tp, dst_tp):
        src = qwen35_geometry(src_tp)
        dst = qwen35_geometry(dst_tp)
        entries = build_gdn_reshard_entries(src, dst)
        validate_entries_against_geometry(entries, src, dst)
        src_managers, dst_managers = self._make_managers(src, dst)
        dst_peers = [manager.transfer_address for manager in dst_managers]
        self._fill_sides(src_managers, dst_managers, src, dst)

        uuid_of = {TAG_GDN_CONV: UUID_BASE + 41, TAG_GDN_SSM: UUID_BASE + 42}
        conv_ids = dst_managers[0].pool_ids_with_tag(TAG_GDN_CONV)
        ssm_ids = dst_managers[0].pool_ids_with_tag(TAG_GDN_SSM)
        try:
            for tag, uuid in uuid_of.items():
                _register_plans(src_managers,
                                dst_managers,
                                entries,
                                tag,
                                uuid,
                                dst_peers,
                                pool_dtype_tags=self._pool_dtype_tags())
                for pool_idx in src_managers[0].pool_ids_with_tag(tag):
                    _push_all(src_managers,
                              entries,
                              tag,
                              uuid,
                              dst_peers,
                              layer_idx=pool_idx)

            for dst_rank, manager in enumerate(dst_managers):
                for layer, (conv_pool,
                            ssm_pool) in enumerate(zip(conv_ids, ssm_ids)):
                    assert _read_pool_block(
                        manager, conv_pool, DST_BLOCK_ID,
                        dst.conv_live_bytes) == _conv_block_bytes(
                            dst, dst_rank,
                            layer), (f"conv mismatch dst_rank={dst_rank} "
                                     f"layer={layer}")
                    assert _read_pool_block(
                        manager, ssm_pool, DST_BLOCK_ID,
                        dst.ssm_live_bytes) == _ssm_block_bytes(
                            dst, dst_rank,
                            layer), (f"ssm mismatch dst_rank={dst_rank} "
                                     f"layer={layer}")
                    # A block no plan wrote stays poisoned.
                    assert _read_pool_block(manager, conv_pool, 0,
                                            dst.conv_live_bytes) == bytes(
                                                [POISON]) * dst.conv_live_bytes
        finally:
            for uuid in uuid_of.values():
                _unregister_plans(src_managers + dst_managers, uuid)

    def test_dtype_tag_mismatch_rejected_at_admission(self):
        src = qwen35_geometry(2)
        dst = qwen35_geometry(2)
        entries = build_gdn_reshard_entries(src, dst)
        _, dst_managers = self._make_managers(src, dst)
        dst_peers = [manager.transfer_address for manager in dst_managers]
        wrong_tags = ["float32", "bfloat16"] * NUM_GDN_LAYERS  # swapped
        request = _receiver_request(uuid=UUID_BASE + 77,
                                    tag=TAG_GDN_CONV,
                                    entries=entries,
                                    dst_rank=0,
                                    dst_peers=dst_peers,
                                    req_id="gdn-bad-dtype",
                                    pool_dtype_tags=wrong_tags)
        with pytest.raises(Exception, match="dtype tag mismatch"):
            dst_managers[0].register_active_plan(UUID_BASE + 77,
                                                 request,
                                                 is_sender=False)
