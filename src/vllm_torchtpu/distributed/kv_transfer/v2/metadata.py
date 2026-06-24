from __future__ import annotations

from dataclasses import dataclass

from .common import (LayerType, check_non_negative, check_positive,
                     check_power_of_two, linear_rank)
from .layout import KVCacheRegion


@dataclass(frozen=True)
class KVParallelLayout:
    """KV parallel layout for full-attention and linear-attention cache."""

    full_attn_pcp_size: int
    full_attn_tp_size: int
    linear_attn_pcp_size: int
    linear_attn_tp_size: int

    def __post_init__(self) -> None:
        check_positive("full_attn_pcp_size", self.full_attn_pcp_size)
        check_positive("full_attn_tp_size", self.full_attn_tp_size)
        check_positive("linear_attn_pcp_size", self.linear_attn_pcp_size)
        check_positive("linear_attn_tp_size", self.linear_attn_tp_size)
        check_power_of_two("full_attn", self.full_attn_tp_size)
        check_power_of_two("linear_attn", self.linear_attn_tp_size)

    @property
    def full_attn_world_size(self) -> int:
        return self.full_attn_pcp_size * self.full_attn_tp_size

    @property
    def linear_attn_world_size(self) -> int:
        return self.linear_attn_pcp_size * self.linear_attn_tp_size

    def full_attn_ranks_for_tp_rank(self, tp_rank: int) -> tuple[int, ...]:
        return tuple(
            linear_rank(pcp_rank, tp_rank, self.full_attn_tp_size)
            for pcp_rank in range(self.full_attn_pcp_size))

    def linear_attn_ranks_for_tp_rank(self, tp_rank: int) -> tuple[int, ...]:
        return tuple(
            linear_rank(pcp_rank, tp_rank, self.linear_attn_tp_size)
            for pcp_rank in range(self.linear_attn_pcp_size))


@dataclass(frozen=True)
class ConnectorMetadataV2:
    """Scheduler-to-worker metadata for one completed prefill request."""

    req_id: int
    block_size: int
    kv_source_layout: KVParallelLayout
    kv_caches: dict[int, dict[str, KVCacheRegion]]
    fa_block_ids: tuple[int, ...]
    mamba_block_ids: tuple[int, ...]
    fa_num_tokens: int | None = None
    mamba_num_tokens: int | None = None

    def __post_init__(self) -> None:
        check_positive("block_size", self.block_size)
        object.__setattr__(self, "fa_block_ids", tuple(self.fa_block_ids))
        object.__setattr__(self, "mamba_block_ids",
                           tuple(self.mamba_block_ids))
        if self.fa_num_tokens is not None:
            check_non_negative("fa_num_tokens", self.fa_num_tokens)
        if self.mamba_num_tokens is not None:
            check_non_negative("mamba_num_tokens", self.mamba_num_tokens)


@dataclass(frozen=True)
class LocalDecodeAllocation:
    """Decode-side local destination selected during block allocation."""

    rank: int
    block_size: int
    kv_caches: dict[str, KVCacheRegion]
    fa_block_ids: tuple[int, ...]
    mamba_block_ids: tuple[int, ...]
    fa_num_tokens: int | None = None
    mamba_num_tokens: int | None = None

    def __post_init__(self) -> None:
        check_non_negative("rank", self.rank)
        check_positive("block_size", self.block_size)
        object.__setattr__(self, "fa_block_ids", tuple(self.fa_block_ids))
        object.__setattr__(self, "mamba_block_ids",
                           tuple(self.mamba_block_ids))
        if self.fa_num_tokens is not None:
            check_non_negative("fa_num_tokens", self.fa_num_tokens)
        if self.mamba_num_tokens is not None:
            check_non_negative("mamba_num_tokens", self.mamba_num_tokens)


@dataclass(frozen=True)
class TpKVTopology:
    """Local TP view used by a decode worker to build a pull plan."""

    local_layout: KVParallelLayout
    block_size: int
    tp_rank: int
    total_num_kv_heads: int
    total_num_mamba_heads: int

    def __post_init__(self) -> None:
        check_positive("block_size", self.block_size)
        check_non_negative("tp_rank", self.tp_rank)
        check_non_negative("total_num_kv_heads", self.total_num_kv_heads)
        check_non_negative("total_num_mamba_heads", self.total_num_mamba_heads)

    @property
    def full_attn_tp_rank(self) -> int:
        return self.tp_rank % self.local_layout.full_attn_tp_size

    @property
    def linear_attn_tp_rank(self) -> int:
        return self.tp_rank % self.local_layout.linear_attn_tp_size


@dataclass(frozen=True)
class HeadMapping:
    """One global head mapped from a producer-local head to a D-local head."""

    global_head: int
    source_head: int
    destination_head: int
    segment_name: str | None = None

    def __post_init__(self) -> None:
        check_non_negative("global_head", self.global_head)
        check_non_negative("source_head", self.source_head)
        check_non_negative("destination_head", self.destination_head)


@dataclass(frozen=True)
class SourceBlockRef:
    """One producer block with its global logical block index."""

    logical_block_index: int
    block_id: int

    def __post_init__(self) -> None:
        check_non_negative("logical_block_index", self.logical_block_index)
        check_non_negative("block_id", self.block_id)


@dataclass(frozen=True)
class PullMeta:
    """Logical P-rank/head relationship for the current D rank."""

    req_id: int
    p_ranks: tuple[int, ...]
    fa_heads_by_rank: dict[int, tuple[int, ...]]
    mamba_heads_by_rank: dict[int, tuple[int, ...]]
    fa_head_mappings_by_rank: dict[int, tuple[HeadMapping, ...]]
    mamba_head_mappings_by_rank: dict[int, tuple[HeadMapping, ...]]
    fa_source_block_ids: tuple[int, ...]
    mamba_source_block_ids: tuple[int, ...]
    fa_block_refs_by_rank: dict[int, tuple[SourceBlockRef, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_ranks", tuple(self.p_ranks))
        object.__setattr__(self, "fa_source_block_ids",
                           tuple(self.fa_source_block_ids))
        object.__setattr__(self, "mamba_source_block_ids",
                           tuple(self.mamba_source_block_ids))
        object.__setattr__(
            self,
            "fa_block_refs_by_rank",
            {
                rank: tuple(block_refs)
                for rank, block_refs in self.fa_block_refs_by_rank.items()
            },
        )


@dataclass(frozen=True)
class StridedSegmentOp:
    """A future raiden-friendly copy descriptor.

    The H2H payload is still compact bytes. The strides only describe how
    D2H packs source HBM and how H2D scatters into destination HBM.
    """

    src_offset_bytes: int
    dst_offset_bytes: int
    segment_bytes: int
    src_stride_bytes: int
    dst_stride_bytes: int
    num_segments: int
    layer_name: str = ""
    layer_type: LayerType = LayerType.FULL_ATTN
    source_base_addr: int = 0
    destination_base_addr: int = 0
    global_head: int | None = None
    segment_name: str | None = None

    def __post_init__(self) -> None:
        check_non_negative("src_offset_bytes", self.src_offset_bytes)
        check_non_negative("dst_offset_bytes", self.dst_offset_bytes)
        check_positive("segment_bytes", self.segment_bytes)
        check_positive("src_stride_bytes", self.src_stride_bytes)
        check_positive("dst_stride_bytes", self.dst_stride_bytes)
        check_positive("num_segments", self.num_segments)
        check_non_negative("source_base_addr", self.source_base_addr)
        check_non_negative("destination_base_addr", self.destination_base_addr)
        if self.global_head is not None:
            check_non_negative("global_head", self.global_head)

    @property
    def total_bytes(self) -> int:
        return self.segment_bytes * self.num_segments

    @property
    def src_addr(self) -> int:
        return self.source_base_addr + self.src_offset_bytes

    @property
    def dst_addr(self) -> int:
        return self.destination_base_addr + self.dst_offset_bytes


@dataclass(frozen=True)
class RankTransferPlan:
    """Lowered transfer ops from one producer rank."""

    p_rank: int
    ops: tuple[StridedSegmentOp, ...]

    @property
    def total_bytes(self) -> int:
        return sum(op.total_bytes for op in self.ops)
