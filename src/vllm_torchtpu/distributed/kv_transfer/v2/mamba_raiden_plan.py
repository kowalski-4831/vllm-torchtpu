# SPDX-License-Identifier: Apache-2.0
"""GDN (mamba) layer resharding push plans for Raiden's registered-plan
transport.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from .common import TAG_GDN_CONV, TAG_GDN_SSM, head_range


class PlanError(ValueError):
    """The two shard geometries cannot be resharded into each other."""


@dataclasses.dataclass(frozen=True)
class GdnShardGeometry:
    """Per-rank GDN state geometry for one side of the transfer.

    All ranks of one side share this geometry. Byte layout per conv block:
    ``taps`` rows of ``[q_heads | k_heads | v_heads]``; per ssm block:
    ``local_value_heads`` dense states of ``key_head_dim × value_head_dim``.
    """

    tp_size: int
    taps: int
    total_key_heads: int
    total_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_itemsize: int
    ssm_itemsize: int

    def __post_init__(self) -> None:
        for name in ("tp_size", "taps", "total_key_heads", "total_value_heads",
                     "key_head_dim", "value_head_dim", "conv_itemsize",
                     "ssm_itemsize"):
            value = getattr(self, name)
            if int(value) <= 0:
                raise PlanError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, int(value))
        # Validate divisibility once up front.
        for name, total in (("total_key_heads", self.total_key_heads),
                            ("total_value_heads", self.total_value_heads)):
            if total >= self.tp_size:
                if total % self.tp_size != 0:
                    raise PlanError(f"{name}={total} is not divisible by "
                                    f"tp_size={self.tp_size}")
            elif self.tp_size % total != 0:
                raise PlanError(f"tp_size={self.tp_size} is not divisible "
                                f"by {name}={total}")

    @property
    def local_key_heads(self) -> int:
        return len(head_range(self.total_key_heads, self.tp_size, 0))

    @property
    def local_value_heads(self) -> int:
        return len(head_range(self.total_value_heads, self.tp_size, 0))

    @property
    def key_head_bytes(self) -> int:
        return self.key_head_dim * self.conv_itemsize

    @property
    def value_head_bytes(self) -> int:
        return self.value_head_dim * self.conv_itemsize

    @property
    def conv_row_bytes(self) -> int:
        """One conv tap row: ``[q | k | v]`` in local heads."""
        return (2 * self.local_key_heads * self.key_head_bytes +
                self.local_value_heads * self.value_head_bytes)

    @property
    def conv_q_base(self) -> int:
        return 0

    @property
    def conv_k_base(self) -> int:
        return self.local_key_heads * self.key_head_bytes

    @property
    def conv_v_base(self) -> int:
        return 2 * self.local_key_heads * self.key_head_bytes

    @property
    def conv_live_bytes(self) -> int:
        return self.taps * self.conv_row_bytes

    @property
    def ssm_head_bytes(self) -> int:
        return self.key_head_dim * self.value_head_dim * self.ssm_itemsize

    @property
    def ssm_live_bytes(self) -> int:
        return self.local_value_heads * self.ssm_head_bytes

    def key_range(self, rank: int) -> range:
        return head_range(self.total_key_heads, self.tp_size, rank)

    def value_range(self, rank: int) -> range:
        return head_range(self.total_value_heads, self.tp_size, rank)


@dataclasses.dataclass(frozen=True)
class GdnPushEntry:
    """One strided copy between a producer rank and a decode rank.

    Offsets are pool-block-relative on both sides; ``count`` segments of
    ``size_bytes`` at the respective strides (conv taps), matching raiden's
    ``ShardPushEntryProto``.
    """

    tag: str
    src_rank: int
    dst_rank: int
    src_offset_bytes: int
    dst_offset_bytes: int
    size_bytes: int
    src_stride_bytes: int
    dst_stride_bytes: int
    count: int
    segment: str  # diagnostic: "conv_q" | "conv_k" | "conv_v" | "ssm"

    @property
    def total_bytes(self) -> int:
        return self.size_bytes * self.count


def _canonical_source(global_head: int, total_heads: int,
                      src_tp_size: int) -> tuple[int, int]:
    """Returns (src_rank, src_local_head) for one global head.

    When heads are replicated on the source (``total < tp``), the first
    replica rank is the canonical sender so each byte has exactly one origin.
    """
    if total_heads >= src_tp_size:
        per_rank = total_heads // src_tp_size
        return global_head // per_rank, global_head % per_rank
    ranks_per_head = src_tp_size // total_heads
    return global_head * ranks_per_head, 0


def _head_runs(
    total_heads: int,
    src: GdnShardGeometry,
    dst: GdnShardGeometry,
    dst_rank: int,
    head_range_of: str,
) -> list[tuple[int, int, int, int]]:
    """Contiguous (src_rank, src_local_start, dst_local_start, num_heads)
    runs covering the destination rank's global head range."""
    dst_heads = (dst.key_range(dst_rank)
                 if head_range_of == "key" else dst.value_range(dst_rank))
    runs: list[tuple[int, int, int, int]] = []
    run: list[int] | None = None  # [src_rank, src_local, dst_local, count]
    for global_head in dst_heads:
        src_rank, src_local = _canonical_source(global_head, total_heads,
                                                src.tp_size)
        dst_local = global_head - dst_heads.start
        if (run is not None and run[0] == src_rank
                and src_local == run[1] + run[3]
                and dst_local == run[2] + run[3]):
            run[3] += 1
            continue
        if run is not None:
            runs.append(tuple(run))
        run = [src_rank, src_local, dst_local, 1]
    if run is not None:
        runs.append(tuple(run))
    return runs


def _check_compatible(src: GdnShardGeometry, dst: GdnShardGeometry) -> None:
    for name in ("taps", "total_key_heads", "total_value_heads",
                 "key_head_dim", "value_head_dim", "conv_itemsize",
                 "ssm_itemsize"):
        src_value = getattr(src, name)
        dst_value = getattr(dst, name)
        if src_value != dst_value:
            raise PlanError(f"source and destination disagree on {name}: "
                            f"{src_value} vs {dst_value}")


def build_gdn_reshard_entries(
    src: GdnShardGeometry,
    dst: GdnShardGeometry,
) -> tuple[GdnPushEntry, ...]:
    """All (src_rank → dst_rank) push entries for one block transfer.

    Block-agnostic: block ids are attached when lowering to a proto request.
    """
    _check_compatible(src, dst)
    entries: list[GdnPushEntry] = []

    conv_segments = (
        ("conv_q", "key", src.conv_q_base, dst.conv_q_base,
         src.key_head_bytes),
        ("conv_k", "key", src.conv_k_base, dst.conv_k_base,
         src.key_head_bytes),
        ("conv_v", "value", src.conv_v_base, dst.conv_v_base,
         src.value_head_bytes),
    )
    for dst_rank in range(dst.tp_size):
        for segment, head_type, src_base, dst_base, head_bytes in (
                conv_segments):
            total = (src.total_key_heads
                     if head_type == "key" else src.total_value_heads)
            for src_rank, src_local, dst_local, num_heads in _head_runs(
                    total, src, dst, dst_rank, head_type):
                entries.append(
                    GdnPushEntry(
                        tag=TAG_GDN_CONV,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        src_offset_bytes=src_base + src_local * head_bytes,
                        dst_offset_bytes=dst_base + dst_local * head_bytes,
                        size_bytes=num_heads * head_bytes,
                        src_stride_bytes=src.conv_row_bytes,
                        dst_stride_bytes=dst.conv_row_bytes,
                        count=src.taps,
                        segment=segment,
                    ))
        for src_rank, src_local, dst_local, num_heads in _head_runs(
                src.total_value_heads, src, dst, dst_rank, "value"):
            entries.append(
                GdnPushEntry(
                    tag=TAG_GDN_SSM,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    src_offset_bytes=src_local * src.ssm_head_bytes,
                    dst_offset_bytes=dst_local * dst.ssm_head_bytes,
                    size_bytes=num_heads * src.ssm_head_bytes,
                    src_stride_bytes=0,
                    dst_stride_bytes=0,
                    count=1,
                    segment="ssm",
                ))
    return tuple(entries)


def validate_entries_against_geometry(
    entries: Sequence[GdnPushEntry],
    src: GdnShardGeometry,
    dst: GdnShardGeometry,
) -> None:
    """Checks that, per destination rank and tag, the entries write every live
  byte exactly once and read only within the source's live bytes."""
    for tag, src_live, dst_live in (
        (TAG_GDN_CONV, src.conv_live_bytes, dst.conv_live_bytes),
        (TAG_GDN_SSM, src.ssm_live_bytes, dst.ssm_live_bytes),
    ):
        for dst_rank in range(dst.tp_size):
            spans: list[tuple[int, int]] = []
            for entry in entries:
                if entry.tag != tag or entry.dst_rank != dst_rank:
                    continue
                if entry.src_rank < 0 or entry.src_rank >= src.tp_size:
                    raise PlanError(f"{tag} entry has out-of-range src_rank "
                                    f"{entry.src_rank}")
                for segment_index in range(entry.count):
                    src_start = (entry.src_offset_bytes +
                                 segment_index * entry.src_stride_bytes)
                    if src_start + entry.size_bytes > src_live:
                        raise PlanError(
                            f"{tag} entry reads beyond source live bytes: "
                            f"{src_start + entry.size_bytes} > {src_live}")
                    dst_start = (entry.dst_offset_bytes +
                                 segment_index * entry.dst_stride_bytes)
                    spans.append((dst_start, dst_start + entry.size_bytes))
            spans.sort()
            cursor = 0
            for start, end in spans:
                if start != cursor:
                    verb = "overlap" if start < cursor else "gap"
                    raise PlanError(
                        f"{tag} destination bytes have a {verb} at "
                        f"{min(start, cursor)} on dst_rank {dst_rank}")
                cursor = end
            if cursor != dst_live:
                raise PlanError(
                    f"{tag} entries cover {cursor} bytes for dst_rank "
                    f"{dst_rank}, expected {dst_live}")


def validate_geometry_against_manifest(
    geometry: GdnShardGeometry,
    geometry_by_tag: Mapping[str, Mapping[str, int]],
) -> None:
    """Cross-checks a shard geometry against a PoolManifest's
    ``geometry_by_tag()`` for the same side."""
    for tag, live in ((TAG_GDN_CONV, geometry.conv_live_bytes),
                      (TAG_GDN_SSM, geometry.ssm_live_bytes)):
        pool_geometry = geometry_by_tag.get(tag)
        if pool_geometry is None:
            raise PlanError(f"manifest has no {tag} pools")
        manifest_live = int(pool_geometry["live_bytes_per_block"])
        if manifest_live != live:
            raise PlanError(f"{tag} live bytes disagree: geometry={live} "
                            f"manifest={manifest_live}")
        stride = int(pool_geometry["block_stride_bytes"])
        if stride < live:
            raise PlanError(f"{tag} block stride {stride} is smaller than "
                            f"live bytes {live}")


__all__ = [
    "GdnPushEntry",
    "GdnShardGeometry",
    "PlanError",
    "build_gdn_reshard_entries",
    "validate_entries_against_geometry",
    "validate_geometry_against_manifest",
]
