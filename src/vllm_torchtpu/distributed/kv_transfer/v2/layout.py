from __future__ import annotations

from dataclasses import dataclass

from .common import LayerType, TensorLayout, check_non_negative, check_positive


@dataclass(frozen=True)
class HeadSegment:
    """Explicit local payload segment for non-uniform state layouts."""

    name: str
    global_heads: tuple[int, ...]
    local_head_start: int
    local_head_count: int
    head_bytes: int
    base_offset_bytes: int = 0
    stride_bytes: int | None = None
    num_segments: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_heads", tuple(self.global_heads))
        if not self.global_heads:
            raise ValueError("global_heads must be non-empty")
        check_non_negative("local_head_start", self.local_head_start)
        check_positive("local_head_count", self.local_head_count)
        check_positive("head_bytes", self.head_bytes)
        check_non_negative("base_offset_bytes", self.base_offset_bytes)
        if self.stride_bytes is not None:
            check_positive("stride_bytes", self.stride_bytes)
        check_positive("num_segments", self.num_segments)
        if len(self.global_heads) != self.local_head_count:
            raise ValueError(
                "global_heads count must match local_head_count, got "
                f"{len(self.global_heads)} and {self.local_head_count}")


SUPPORTED_TOKEN_FIRST_LAYOUT_IDS = frozenset({
    "pallas_batched_rpa_token_first_v1",
})


@dataclass(frozen=True)
class TokenFirstLayoutSpec:
    """Strict token-first cache layout contract for pointer lowering."""

    block_size: int
    block_bytes: int
    block_stride_bytes: int
    token_stride_bytes: int
    head_stride_bytes: int
    live_head_bytes: int
    num_heads: int
    layout_id: str = "pallas_batched_rpa_token_first_v1"

    def __post_init__(self) -> None:
        check_positive("block_size", self.block_size)
        check_positive("block_bytes", self.block_bytes)
        check_positive("block_stride_bytes", self.block_stride_bytes)
        check_positive("token_stride_bytes", self.token_stride_bytes)
        check_positive("head_stride_bytes", self.head_stride_bytes)
        check_positive("live_head_bytes", self.live_head_bytes)
        check_positive("num_heads", self.num_heads)
        if self.layout_id not in SUPPORTED_TOKEN_FIRST_LAYOUT_IDS:
            raise ValueError(
                f"unsupported TOKEN_FIRST layout_id: {self.layout_id}")
        if self.block_bytes != self.block_size * self.token_stride_bytes:
            raise ValueError(
                "block_bytes must equal block_size * token_stride_bytes "
                "for TOKEN_FIRST layout")
        if self.block_stride_bytes < self.block_bytes:
            raise ValueError(
                "block_stride_bytes must be >= block_bytes, got "
                f"{self.block_stride_bytes} and {self.block_bytes}")
        if self.head_stride_bytes < self.live_head_bytes:
            raise ValueError(
                "head_stride_bytes must be >= live_head_bytes, got "
                f"{self.head_stride_bytes} and {self.live_head_bytes}")
        token_extent = ((self.num_heads - 1) * self.head_stride_bytes +
                        self.live_head_bytes)
        if self.token_stride_bytes < token_extent:
            raise ValueError(
                "token_stride_bytes cannot cover num_heads/head_stride_bytes/"
                "live_head_bytes")


@dataclass(frozen=True)
class KVCacheRegion:
    """One physical HBM cache tensor exposed by a connector worker."""

    rank: int
    layer_name: str
    layer_type: LayerType
    base_addr: int
    block_size: int | None
    block_bytes: int
    layout: TensorLayout
    num_heads: int = 1
    head_bytes: int | None = None
    token_first_layout: TokenFirstLayoutSpec | None = None
    token_stride_bytes: int | None = None
    head_stride_bytes: int | None = None
    live_head_bytes: int | None = None
    block_stride_bytes: int | None = None
    block_id_index: int | None = None
    head_segments: tuple[HeadSegment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_segments", tuple(self.head_segments))
        check_non_negative("rank", self.rank)
        check_non_negative("base_addr", self.base_addr)
        check_positive("block_bytes", self.block_bytes)
        check_positive("num_heads", self.num_heads)
        if self.block_size is not None:
            check_positive("block_size", self.block_size)
            if self.block_bytes % self.block_size != 0:
                raise ValueError(
                    "block_bytes must be divisible by block_size, got "
                    f"{self.block_bytes} and {self.block_size}")
        if self.block_stride_bytes is None:
            object.__setattr__(self, "block_stride_bytes", self.block_bytes)
        else:
            check_positive("block_stride_bytes", self.block_stride_bytes)
            if self.block_stride_bytes < self.block_bytes:
                raise ValueError(
                    "block_stride_bytes must be >= block_bytes, got "
                    f"{self.block_stride_bytes} and {self.block_bytes}")
        if self.block_id_index is not None:
            check_non_negative("block_id_index", self.block_id_index)
        inferred_head_bytes = self._init_token_first_layout()
        if inferred_head_bytes is None:
            if self.block_bytes % self.num_heads != 0:
                raise ValueError(
                    "block_bytes must be divisible by num_heads, got "
                    f"{self.block_bytes} and {self.num_heads}")
            inferred_head_bytes = self.block_bytes // self.num_heads
        if self.head_bytes is None:
            object.__setattr__(self, "head_bytes", inferred_head_bytes)
        elif self.head_bytes != inferred_head_bytes:
            raise ValueError(
                f"head_bytes={self.head_bytes} does not match inferred "
                f"head_bytes={inferred_head_bytes}")

    def _init_token_first_layout(self) -> int | None:
        if self.layout != TensorLayout.TOKEN_FIRST:
            return None
        if self.block_size is None:
            raise ValueError("TOKEN_FIRST layout requires block_size")

        if self.token_first_layout is not None:
            spec = self.token_first_layout
            if (self.block_size != spec.block_size
                    or self.block_bytes != spec.block_bytes or
                    self.physical_block_stride_bytes != spec.block_stride_bytes
                    or self.num_heads != spec.num_heads):
                raise ValueError("KVCacheRegion TOKEN_FIRST fields must match "
                                 "token_first_layout")
            if (self.token_stride_bytes is not None
                    and self.token_stride_bytes != spec.token_stride_bytes):
                raise ValueError("KVCacheRegion TOKEN_FIRST fields must match "
                                 "token_first_layout")
            if (self.head_stride_bytes is not None
                    and self.head_stride_bytes != spec.head_stride_bytes):
                raise ValueError("KVCacheRegion TOKEN_FIRST fields must match "
                                 "token_first_layout")
            if (self.live_head_bytes is not None
                    and self.live_head_bytes != spec.live_head_bytes):
                raise ValueError("KVCacheRegion TOKEN_FIRST fields must match "
                                 "token_first_layout")
            object.__setattr__(self, "token_stride_bytes",
                               spec.token_stride_bytes)
            object.__setattr__(self, "head_stride_bytes",
                               spec.head_stride_bytes)
            object.__setattr__(self, "live_head_bytes", spec.live_head_bytes)
            return spec.live_head_bytes * spec.block_size

        inferred_token_stride = self.block_bytes // self.block_size
        token_stride = self.token_stride_bytes
        if token_stride is None:
            token_stride = inferred_token_stride

        live_head = self.live_head_bytes
        if live_head is None:
            if self.head_stride_bytes is not None:
                live_head = self.head_stride_bytes
            else:
                if token_stride % self.num_heads != 0:
                    raise ValueError(
                        "token_stride_bytes must be divisible by num_heads "
                        "when live_head_bytes is not set")
                live_head = token_stride // self.num_heads

        head_stride = self.head_stride_bytes
        if head_stride is None:
            head_stride = live_head
        spec = TokenFirstLayoutSpec(
            block_size=self.block_size,
            block_bytes=self.block_bytes,
            block_stride_bytes=self.physical_block_stride_bytes,
            token_stride_bytes=token_stride,
            head_stride_bytes=head_stride,
            live_head_bytes=live_head,
            num_heads=self.num_heads,
        )
        object.__setattr__(self, "token_first_layout", spec)
        object.__setattr__(self, "token_stride_bytes", spec.token_stride_bytes)
        object.__setattr__(self, "head_stride_bytes", spec.head_stride_bytes)
        object.__setattr__(self, "live_head_bytes", spec.live_head_bytes)
        return spec.live_head_bytes * spec.block_size

    @property
    def token_bytes(self) -> int:
        if self.block_size is None:
            raise ValueError(
                f"{self.layer_name} has opaque state layout; token_bytes is "
                "not defined when block_size is None")
        if self.token_stride_bytes is not None:
            return self.token_stride_bytes
        return self.block_bytes // self.block_size

    @property
    def lowering_units_per_block(self) -> int:
        if self.block_size is None:
            return 1
        return self.block_size

    @property
    def physical_block_stride_bytes(self) -> int:
        assert self.block_stride_bytes is not None
        return self.block_stride_bytes

    @property
    def token_head_bytes(self) -> int:
        assert self.head_bytes is not None
        if self.block_size is None:
            raise ValueError(
                f"{self.layer_name} has opaque state layout; "
                "token_head_bytes is not defined when block_size is None")
        if self.live_head_bytes is not None:
            return self.live_head_bytes
        if self.head_bytes % self.block_size != 0:
            raise ValueError(
                "head_bytes must be divisible by block_size for token-level "
                f"lowering, got {self.head_bytes} and {self.block_size}")
        return self.head_bytes // self.block_size

    @property
    def token_head_stride_bytes(self) -> int:
        if self.block_size is None:
            raise ValueError(
                f"{self.layer_name} has opaque state layout; "
                "token_head_stride_bytes is not defined when block_size is None"
            )
        if self.head_stride_bytes is not None:
            return self.head_stride_bytes
        return self.token_head_bytes
