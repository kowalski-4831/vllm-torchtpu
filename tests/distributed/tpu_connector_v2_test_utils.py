# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from typing import Any

from vllm_torchtpu.distributed.kv_transfer.v2.common import (LayerType,
                                                             TensorLayout)
from vllm_torchtpu.distributed.kv_transfer.v2.layout import (
    HeadSegment, KVCacheRegion, TokenFirstLayoutSpec)
from vllm_torchtpu.distributed.kv_transfer.v2.metadata import (
    ConnectorMetadataV2, HeadMapping, KVParallelLayout, LocalDecodeAllocation,
    PullMeta, RankTransferPlan, SourceBlockRef, StridedSegmentOp, TpKVTopology)
from vllm_torchtpu.distributed.kv_transfer.v2.pcp_policy import (
    PcpReshardingPolicy, PcpTokenTransfer)
from vllm_torchtpu.distributed.kv_transfer.v2.planner import (
    ContiguousHeadTPTransferPlanner, TPTransferPlanner)


def load_v2_module():
    return SimpleNamespace(
        ConnectorMetadataV2=ConnectorMetadataV2,
        ContiguousHeadTPTransferPlanner=ContiguousHeadTPTransferPlanner,
        HeadSegment=HeadSegment,
        HeadMapping=HeadMapping,
        KVCacheRegion=KVCacheRegion,
        KVParallelLayout=KVParallelLayout,
        LayerType=LayerType,
        LocalDecodeAllocation=LocalDecodeAllocation,
        PcpReshardingPolicy=PcpReshardingPolicy,
        PcpTokenTransfer=PcpTokenTransfer,
        PullMeta=PullMeta,
        RankTransferPlan=RankTransferPlan,
        SourceBlockRef=SourceBlockRef,
        StridedSegmentOp=StridedSegmentOp,
        TensorLayout=TensorLayout,
        TokenFirstLayoutSpec=TokenFirstLayoutSpec,
        TPTransferPlanner=TPTransferPlanner,
        TpKVTopology=TpKVTopology,
    )


def qwen35_4tp_layer_layout():
    raw_layout = []
    for raw_idx in range(10):
        base_layer = raw_idx * 4
        raw_layout.append({
            "raw_idx":
            raw_idx,
            "linear_layers":
            tuple(f"model.layers.{base_layer + offset}.linear_attn"
                  for offset in range(3)),
            "full_layer":
            f"model.layers.{base_layer + 3}.self_attn",
        })
    return tuple(raw_layout)


def layout(mod: Any, tp_size: int, pcp_size: int = 1):
    return mod.KVParallelLayout(
        full_attn_pcp_size=pcp_size,
        full_attn_tp_size=tp_size,
        linear_attn_pcp_size=pcp_size,
        linear_attn_tp_size=tp_size,
    )


def fa_pcp_source_layout(mod: Any, pcp_size: int):
    return mod.KVParallelLayout(
        full_attn_pcp_size=pcp_size,
        full_attn_tp_size=pcp_size,
        linear_attn_pcp_size=1,
        linear_attn_tp_size=1,
    )


def head_mapping_rows(mappings):
    return tuple((
        mapping.global_head,
        mapping.source_head,
        mapping.destination_head,
        mapping.segment_name,
    ) for mapping in mappings)


def block_ref_rows(block_refs):
    return tuple((block_ref.logical_block_index, block_ref.block_id)
                 for block_ref in block_refs)


def manual_pull_meta(
    mod: Any,
    *,
    req_id: int = 1,
    p_ranks: tuple[int, ...],
    fa_mapping_rows_by_rank=None,
    mamba_mapping_rows_by_rank=None,
    fa_source_block_ids: tuple[int, ...] = (),
    mamba_source_block_ids: tuple[int, ...] = (),
    fa_block_ref_rows_by_rank=None,
):
    fa_mapping_rows_by_rank = fa_mapping_rows_by_rank or {}
    mamba_mapping_rows_by_rank = mamba_mapping_rows_by_rank or {}
    fa_block_ref_rows_by_rank = fa_block_ref_rows_by_rank or {}

    fa_head_mappings_by_rank = build_head_mappings(mod,
                                                   fa_mapping_rows_by_rank)
    mamba_head_mappings_by_rank = build_head_mappings(
        mod, mamba_mapping_rows_by_rank)
    return mod.PullMeta(
        req_id=req_id,
        p_ranks=p_ranks,
        fa_heads_by_rank=heads_by_rank(fa_mapping_rows_by_rank),
        mamba_heads_by_rank=heads_by_rank(mamba_mapping_rows_by_rank),
        fa_head_mappings_by_rank=fa_head_mappings_by_rank,
        mamba_head_mappings_by_rank=mamba_head_mappings_by_rank,
        fa_source_block_ids=fa_source_block_ids,
        mamba_source_block_ids=mamba_source_block_ids,
        fa_block_refs_by_rank={
            rank:
            tuple(
                mod.SourceBlockRef(
                    logical_block_index=logical_block_index,
                    block_id=block_id,
                ) for logical_block_index, block_id in block_refs)
            for rank, block_refs in fa_block_ref_rows_by_rank.items()
        },
    )


def build_head_mappings(mod: Any, mapping_rows_by_rank):
    return {
        rank:
        tuple(
            mod.HeadMapping(
                global_head=global_head,
                source_head=source_head,
                destination_head=destination_head,
                segment_name=segment_name,
            ) for global_head, source_head, destination_head, segment_name in
            rows)
        for rank, rows in mapping_rows_by_rank.items()
    }


def heads_by_rank(mapping_rows_by_rank):
    return {
        rank: tuple(global_head for global_head, _, _, _ in rows)
        for rank, rows in mapping_rows_by_rank.items()
    }


def full_attention_region(
    mod: Any,
    *,
    rank: int,
    layer_name: str,
    base_addr: int,
    block_size: int,
    token_stride_bytes: int,
    head_stride_bytes: int,
    live_head_bytes: int,
    num_heads: int = 1,
    layout_id: str = "pallas_batched_rpa_token_first_v1",
):
    block_bytes = block_size * token_stride_bytes
    layout_spec = mod.TokenFirstLayoutSpec(
        block_size=block_size,
        block_bytes=block_bytes,
        block_stride_bytes=block_bytes,
        token_stride_bytes=token_stride_bytes,
        head_stride_bytes=head_stride_bytes,
        live_head_bytes=live_head_bytes,
        num_heads=num_heads,
        layout_id=layout_id,
    )
    return mod.KVCacheRegion(
        rank=rank,
        layer_name=layer_name,
        layer_type=mod.LayerType.FULL_ATTN,
        base_addr=base_addr,
        block_size=block_size,
        block_bytes=block_bytes,
        layout=mod.TensorLayout.TOKEN_FIRST,
        num_heads=num_heads,
        token_first_layout=layout_spec,
    )


def build_block_size_case(
    mod: Any,
    *,
    source_block_size: int,
    destination_block_size: int,
):
    layer_name = "model.layers.0.self_attn"
    live_head_bytes = 10
    source_region = full_attention_region(
        mod,
        rank=0,
        layer_name=layer_name,
        base_addr=1_000,
        block_size=source_block_size,
        token_stride_bytes=live_head_bytes,
        head_stride_bytes=live_head_bytes,
        live_head_bytes=live_head_bytes,
    )
    destination_region = full_attention_region(
        mod,
        rank=0,
        layer_name=layer_name,
        base_addr=2_000,
        block_size=destination_block_size,
        token_stride_bytes=live_head_bytes,
        head_stride_bytes=live_head_bytes,
        live_head_bytes=live_head_bytes,
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=1,
        block_size=source_block_size,
        kv_source_layout=layout(mod, tp_size=1),
        kv_caches={0: {
            layer_name: source_region
        }},
        fa_block_ids=(10, 11, 12),
        mamba_block_ids=(),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=1),
        block_size=destination_block_size,
        tp_rank=0,
        total_num_kv_heads=1,
        total_num_mamba_heads=0,
    )
    destination = mod.LocalDecodeAllocation(
        rank=0,
        block_size=destination_block_size,
        kv_caches={layer_name: destination_region},
        fa_block_ids=(20, 21),
        mamba_block_ids=(),
    )
    return metadata, topology, destination


def build_fa_pcp_case(
        mod: Any,
        *,
        destination_tp_size: int = 2,
        destination_tp_rank: int = 1,
        destination_block_size: int = 4,
        destination_block_ids: tuple[int, ...] = (200, 201, 202, 203),
):
    layer_name = "model.layers.0.self_attn"
    source_pcp_size = 4
    source_block_size = 2
    live_head_bytes = 10
    total_num_kv_heads = 2
    destination_num_heads = total_num_kv_heads // destination_tp_size
    token_stride_bytes = live_head_bytes * total_num_kv_heads
    source_regions_by_rank = {
        rank: {
            layer_name:
            full_attention_region(
                mod,
                rank=rank,
                layer_name=layer_name,
                base_addr=10_000 + rank * 1_000,
                block_size=source_block_size,
                token_stride_bytes=token_stride_bytes,
                head_stride_bytes=live_head_bytes,
                live_head_bytes=live_head_bytes,
                num_heads=total_num_kv_heads,
            )
        }
        for rank in range(source_pcp_size)
    }
    destination = mod.LocalDecodeAllocation(
        rank=destination_tp_rank,
        block_size=destination_block_size,
        kv_caches={
            layer_name:
            full_attention_region(
                mod,
                rank=destination_tp_rank,
                layer_name=layer_name,
                base_addr=100_000,
                block_size=destination_block_size,
                token_stride_bytes=token_stride_bytes,
                head_stride_bytes=live_head_bytes,
                live_head_bytes=live_head_bytes,
                num_heads=destination_num_heads,
            )
        },
        fa_block_ids=destination_block_ids,
        mamba_block_ids=(),
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=3,
        block_size=source_block_size,
        kv_source_layout=fa_pcp_source_layout(mod, pcp_size=source_pcp_size),
        kv_caches=source_regions_by_rank,
        fa_block_ids=(100, 101),
        mamba_block_ids=(),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=destination_tp_size),
        block_size=destination_block_size,
        tp_rank=destination_tp_rank,
        total_num_kv_heads=total_num_kv_heads,
        total_num_mamba_heads=0,
    )
    return metadata, topology, destination


def mamba_state_regions(
    mod: Any,
    *,
    rank: int,
    tp_rank: int,
    tp_size: int,
    base_addr: int,
    block_stride_bytes: int,
):
    layer_name = "model.layers.0.linear_attn"
    head_dim = 128
    bf16_bytes = 2
    fp32_bytes = 4
    key_heads = 16
    value_heads = 32
    local_key_heads = key_heads // tp_size
    local_value_heads = value_heads // tp_size
    local_conv_dim = (2 * key_heads * head_dim +
                      value_heads * head_dim) // tp_size
    conv_state_bytes = 3 * local_conv_dim * bf16_bytes
    recurrent_head_bytes = head_dim * head_dim * fp32_bytes
    recurrent_state_bytes = local_value_heads * recurrent_head_bytes
    conv_slot_stride_bytes = local_conv_dim * bf16_bytes
    q_heads = tuple(
        range(tp_rank * local_key_heads, (tp_rank + 1) * local_key_heads))
    v_heads = tuple(
        range(tp_rank * local_value_heads, (tp_rank + 1) * local_value_heads))
    q_offset = 0
    k_offset = local_key_heads * head_dim * bf16_bytes
    v_offset = k_offset * 2
    state0_name = f"{layer_name}.state0"
    state1_name = f"{layer_name}.state1"

    return {
        state0_name:
        mod.KVCacheRegion(
            rank=rank,
            layer_name=state0_name,
            layer_type=mod.LayerType.MAMBA_STATE,
            base_addr=base_addr,
            block_size=None,
            block_bytes=conv_state_bytes,
            block_stride_bytes=block_stride_bytes,
            layout=mod.TensorLayout.BLOCKS_FIRST,
            num_heads=local_key_heads * 2 + local_value_heads,
            head_segments=(
                mod.HeadSegment(
                    name="q",
                    global_heads=q_heads,
                    local_head_start=0,
                    local_head_count=local_key_heads,
                    head_bytes=head_dim * bf16_bytes,
                    base_offset_bytes=q_offset,
                    stride_bytes=conv_slot_stride_bytes,
                    num_segments=3,
                ),
                mod.HeadSegment(
                    name="k",
                    global_heads=q_heads,
                    local_head_start=local_key_heads,
                    local_head_count=local_key_heads,
                    head_bytes=head_dim * bf16_bytes,
                    base_offset_bytes=k_offset,
                    stride_bytes=conv_slot_stride_bytes,
                    num_segments=3,
                ),
                mod.HeadSegment(
                    name="v",
                    global_heads=v_heads,
                    local_head_start=local_key_heads * 2,
                    local_head_count=local_value_heads,
                    head_bytes=head_dim * bf16_bytes,
                    base_offset_bytes=v_offset,
                    stride_bytes=conv_slot_stride_bytes,
                    num_segments=3,
                ),
            ),
            block_id_index=0,
        ),
        state1_name:
        mod.KVCacheRegion(
            rank=rank,
            layer_name=state1_name,
            layer_type=mod.LayerType.MAMBA_STATE,
            base_addr=base_addr + conv_state_bytes,
            block_size=None,
            block_bytes=recurrent_state_bytes,
            block_stride_bytes=block_stride_bytes,
            layout=mod.TensorLayout.BLOCKS_FIRST,
            num_heads=local_value_heads,
            block_id_index=0,
        ),
    }
