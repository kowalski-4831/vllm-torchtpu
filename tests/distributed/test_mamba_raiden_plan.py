# SPDX-License-Identifier: Apache-2.0
"""Deviceless tests for the GDN raiden resharding plan builder."""

import pytest

from vllm_torchtpu.distributed.kv_transfer.v2.common import (TAG_GDN_CONV,
                                                             TAG_GDN_SSM)
from vllm_torchtpu.distributed.kv_transfer.v2.mamba_raiden_plan import (
    GdnShardGeometry, PlanError, build_gdn_reshard_entries,
    validate_entries_against_geometry, validate_geometry_against_manifest)


def qwen35_geometry(tp_size: int) -> GdnShardGeometry:
    """Qwen3.5-397B-A17B GDN geometry (16 key heads, 64 value heads, d=128,
    conv kernel 4 -> 3 taps, conv bf16, ssm fp32)."""
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


class TestGeometry:

    def test_replicated_heads_local_count(self):
        geometry = GdnShardGeometry(tp_size=8,
                                    taps=3,
                                    total_key_heads=4,
                                    total_value_heads=8,
                                    key_head_dim=128,
                                    value_head_dim=128,
                                    conv_itemsize=2,
                                    ssm_itemsize=4)
        assert geometry.local_key_heads == 1
        assert geometry.local_value_heads == 1

    def test_indivisible_heads_rejected(self):
        with pytest.raises(PlanError, match="not divisible"):
            GdnShardGeometry(tp_size=8,
                             taps=3,
                             total_key_heads=12,
                             total_value_heads=32,
                             key_head_dim=128,
                             value_head_dim=128,
                             conv_itemsize=2,
                             ssm_itemsize=4)


class TestEntryBuilding:

    @pytest.mark.parametrize("src_tp,dst_tp", [(8, 8), (8, 2), (8, 1), (2, 8),
                                               (4, 8), (8, 4)])
    def test_coverage_is_exact(self, src_tp, dst_tp):
        src = qwen35_geometry(src_tp)
        dst = qwen35_geometry(dst_tp)
        entries = build_gdn_reshard_entries(src, dst)
        validate_entries_against_geometry(entries, src, dst)

    def test_tp8_to_tp1_gathers_all_ranks(self):
        """The eventual target decode shape (D: DP8, TP1): rank 0 fans in
        from every producer rank."""
        src = qwen35_geometry(8)
        dst = qwen35_geometry(1)
        entries = build_gdn_reshard_entries(src, dst)
        assert {entry.dst_rank for entry in entries} == {0}
        assert {entry.src_rank for entry in entries} == set(range(8))
        # Every producer contributes its full local state.
        for src_rank in range(8):
            conv_bytes = sum(
                entry.total_bytes for entry in entries
                if entry.tag == TAG_GDN_CONV and entry.src_rank == src_rank)
            assert conv_bytes == src.conv_live_bytes
            ssm_bytes = sum(
                entry.total_bytes for entry in entries
                if entry.tag == TAG_GDN_SSM and entry.src_rank == src_rank)
            assert ssm_bytes == src.ssm_live_bytes

    def test_replicated_source_heads_have_canonical_sender(self):
        """4 key heads on TP8: two ranks replicate each head; only the first
        replica (even rank) sends."""
        src = GdnShardGeometry(tp_size=8,
                               taps=3,
                               total_key_heads=4,
                               total_value_heads=8,
                               key_head_dim=128,
                               value_head_dim=128,
                               conv_itemsize=2,
                               ssm_itemsize=4)
        dst = GdnShardGeometry(tp_size=2,
                               taps=3,
                               total_key_heads=4,
                               total_value_heads=8,
                               key_head_dim=128,
                               value_head_dim=128,
                               conv_itemsize=2,
                               ssm_itemsize=4)
        entries = build_gdn_reshard_entries(src, dst)
        validate_entries_against_geometry(entries, src, dst)
        key_senders = {
            entry.src_rank
            for entry in entries if entry.segment in ("conv_q", "conv_k")
        }
        assert key_senders == {0, 2, 4, 6}
        for entry in entries:
            assert entry.src_offset_bytes >= 0

    def test_replicated_destination_heads_all_receive(self):
        """4 key heads onto TP8 decode: each decode rank still gets its
        (replicated) head."""
        src = GdnShardGeometry(tp_size=2,
                               taps=3,
                               total_key_heads=4,
                               total_value_heads=8,
                               key_head_dim=128,
                               value_head_dim=128,
                               conv_itemsize=2,
                               ssm_itemsize=4)
        dst = GdnShardGeometry(tp_size=8,
                               taps=3,
                               total_key_heads=4,
                               total_value_heads=8,
                               key_head_dim=128,
                               value_head_dim=128,
                               conv_itemsize=2,
                               ssm_itemsize=4)
        entries = build_gdn_reshard_entries(src, dst)
        validate_entries_against_geometry(entries, src, dst)
        assert {entry.dst_rank for entry in entries} == set(range(8))


class TestManifestCrossCheck:

    def test_manifest_mismatch_rejected(self):
        geometry = qwen35_geometry(8)
        wrong = {
            TAG_GDN_CONV: {
                "num_blocks": 16,
                "block_stride_bytes": geometry.conv_live_bytes,
                "live_bytes_per_block": geometry.conv_live_bytes + 8,
            },
            TAG_GDN_SSM: {
                "num_blocks": 16,
                "block_stride_bytes": geometry.ssm_live_bytes,
                "live_bytes_per_block": geometry.ssm_live_bytes,
            },
        }
        with pytest.raises(PlanError, match="live bytes disagree"):
            validate_geometry_against_manifest(geometry, wrong)
