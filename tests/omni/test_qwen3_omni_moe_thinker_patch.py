# SPDX-License-Identifier: Apache-2.0
"""Unit tests for src/vllm_torchtpu/omni/qwen3_omni_moe_thinker_patch.py."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm_torchtpu.omni.qwen3_omni_moe_thinker_patch import (
    apply_omni_ar_patches,
)

pytestmark = pytest.mark.cpu_test


def test_apply_omni_ar_patches_audio_attention(monkeypatch):
    class DummyQKVParallelLinear(nn.Module):
        def __init__(
            self,
            input_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            bias=True,
            quant_config=None,
            prefix="",
            disable_tp=False,
        ):
            super().__init__()
            self.prefix = prefix
            self.disable_tp = disable_tp

    class DummyRowParallelLinear(nn.Module):
        def __init__(
            self,
            input_size,
            output_size,
            bias=True,
            quant_config=None,
            prefix="",
            disable_tp=False,
        ):
            super().__init__()
            self.prefix = prefix
            self.disable_tp = disable_tp

    class DummyMMEncoderAttention(nn.Module):
        def __init__(self, num_heads, head_size, scale=None, prefix=""):
            super().__init__()
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = scale
            self.prefix = prefix

    class DummyAudioAttention(nn.Module):
        pass

    _current_tp_size = 8

    qwen3_pkg = types.ModuleType("vllm_omni.model_executor.models.qwen3_omni")
    thinker_mod = types.ModuleType(
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker"
    )
    thinker_mod.Qwen3OmniMoeAudioAttention = DummyAudioAttention
    qwen3_pkg.qwen3_omni_moe_thinker = thinker_mod

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni",
        types.ModuleType("vllm_omni"),
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor",
        types.ModuleType("vllm_omni.model_executor"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models",
        types.ModuleType("vllm_omni.model_executor.models"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.qwen3_omni",
        qwen3_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker",
        thinker_mod,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size",
        lambda: _current_tp_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mm_encoder_attention.MMEncoderAttention",
        DummyMMEncoderAttention,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.QKVParallelLinear",
        DummyQKVParallelLinear,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.RowParallelLinear",
        DummyRowParallelLinear,
    )

    apply_omni_ar_patches.cache_clear()
    apply_omni_ar_patches()

    cfg = SimpleNamespace(d_model=80, encoder_attention_heads=20)

    # Case 1: tp_size=8 (20 % 8 != 0) -> disable_tp=True, num_local_heads=20
    _current_tp_size = 8
    attn_tp8 = DummyAudioAttention(cfg, prefix="audio.layers.0.self_attn")
    assert attn_tp8.num_local_heads == 20
    assert attn_tp8.qkv.disable_tp is True
    assert attn_tp8.out_proj.disable_tp is True
    assert attn_tp8.qkv.prefix == "audio.layers.0.self_attn.qkv"
    assert attn_tp8.out_proj.prefix == "audio.layers.0.self_attn.out_proj"
    assert isinstance(attn_tp8.attn, DummyMMEncoderAttention)
    assert attn_tp8.attn.num_heads == 20
    assert attn_tp8.attn.head_size == 4
    assert attn_tp8.attn.scale == 4**-0.5
    assert attn_tp8.attn.prefix == "audio.layers.0.self_attn.attn"

    # Case 2: tp_size=4 (20 % 4 == 0) -> disable_tp=False, num_local_heads=5
    _current_tp_size = 4
    attn_tp4 = DummyAudioAttention(cfg)
    assert attn_tp4.num_local_heads == 5
    assert attn_tp4.qkv.disable_tp is False
    assert attn_tp4.out_proj.disable_tp is False
    assert isinstance(attn_tp4.attn, DummyMMEncoderAttention)
    assert attn_tp4.attn.num_heads == 5
    assert attn_tp4.attn.head_size == 4
