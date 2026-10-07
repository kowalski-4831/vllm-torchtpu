# SPDX-License-Identifier: Apache-2.0
"""Unit tests for src/vllm_torchtpu/omni/qwen3_omni_moe_thinker_patch.py."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.omni.qwen3_omni_moe_thinker_patch import (
    _select_bucket,
    apply_omni_ar_patches,
)

pytestmark = pytest.mark.cpu_test


def test_apply_omni_ar_patches_audio_attention_and_padded_encoder(monkeypatch):
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

    seen_layer_seq_lens = []
    seen_conv_batches = []

    class DummyLayer(nn.Module):
        def forward(self, hidden_states, cu_seqlens, max_seqlen=None):
            seen_layer_seq_lens.append(int(hidden_states.shape[0]))
            return hidden_states

    class DummyAudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_window = 50
            self.n_window_infer = 400
            self.conv_chunksize = 500
            self.conv2d1 = nn.Conv2d(1, 4, 3, 2, padding=1)
            self.conv2d2 = nn.Conv2d(4, 4, 3, 2, padding=1)
            self.conv2d3 = nn.Conv2d(4, 4, 3, 2, padding=1)
            self.conv_out = lambda x: (nn.functional.linear(x, torch.ones(8, x.shape[-1])), None)
            self.positional_embedding = SimpleNamespace(
                positional_embedding=torch.zeros(100, 8)
            )
            self.layers = nn.ModuleList([DummyLayer()])
            self.ln_post = nn.LayerNorm(8)
            self.proj1 = lambda x: (x, None)
            self.act = nn.GELU()
            self.proj2 = lambda x: (x, None)

        def compute_attn_mask_seqlen(self, cu_seqlens):
            return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        def forward(self, input_features, feature_lens, aftercnn_lens):
            raise AssertionError("orig_forward should not be called directly")

    class DummyMixin:
        def _process_audio_input(self, audio_input):
            return (audio_input["input_features"], audio_input["audio_feature_lengths"])

    _current_tp_size = 8

    qwen3_pkg = types.ModuleType("vllm_omni.model_executor.models.qwen3_omni")
    thinker_mod = types.ModuleType(
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker"
    )
    thinker_mod.Qwen3OmniMoeAudioAttention = DummyAudioAttention
    thinker_mod.Qwen3OmniMoeAudioEncoder = DummyAudioEncoder
    thinker_mod.Qwen3OmniMoeConditionalGenerationMixin = DummyMixin
    qwen3_pkg.qwen3_omni_moe_thinker = thinker_mod

    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
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
    assert isinstance(attn_tp8.attn, DummyMMEncoderAttention)
    assert attn_tp8.attn.num_heads == 20
    assert attn_tp8.attn.head_size == 4

    # Case 2: tp_size=4 (20 % 4 == 0) -> disable_tp=False, num_local_heads=5
    _current_tp_size = 4
    attn_tp4 = DummyAudioAttention(cfg)
    assert attn_tp4.num_local_heads == 5
    assert attn_tp4.qkv.disable_tp is False
    assert attn_tp4.out_proj.disable_tp is False

    # Case 3: Padded encoder wrapper buckets all <= 800-frame clips to 8 chunks
    # (104 CNN tokens -> 128-aligned sequence length) and slices back to aftercnn_lens.
    enc = DummyAudioEncoder()
    for flen, alen in ((220, 29), (310, 41), (670, 87)):
        out = enc(
            torch.zeros(16, flen),
            torch.tensor([flen], dtype=torch.long),
            torch.tensor([alen], dtype=torch.long),
        )
        assert out.shape == (alen, 8)
    assert seen_layer_seq_lens == [128, 128, 128]
    assert _select_bucket(5, (8, 16), align=8) == 8
    assert _select_bucket(19, (8, 16), align=8) == 24

    class SchemaLikeAudioInput:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

    mixin = DummyMixin()
    feats, flens = mixin._process_audio_input(
        SchemaLikeAudioInput(
            {
                "input_features": torch.zeros(16, 220),
                "audio_feature_lengths": torch.tensor([220]),
            }
        )
    )
    assert feats.shape == (16, 220)
    assert flens.tolist() == [220]

    import vllm.model_executor.models.utils as vllm_utils

    inputs_embeds = torch.zeros(16, 8)
    mm_emb = torch.ones(4, 8)
    is_mm = torch.zeros(16, dtype=torch.bool)
    is_mm[:4] = True
    merged = vllm_utils._merge_multimodal_embeddings(
        inputs_embeds=inputs_embeds.clone(),
        multimodal_embeddings=[mm_emb],
        is_multimodal=is_mm,
    )
    assert torch.allclose(merged[:4], torch.ones(4, 8))
    assert torch.allclose(merged[4:], torch.zeros(12, 8))

