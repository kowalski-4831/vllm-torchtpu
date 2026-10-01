# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Qwen3-Omni-MoE Thinker TPU patches and Stage 1/2 stubs."""

from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.qwen3_omni_moe_thinker_patch import (
    _is_qwen3_omni_thinker_model,
    _patch_qwen3_omni_audio_attention,
    _patch_qwen3_omni_audio_encoder,
    _patch_qwen3_omni_mrope,
)
from vllm_torchtpu.omni.patches import (
    patch_qwen3_code_predictor,
    patch_stage_input_processors,
)
from vllm_torchtpu.omni.runner import VocoderStep

pytestmark = pytest.mark.cpu_test


def test_is_qwen3_omni_thinker_model():
    assert not _is_qwen3_omni_thinker_model(None)
    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_omni_moe", architectures=[]),
        model="Qwen/Qwen3-Omni-30B-A3B-Thinking",
    )
    assert _is_qwen3_omni_thinker_model(cfg)


def test_audio_attention_and_encoder(monkeypatch):
    class DummyAudioAttn(nn.Module):
        pass

    class DummyEncoder:
        def forward(self, input_features, feature_lens, aftercnn_lens):
            return input_features.shape[-1]

    class FakeLinear(nn.Module):
        def __init__(self, in_dim=80, out_dim=240, **kwargs):
            super().__init__()
            self.disable_tp = kwargs.get("disable_tp", False)
            self.proj = nn.Linear(
                kwargs.get("hidden_size", kwargs.get("input_size", in_dim)),
                kwargs.get("output_size", out_dim),
                bias=False,
            )

        def forward(self, x):
            return self.proj(x), None

    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr("vllm.model_executor.layers.linear.QKVParallelLinear", FakeLinear)
    monkeypatch.setattr("vllm.model_executor.layers.linear.RowParallelLinear", FakeLinear)

    dummy_mod = SimpleNamespace(
        Qwen3OmniMoeAudioAttention=DummyAudioAttn,
        Qwen3OmniMoeAudioEncoder=DummyEncoder,
    )
    _patch_qwen3_omni_audio_attention(dummy_mod)
    _patch_qwen3_omni_audio_encoder(dummy_mod)

    attn = DummyAudioAttn(SimpleNamespace(d_model=80, encoder_attention_heads=20))
    assert attn.qkv.disable_tp is True
    out = attn(torch.randn(10, 80), torch.tensor([0, 4, 10], dtype=torch.int32))
    assert out.shape == (10, 80)

    enc = DummyEncoder()
    assert enc.forward(torch.zeros(128, 95), torch.tensor([50, 50]), None) == 100


def test_mrope_and_stage_stubs():
    class DummyThinker:
        def _get_mrope_input_positions(self, input_tokens, image_grid_thw=None):
            return image_grid_thw + 1, torch.tensor(0)

    dummy_mod = SimpleNamespace(Qwen3OmniMoeThinkerForConditionalGeneration=DummyThinker)
    _patch_qwen3_omni_mrope(dummy_mod)
    pos, _ = DummyThinker()._get_mrope_input_positions(
        [1, 2], image_grid_thw=torch.tensor([[1, 2, 2]])
    )
    assert torch.equal(pos, torch.tensor([[2, 3, 3]]))

    with pytest.raises(NotImplementedError):
        VocoderStep().warmup()
    with pytest.raises(NotImplementedError):
        patch_qwen3_code_predictor()
    with pytest.raises(NotImplementedError):
        patch_stage_input_processors()
