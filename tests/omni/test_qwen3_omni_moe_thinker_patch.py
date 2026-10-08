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
    class DummyLinear(nn.Module):
        def __init__(self, *args, prefix="", disable_tp=False, **kwargs):
            super().__init__()
            self.prefix, self.disable_tp = prefix, disable_tp

    class DummyMMEncoderAttention(nn.Module):
        def __init__(self, num_heads, head_size, scale=None, prefix=""):
            super().__init__()
            self.num_heads, self.head_size, self.scale, self.prefix = (
                num_heads,
                head_size,
                scale,
                prefix,
            )

    class DummyAudioAttention(nn.Module):
        pass

    seen_layer_seq_lens = []

    class DummyLayer(nn.Module):
        def forward(self, hidden_states, cu_seqlens, max_seqlen=None):
            seen_layer_seq_lens.append(int(hidden_states.shape[0]))
            return hidden_states

    class DummyAudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_window, self.n_window_infer, self.conv_chunksize = 50, 400, 500
            self.conv2d1 = nn.Conv2d(1, 4, 3, 2, padding=1)
            self.conv2d2 = nn.Conv2d(4, 4, 3, 2, padding=1)
            self.conv2d3 = nn.Conv2d(4, 4, 3, 2, padding=1)
            self.conv_out = lambda x: (nn.functional.linear(x, torch.ones(8, x.shape[-1])), None)
            self.positional_embedding = SimpleNamespace(positional_embedding=torch.zeros(100, 8))
            self.layers = nn.ModuleList([DummyLayer()])
            self.ln_post, self.act = nn.LayerNorm(8), nn.GELU()
            self.proj1 = self.proj2 = lambda x: (x, None)

    class DummyViTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_shapes = []

        def forward(self, hs, **kwargs):
            self.seen_shapes.append(int(hs.shape[0]))
            return hs

    class DummyViT(nn.Module):
        dtype = torch.float32
        apply_vit_abs_pos_embed = True
        deepstack_visual_indexes = [0]
        spatial_merge_unit = 4

        def __init__(self):
            super().__init__()
            self.pos_embed = nn.Embedding(16, 8)
            self.rotary_pos_emb = nn.Embedding(16, 4)
            self.patch_embed = nn.Identity()
            self.patch_embed.proj = SimpleNamespace(weight=torch.ones(8, 8))
            self.blk = DummyViTBlock()
            self.blocks = nn.ModuleList([self.blk])
            self.merger = lambda hs: hs.squeeze(1)[::4]
            self.merger_list = [lambda d: d.squeeze(1)[::4]]

        def fast_pos_embed_interpolate(self, grid_thw):
            return torch.zeros(sum(int(t) * int(h) * int(w) for t, h, w in grid_thw), 8)

        def rot_pos_emb(self, grid_thw):
            n = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
            return torch.ones(n, 4), torch.zeros(n, 4)

    class DummyMixin:
        def _process_audio_input(self, ai):
            return ai["input_features"], ai["audio_feature_lengths"]

    _current_tp_size = 8
    qwen3_pkg = types.ModuleType("vllm_omni.model_executor.models.qwen3_omni")
    thinker_mod = types.ModuleType(
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker"
    )
    thinker_mod.Qwen3OmniMoeAudioAttention = DummyAudioAttention
    thinker_mod.Qwen3OmniMoeAudioEncoder = DummyAudioEncoder
    thinker_mod.Qwen3Omni_VisionTransformer = DummyViT
    thinker_mod.Qwen3OmniMoeConditionalGenerationMixin = DummyMixin
    qwen3_pkg.qwen3_omni_moe_thinker = thinker_mod

    for mod_name, mod_obj in (
        ("vllm_omni", types.ModuleType("vllm_omni")),
        ("vllm_omni.model_executor", types.ModuleType("vllm_omni.model_executor")),
        ("vllm_omni.model_executor.models", types.ModuleType("vllm_omni.model_executor.models")),
        ("vllm_omni.model_executor.models.qwen3_omni", qwen3_pkg),
        ("vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker", thinker_mod),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: _current_tp_size
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mm_encoder_attention.MMEncoderAttention",
        DummyMMEncoderAttention,
    )
    monkeypatch.setattr("vllm.model_executor.layers.linear.QKVParallelLinear", DummyLinear)
    monkeypatch.setattr("vllm.model_executor.layers.linear.RowParallelLinear", DummyLinear)

    apply_omni_ar_patches.cache_clear()
    apply_omni_ar_patches()

    cfg = SimpleNamespace(d_model=80, encoder_attention_heads=20)
    attn_tp8 = DummyAudioAttention(cfg, prefix="audio.layers.0.self_attn")
    assert attn_tp8.num_local_heads == 20 and attn_tp8.qkv.disable_tp is True

    _current_tp_size = 4
    attn_tp4 = DummyAudioAttention(cfg)
    assert attn_tp4.num_local_heads == 5 and attn_tp4.qkv.disable_tp is False

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

    vit = DummyViT()
    out_vit = vit(torch.ones(20000, 8), torch.tensor([[5, 40, 100]], dtype=torch.int32))
    assert out_vit.shape == (5000, 16) and vit.blk.seen_shapes == [16384, 4096]

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    vid_emb, aud_emb = torch.ones(6, 32), torch.ones(4, 8)
    vid_emb.modality, aud_emb.modality = "video", "audio"
    captured_mm = []

    class DummyRunner:
        supports_mm_inputs = True
        _omni_mm_embeds_list = [vid_emb, aud_emb]
        model = SimpleNamespace(
            embed_input_ids=lambda ids, multimodal_embeddings=None, is_multimodal=None: (
                captured_mm.append(multimodal_embeddings) or torch.zeros(ids.shape[0], 8)
            )
        )

    _, embeds_out = TPUModelRunner._get_model_inputs(
        DummyRunner(),
        torch.zeros(5, dtype=torch.long),
        ([torch.arange(4, 9, dtype=torch.int64)], torch.ones(5, dtype=torch.bool)),
    )
    assert embeds_out.shape == (5, 8) and len(captured_mm[0]) == 2
    assert captured_mm[0][0].shape == (2, 32) and captured_mm[0][0].modality == "video"
    assert captured_mm[0][1].shape == (3, 8) and captured_mm[0][1].modality == "audio"
