# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Qwen3-Omni MoE Thinker TPU runtime patches."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.model_executor.models.utils as vllm_utils
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni_moe_thinker as thinker_mod,
)

from vllm_torchtpu.omni.qwen3_omni_moe_thinker_patch import (
    _select_bucket,
    apply_omni_ar_patches,
)

pytestmark = pytest.mark.omni


def test_apply_omni_ar_patches_padded_encoders_and_multimodal_slicing(monkeypatch):
    class DummyLinear(nn.Module):
        def forward(self, x):
            return x, None

    seen_layer_seq_lens: list[int] = []

    class DummyEncoderLayer(nn.Module):
        def forward(self, hidden_states, cu_seqlens, max_seqlen=None):
            seen_layer_seq_lens.append(int(hidden_states.shape[0]))
            return hidden_states

    class DummyAudioEncoder(nn.Module):
        n_window = 50
        n_window_infer = 400
        conv_chunksize = 500

        def __init__(self):
            super().__init__()
            self.conv2d1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
            self.conv2d2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
            self.conv2d3 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
            self.conv_out = DummyLinear()
            self.positional_embedding = SimpleNamespace(
                positional_embedding=torch.zeros(64, 8)
            )
            self.layers = nn.ModuleList([DummyEncoderLayer()])
            self.ln_post = nn.Identity()
            self.proj1 = DummyLinear()
            self.act = F.gelu
            self.proj2 = DummyLinear()

    class DummyViTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_shapes: list[int] = []

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
            self.merger = self._merge_patches
            self.merger_list = [self._merge_patches]

        @staticmethod
        def _merge_patches(hidden_states):
            return hidden_states.squeeze(1)[::4]

        def fast_pos_embed_interpolate(self, grid_thw):
            total = sum(int(t) * int(h) * int(w) for t, h, w in grid_thw)
            return torch.zeros(total, 8)

        def rot_pos_emb(self, grid_thw):
            n = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
            return torch.ones(n, 4), torch.zeros(n, 4)

    class DummyMixin:
        def embed_input_ids(
            self, input_ids, multimodal_embeddings=None, *, is_multimodal=None
        ):
            return multimodal_embeddings

    monkeypatch.setattr(thinker_mod, "Qwen3OmniMoeAudioEncoder", DummyAudioEncoder)
    monkeypatch.setattr(thinker_mod, "Qwen3Omni_VisionTransformer", DummyViT)
    monkeypatch.setattr(
        thinker_mod, "Qwen3OmniMoeThinkerForConditionalGeneration", DummyMixin
    )
    monkeypatch.setattr(
        vllm_utils,
        "_merge_multimodal_embeddings",
        vllm_utils._merge_multimodal_embeddings,
    )
    if hasattr(thinker_mod, "_merge_multimodal_embeddings"):
        monkeypatch.setattr(
            thinker_mod,
            "_merge_multimodal_embeddings",
            thinker_mod._merge_multimodal_embeddings,
        )
    if hasattr(thinker_mod, "check_interleaved_audio_video"):
        monkeypatch.setattr(
            thinker_mod,
            "check_interleaved_audio_video",
            thinker_mod.check_interleaved_audio_video,
        )
    if hasattr(thinker_mod, "merge_interleaved_embeddings"):
        monkeypatch.setattr(
            thinker_mod,
            "merge_interleaved_embeddings",
            thinker_mod.merge_interleaved_embeddings,
        )

    apply_omni_ar_patches.cache_clear()
    try:
        apply_omni_ar_patches()

        # 1. Verify Bucketed Audio Encoder
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

        # 2. Verify Frame-Chunked & Bucketed ViT
        vit = DummyViT()
        out_vit = vit(
            torch.ones(20000, 8), torch.tensor([[5, 40, 100]], dtype=torch.int32)
        )
        assert out_vit.shape == (5000, 16)
        assert vit.blk.seen_shapes == [16384, 4096]

        # 3. Verify CPU-staged Keyword embed_input_ids Preserves Modality
        vid_emb = torch.ones(6, 32)
        vid_emb.modality = "video"
        kw_out = DummyMixin().embed_input_ids(
            input_ids=torch.zeros(5, dtype=torch.long),
            multimodal_embeddings=[vid_emb],
            is_multimodal=torch.ones(5, dtype=torch.bool),
        )
        assert kw_out[0].shape == (6, 32) and kw_out[0].modality == "video"
    finally:
        apply_omni_ar_patches.cache_clear()
