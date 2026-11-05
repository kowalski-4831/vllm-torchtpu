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
"""Registry gating and wiring of the Qwen3-Omni Thinker patches.

The on-TPU behaviour of the patched encoder (exact window multiples, token
counts, CPU parity) is covered by test_qwen3_omni_audio_encoder_tpu.py; these
tests check the callback contract the patch registry relies on, the two
device moves the audio encoder patch performs, and the static-shape bucketing
and multimodal merge wrappers on CPU with the encoder's methods stubbed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.model_executor.models.utils as vllm_utils
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni_moe_thinker as thinker_mod,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    _select_bucket,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni


def _model_config(*architectures: str) -> SimpleNamespace:
    """The only ModelConfig field the callback reads."""
    return SimpleNamespace(architectures=list(architectures))


OMNI_THINKER = _model_config("Qwen3OmniMoeForConditionalGeneration")


def test_non_omni_config_is_not_consumed():
    # ``False`` (not ``None``) tells the registry to retry at model_load. The
    # match is the literal architecture: a substring such as the thinker's own
    # class name does not count.
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    assert maybe_patch_qwen3_omni_thinker(_model_config("LlamaForCausalLM")) is False
    assert (
        maybe_patch_qwen3_omni_thinker(
            _model_config("Qwen3OmniMoeThinkerForConditionalGeneration")
        )
        is False
    )
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_qwen3_omni_audio_chunk_lengths_patch(monkeypatch):
    seen = {}

    def _forward(self, input_features, feature_lens, aftercnn_lens):
        seen["feature_lens_device"] = feature_lens.device.type
        return input_features, feature_lens, aftercnn_lens

    # Upstream forward runs the whole encoder. Both class attributes the patch
    # rebinds are restored after the test.
    monkeypatch.setattr(Qwen3OmniMoeAudioEncoder, "forward", _forward)
    monkeypatch.setattr(
        Qwen3OmniMoeAudioEncoder, "_get_cnn_output_lengths", lambda self, x: x
    )

    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    patched = Qwen3OmniMoeAudioEncoder.forward
    assert patched is not _forward
    assert getattr(patched, _AUDIO_CHUNK_LENGTHS_PATCHED, False)

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    enc = SimpleNamespace(conv2d1=SimpleNamespace(weight=torch.empty(0)))
    lens = torch.tensor([800]).as_subclass(TrackedTensor)
    _, out_lens, _ = patched(enc, torch.randn(128, 800), lens, torch.tensor([104]))
    # chunk_num's ceil(feature_lens / window) runs on CPU, where it is exact.
    assert lens.cpu_called is True
    assert seen["feature_lens_device"] == "cpu"
    assert out_lens.device.type == "cpu"
    # ... and the CNN lengths come back on the encoder's device.
    out = Qwen3OmniMoeAudioEncoder._get_cnn_output_lengths(enc, torch.tensor([100]))
    assert out.device == enc.conv2d1.weight.device
    assert torch.equal(out, torch.tensor([100]))

    # Idempotent: a second application keeps the same wrapper.
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    assert Qwen3OmniMoeAudioEncoder.forward is patched


def test_maybe_patch_qwen3_omni_thinker_static_bucketing_and_mm_merge(monkeypatch):
    seen_layer_seq_lens: list[int] = []

    class DummyLinear(nn.Module):
        def forward(self, x):
            return x, None

    class DummyEncoderLayer(nn.Module):
        def forward(self, hs, cu_seqlens, max_seqlen=None):
            seen_layer_seq_lens.append(int(hs.shape[0]))
            return hs

    class DummyAudioEncoder(nn.Module):
        n_window = 50
        n_window_infer = 400
        conv_chunksize = 16

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

        def _get_cnn_output_lengths(self, input_lengths):
            return input_lengths

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

        @property
        def device(self) -> torch.device:
            return self.patch_embed.proj.weight.device

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
            return multimodal_embeddings, is_multimodal

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
    monkeypatch.setattr(
        thinker_mod,
        "_merge_multimodal_embeddings",
        thinker_mod._merge_multimodal_embeddings,
    )
    monkeypatch.setattr(
        thinker_mod,
        "check_interleaved_audio_video",
        thinker_mod.check_interleaved_audio_video,
    )
    monkeypatch.setattr(
        thinker_mod,
        "merge_interleaved_embeddings",
        thinker_mod.merge_interleaved_embeddings,
    )

    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True

    # Submodules must never be moved to host CPU.
    with patch.object(
        nn.Module,
        "cpu",
        side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
    ):
        # 1. Verify Bucketed Audio Encoder (including exact 800-frame boundary)
        enc = DummyAudioEncoder()
        for flen, alen in ((220, 29), (310, 41), (670, 87), (800, 104)):
            out = enc(
                torch.zeros(16, flen),
                torch.tensor([flen], dtype=torch.long),
                torch.tensor([alen], dtype=torch.long),
            )
            assert out.shape == (alen, 8)
        assert seen_layer_seq_lens == [128, 128, 128, 128]
        assert _select_bucket(5, (8, 16), align=8) == 8
        assert _select_bucket(19, (8, 16), align=8) == 24

        # 2. Verify Frame-Chunked & Bucketed ViT (keeps device/submodules intact)
        vit = DummyViT()
        out_vit = vit(
            torch.ones(20000, 8), torch.tensor([[5, 40, 100]], dtype=torch.int32)
        )
        assert out_vit.shape == (5000, 16)
        assert vit.blk.seen_shapes == [16384, 4096]
        assert vit.device == vit.patch_embed.proj.weight.device

        # 3. Verify embed_input_ids keeps embeddings on-device & preserves modality
        vid_emb = torch.ones(6, 32)
        vid_emb.modality = "video"
        kw_embeds, kw_mask = DummyMixin().embed_input_ids(
            input_ids=torch.zeros(5, dtype=torch.long),
            multimodal_embeddings=(vid_emb,),
            is_multimodal=torch.ones(5, dtype=torch.bool),
        )
        assert kw_embeds[0] is vid_emb
        assert kw_embeds[0].modality == "video"
        assert kw_mask.device.type == "cpu"
