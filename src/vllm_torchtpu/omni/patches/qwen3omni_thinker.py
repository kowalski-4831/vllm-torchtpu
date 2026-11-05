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
"""TPU patches for the Qwen3-Omni Thinker encoders and multimodal merge.

1. Audio encoder chunk-length math & static-shape bucketing:
   ``Qwen3OmniMoeAudioEncoder.forward`` (vLLM-Omni's thinker, a subclass of the
   vLLM implementation) splits the mel features into windows of
   ``n_window * 2 = 100`` frames with

       chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

   and then calls ``input_features.T.split(chunk_lengths)`` with one entry per
   window. On TPU the float division lowers to a multiplication by the
   reciprocal of 100, which float32 cannot represent exactly
   (0.01000000074505806), so every exact multiple of the window rounds up:
   800 * 0.01000000074505806 = 8.000001, ``torch.ceil`` yields 9 windows for an
   8 s clip, the window lengths add up to 900 frames instead of 800, and
   ``split_with_sizes`` raises. On TPU v6e every multiple of 100 frames up to
   100000 is affected and no other length is. CPU evaluates the same expression
   exactly. In addition, unpadded chunk and token counts trigger an XLA
   recompilation for every distinct audio length. The patched ``forward`` moves
   ``feature_lens`` and ``aftercnn_lens`` to CPU, computes window counts with
   exact integer arithmetic, pads chunk counts to ``_AUDIO_CHUNK_BUCKETS`` and
   token counts to multiples of 128, and slices back to ``cnn_len`` valid
   tokens per clip. ``_get_cnn_output_lengths`` is wrapped so the CNN mask
   lengths computed from CPU chunk lengths come back on the encoder's device.

2. Vision encoder frame-chunking & static-shape bucketing:
   ``Qwen3Omni_VisionTransformer.forward`` chunks video frames with
   ``embed_multimodal_chunked`` and pads patch counts to
   ``_VISION_PATCH_BUCKETS`` while keeping 3-int ``grid_thw`` metadata on CPU
   and all weights, submodules, and activations on TPU.

3. Multimodal embedding merge:
   Pads flattened multimodal embeddings to ``inputs_embeds.shape[0]`` for XLA
   static shapes while keeping ``multimodal_embeddings`` on TPU and moving only
   1D boolean masks (``is_video``, ``is_audio``, ``is_multimodal``) to CPU
   where upstream GPU already indexes on host.

Installed through ``patch_registry.PATCHES`` at ``platform_activation`` and
``model_load`` for ``Qwen3OmniMoeForConditionalGeneration`` only. The thinker
stage runs in a ``TPUWorker`` (``TPUARWorker``), which drives both stages; the
callback returns ``False`` for every other architecture so the registry
retries it once a model config is known.

TODO: remove the CPU chunk-length hop once vLLM-Omni computes ``chunk_num``
with integer ceil-division, ``(feature_lens + window - 1) // window``, in
https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
or once TPU float division is exact for these lengths. The canary
``test_tpu_float_ceil_still_inexact_for_window_multiples`` in
``tests_omni/test_qwen3_omni_audio_encoder_tpu.py`` fails when the second
condition holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.mm_video_chunking import embed_multimodal_chunked
from vllm_torchtpu.utils import align_to

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence markers on wrapped methods/functions. The registry runs a
# callback once per process; the markers cover forks and direct calls from
# tests.
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"
_OMNI_STATIC_PADDED_ATTR = "_is_omni_static_padded"

_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 500)
_VISION_PATCH_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)


def _select_bucket(
    size: int,
    buckets: tuple[int, ...] = _AUDIO_CHUNK_BUCKETS,
    align: int = 8,
) -> int:
    for bucket in buckets:
        if size <= bucket:
            return bucket
    return align_to(max(1, size), align)


def _pad_rows(tensor: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if tensor.shape[0] >= padded_rows:
        return tensor
    return F.pad(tensor, (0, 0, 0, padded_rows - tensor.shape[0]))


def _build_padded_cu_seqlens(
    seqlens: list[int], padded_total: int, device: torch.device
) -> torch.Tensor:
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    pad_len = (-cu_seqlens.numel()) % 16
    return F.pad(cu_seqlens, (0, pad_len), value=padded_total + 1).to(device)


def maybe_patch_qwen3_omni_thinker(model_config: ModelConfig | None = None) -> bool:
    """Apply TPU patches for Qwen3-Omni Thinker encoders and multimodal merge.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``ModelConfig.architectures`` (for a vLLM-Omni stage that is
    the pipeline's ``model_arch``, which is this name for every Qwen3-Omni
    stage). Returns ``False`` without importing vLLM-Omni for anything else,
    including the ``None`` that ``platform_activation`` passes before a model
    is configured, so the registry does not record the patch as applied and
    retries at ``model_load``. For a Qwen3-Omni model the import is
    unconditional: vLLM-Omni is a hard requirement of that model and an import
    error must surface.
    """
    if model_config is None or _QWEN3_OMNI_ARCH not in model_config.architectures:
        return False

    # Imported in the callback, not at module top: the registry imports this
    # module in every process, including ones without the omni extra.
    import vllm.model_executor.models.utils as vllm_utils
    from vllm.multimodal.utils import get_mm_embedding_modalities
    from vllm_omni.model_executor.models.qwen3_omni import (
        qwen3_omni_moe_thinker as thinker_mod,
    )

    from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings

    audio_cls = thinker_mod.Qwen3OmniMoeAudioEncoder
    orig_forward = audio_cls.forward
    if not getattr(orig_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False):

        def _padded_encoder_forward(
            self: Any,
            input_features: torch.Tensor,
            feature_lens: torch.Tensor,
            aftercnn_lens: torch.Tensor,
        ) -> torch.Tensor:
            """Bucketed audio encoder forward with ``feature_lens`` on CPU."""
            feature_lens = feature_lens.cpu()
            if not hasattr(self, "layers"):
                return orig_forward(self, input_features, feature_lens, aftercnn_lens)
            aftercnn_lens = aftercnn_lens.cpu()
            window_size = self.n_window * 2
            tokens_per_chunk = (((window_size + 1) // 2 + 1) // 2 + 1) // 2
            window_aftercnn = tokens_per_chunk * (self.n_window_infer // window_size)
            device = self.conv2d1.weight.device
            dtype = self.conv2d1.weight.dtype
            feat_lens = feature_lens.tolist()
            cnn_lens = aftercnn_lens.tolist()
            feature_list = input_features.to(device=device, dtype=dtype).split(
                feat_lens, dim=1
            )
            pos_emb = (
                self.positional_embedding.positional_embedding[:tokens_per_chunk]
                .unsqueeze(0)
                .to(device=device, dtype=dtype)
            )
            outputs = []
            for feature, feat_len, cnn_len in zip(feature_list, feat_lens, cnn_lens):
                num_chunks = _select_bucket(
                    align_to(feat_len, window_size) // window_size
                )
                padded_tokens = align_to(num_chunks * tokens_per_chunk, 128)
                chunk_seqlens = [window_aftercnn] * (cnn_len // window_aftercnn)
                if cnn_len % window_aftercnn:
                    chunk_seqlens.append(cnn_len % window_aftercnn)
                cu_seqlens = _build_padded_cu_seqlens(
                    chunk_seqlens, padded_tokens, device
                )
                padded_feature = (
                    F.pad(feature, (0, num_chunks * window_size - feat_len))
                    .T.reshape(num_chunks, window_size, -1)
                    .transpose(1, 2)
                    .unsqueeze(1)
                    .contiguous()
                )
                conv_chunks = [
                    F.gelu(self.conv2d3(F.gelu(self.conv2d2(F.gelu(self.conv2d1(c))))))
                    for c in padded_feature.split(self.conv_chunksize, dim=0)
                ]
                conv_out_in = (
                    torch.cat(conv_chunks, dim=0)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .view(num_chunks, tokens_per_chunk, -1)
                )
                hidden_states, _ = self.conv_out(conv_out_in)
                hidden_states = _pad_rows(
                    (hidden_states + pos_emb).reshape(
                        num_chunks * tokens_per_chunk, -1
                    ),
                    padded_tokens,
                )
                for encoder_layer in self.layers:
                    hidden_states = encoder_layer(
                        hidden_states, cu_seqlens, max_seqlen=max(chunk_seqlens)
                    )
                hidden_states, _ = self.proj1(self.ln_post(hidden_states))
                hidden_states, _ = self.proj2(self.act(hidden_states))
                outputs.append(hidden_states[:cnn_len])
            return torch.cat(outputs, dim=0)

        setattr(_padded_encoder_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, True)
        audio_cls.forward = _padded_encoder_forward

        orig_get_cnn_output_lengths = audio_cls._get_cnn_output_lengths

        def _get_cnn_output_lengths_on_device(
            self: Any, input_lengths: torch.Tensor
        ) -> torch.Tensor:
            """Upstream ``_get_cnn_output_lengths`` moved to the encoder's device."""
            return orig_get_cnn_output_lengths(self, input_lengths).to(
                self.conv2d1.weight.device
            )

        audio_cls._get_cnn_output_lengths = _get_cnn_output_lengths_on_device

    vit_cls = thinker_mod.Qwen3Omni_VisionTransformer
    if not getattr(vit_cls.forward, _OMNI_STATIC_PADDED_ATTR, False):

        def _padded_vit_forward(
            self: Any, x: torch.Tensor, grid_thw: torch.Tensor | list[list[int]]
        ) -> torch.Tensor:
            device = self.patch_embed.proj.weight.device
            x_dev = x.to(device=device, dtype=self.dtype)
            # Matches GPU path (`grid_thw.cpu().numpy()`): keep 3-int grid
            # metadata on CPU while weights and activations stay on TPU.
            grid_cpu = torch.as_tensor(grid_thw).cpu().to(torch.int32)

            def _encode_vit_chunk(
                pixel_values_videos: torch.Tensor,
                video_grid_thw: torch.Tensor,
            ) -> list[torch.Tensor]:
                chunk_seqlens = torch.repeat_interleave(
                    video_grid_thw[:, 1] * video_grid_thw[:, 2],
                    video_grid_thw[:, 0],
                ).tolist()
                num_patches = sum(chunk_seqlens)
                padded_patches = _select_bucket(
                    num_patches, _VISION_PATCH_BUCKETS, align=512
                )
                rotary_cos, rotary_sin = self.rot_pos_emb(video_grid_thw)
                cos = _pad_rows(rotary_cos, padded_patches).to(device)
                sin = _pad_rows(rotary_sin, padded_patches).to(device)
                hidden_states = self.patch_embed(
                    _pad_rows(pixel_values_videos, padded_patches)
                )
                if self.apply_vit_abs_pos_embed:
                    pos_embeds = self.fast_pos_embed_interpolate(
                        video_grid_thw.tolist()
                    )
                    hidden_states = hidden_states + _pad_rows(
                        pos_embeds, padded_patches
                    ).to(device=device, dtype=self.dtype)
                hidden_states = hidden_states.unsqueeze(1)
                cu_seqlens = _build_padded_cu_seqlens(
                    chunk_seqlens, padded_patches, device
                )
                deepstack_indexes = self.deepstack_visual_indexes or ()
                deepstack_features = []
                for layer_idx, block in enumerate(self.blocks):
                    hidden_states = block(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        rotary_pos_emb_cos=cos,
                        rotary_pos_emb_sin=sin,
                        max_seqlen=max(chunk_seqlens),
                        sequence_lengths=None,
                    )
                    if layer_idx in deepstack_indexes:
                        deepstack_features.append(hidden_states)
                merged = [self.merger(hidden_states)]
                for merger, feat in zip(self.merger_list, deepstack_features):
                    merged.append(merger(feat))
                hidden_states = torch.cat(merged, dim=1)
                valid_tokens = num_patches // self.spatial_merge_unit
                return [hidden_states[:valid_tokens]]

            chunks = embed_multimodal_chunked(
                _encode_vit_chunk,
                {"pixel_values_videos": x_dev, "video_grid_thw": grid_cpu},
                _VISION_PATCH_BUCKETS[-1],
            )
            return torch.cat(list(chunks), dim=0)

        setattr(_padded_vit_forward, _OMNI_STATIC_PADDED_ATTR, True)
        vit_cls.forward = _padded_vit_forward

    _patch_vllm_merge_multimodal_embeddings()
    base_merge = vllm_utils._merge_multimodal_embeddings
    if not getattr(base_merge, _OMNI_STATIC_PADDED_ATTR, False):

        def _omni_merge(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: Any,
            is_multimodal: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if len(multimodal_embeddings) > 0:
                flat = vllm_utils._flatten_embeddings(multimodal_embeddings)
                if 0 < flat.shape[0] < inputs_embeds.shape[0]:
                    multimodal_embeddings = [_pad_rows(flat, inputs_embeds.shape[0])]
                else:
                    multimodal_embeddings = [flat]
            if is_multimodal is not None:
                is_multimodal = is_multimodal.to(inputs_embeds.device)
            return base_merge(inputs_embeds, multimodal_embeddings, is_multimodal)

        setattr(_omni_merge, _OMNI_STATIC_PADDED_ATTR, True)
        vllm_utils._merge_multimodal_embeddings = _omni_merge
        _patch_vllm_merge_multimodal_embeddings()

    active_merge = vllm_utils._merge_multimodal_embeddings
    thinker_mod._merge_multimodal_embeddings = active_merge

    orig_check = thinker_mod.check_interleaved_audio_video
    if not getattr(orig_check, _OMNI_STATIC_PADDED_ATTR, False):

        def _check_interleaved(
            is_video: torch.Tensor,
            is_audio: torch.Tensor,
            num_videos: int,
            num_audios: int,
        ) -> bool:
            if not num_videos or not num_audios:
                return False
            # Matches upstream GPU path (`input_ids.cpu()` before span check).
            return orig_check(is_video.cpu(), is_audio.cpu(), num_videos, num_audios)

        setattr(_check_interleaved, _OMNI_STATIC_PADDED_ATTR, True)
        thinker_mod.check_interleaved_audio_video = _check_interleaved

    if not getattr(
        thinker_mod.merge_interleaved_embeddings, _OMNI_STATIC_PADDED_ATTR, False
    ):

        def _merge_interleaved(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: Any,
            is_video: torch.Tensor,
            is_audio: torch.Tensor,
            is_multimodal: torch.Tensor,
        ) -> torch.Tensor:
            is_image = is_multimodal & ~is_video & ~is_audio
            modalities = get_mm_embedding_modalities(multimodal_embeddings)
            for modality_mask, modality_name in (
                (is_video, "video"),
                (is_audio, "audio"),
                (is_image, "image"),
            ):
                modality_embeds = [
                    embed
                    for embed, mod in zip(multimodal_embeddings, modalities)
                    if mod == modality_name
                ]
                if modality_embeds:
                    inputs_embeds = active_merge(
                        inputs_embeds, modality_embeds, modality_mask
                    )
            return inputs_embeds

        setattr(_merge_interleaved, _OMNI_STATIC_PADDED_ATTR, True)
        thinker_mod.merge_interleaved_embeddings = _merge_interleaved

    thinker_cls = thinker_mod.Qwen3OmniMoeThinkerForConditionalGeneration
    if not getattr(thinker_cls.embed_input_ids, _OMNI_STATIC_PADDED_ATTR, False):
        orig_embed_input_ids = thinker_cls.embed_input_ids

        def _patched_embed_input_ids(
            self: Any,
            input_ids: torch.Tensor,
            multimodal_embeddings: Any = None,
            *,
            is_multimodal: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if multimodal_embeddings:
                multimodal_embeddings = list(multimodal_embeddings)
            # Matches GPUModelRunner, where `is_mm_embed` stays on CPU so
            # `torch.nonzero(is_multimodal)` runs on host while embeddings
            # stay on device.
            if is_multimodal is not None and is_multimodal.device.type != "cpu":
                is_multimodal = is_multimodal.cpu()
            return orig_embed_input_ids(
                self,
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        setattr(_patched_embed_input_ids, _OMNI_STATIC_PADDED_ATTR, True)
        thinker_cls.embed_input_ids = _patched_embed_input_ids

    logger.info("Applied TPU patch: Qwen3-Omni Thinker encoders and multimodal merge.")
    return True
