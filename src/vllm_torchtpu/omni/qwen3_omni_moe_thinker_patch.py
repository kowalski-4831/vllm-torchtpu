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
from functools import cache as run_once
from typing import Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.mm_video_chunking import embed_multimodal_chunked
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 500)
_VISION_PATCH_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)
_OMNI_STATIC_PADDED_ATTR = "_is_omni_static_padded"


def _select_bucket(
    size: int,
    buckets: tuple[int, ...] = _AUDIO_CHUNK_BUCKETS,
    align: int = 8,
) -> int:
    for bucket in buckets:
        if size <= bucket:
            return bucket
    return align_to(max(1, size), align)


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        out = tensor.detach().cpu()
        if hasattr(tensor, "modality"):
            out.modality = tensor.modality
        return out
    return tensor


def _pad_rows(tensor: torch.Tensor, padded_rows: int) -> torch.Tensor:
    return F.pad(tensor, (0, 0, 0, padded_rows - tensor.shape[0]))


def _build_padded_cu_seqlens(
    seqlens: list[int], padded_total: int, device: torch.device
) -> torch.Tensor:
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    pad_len = (-cu_seqlens.numel()) % 16
    return F.pad(cu_seqlens, (0, pad_len), value=padded_total + 1).to(device)


@run_once
def apply_omni_ar_patches() -> None:
    try:
        import vllm.model_executor.models.utils as vllm_utils
        from vllm.multimodal.utils import get_mm_embedding_modalities
        from vllm_omni.model_executor.models.qwen3_omni import (
            qwen3_omni_moe_thinker as thinker_mod,
        )
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeAudioEncoder,
        )

        from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings

        def _padded_encoder_forward(
            self: Any,
            input_features: torch.Tensor,
            feature_lens: torch.Tensor,
            aftercnn_lens: torch.Tensor,
        ) -> torch.Tensor:
            window_size = self.n_window * 2
            tokens_per_chunk = (((window_size + 1) // 2 + 1) // 2 + 1) // 2
            window_aftercnn = tokens_per_chunk * (self.n_window_infer // window_size)
            device = self.conv2d1.weight.device
            dtype = self.conv2d1.weight.dtype
            feat_lens = feature_lens.tolist()
            cnn_lens = aftercnn_lens.tolist()
            feature_list = input_features.cpu().to(dtype).split(feat_lens, dim=1)
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
                    .to(device)
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
                outputs.append(hidden_states.cpu()[:cnn_len])
            return torch.cat(outputs, dim=0)

        Qwen3OmniMoeAudioEncoder.forward = _padded_encoder_forward

        vit_cls = getattr(thinker_mod, "Qwen3Omni_VisionTransformer", None)
        if vit_cls is not None:

            def _vit_device(self: Any) -> torch.device:
                return self.pos_embed.weight.device

            vit_cls.device = property(_vit_device)

            def _padded_vit_forward(
                self: Any, x: torch.Tensor, grid_thw: Any
            ) -> torch.Tensor:
                device = self.patch_embed.proj.weight.device
                if device.type == "tpu" and self.pos_embed.weight.device.type == "tpu":
                    self.pos_embed.cpu()
                    self.rotary_pos_emb.cpu()
                grid_cpu = torch.as_tensor(grid_thw).cpu().to(torch.int32)
                x_cpu = x.cpu().to(self.dtype)

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
                        _pad_rows(pixel_values_videos, padded_patches).to(device)
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
                    return [hidden_states.cpu()[:valid_tokens]]

                chunks = embed_multimodal_chunked(
                    _encode_vit_chunk,
                    {"pixel_values_videos": x_cpu, "video_grid_thw": grid_cpu},
                    _VISION_PATCH_BUCKETS[-1],
                )
                return torch.cat(list(chunks), dim=0)

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
                    cpu_embeds = [_to_cpu(embed) for embed in multimodal_embeddings]
                    flat = vllm_utils._flatten_embeddings(cpu_embeds)
                    if 0 < flat.shape[0] < inputs_embeds.shape[0]:
                        multimodal_embeddings = [
                            _pad_rows(flat, inputs_embeds.shape[0])
                        ]
                if is_multimodal is not None:
                    is_multimodal = is_multimodal.to(inputs_embeds.device)
                return base_merge(inputs_embeds, multimodal_embeddings, is_multimodal)

            setattr(_omni_merge, _OMNI_STATIC_PADDED_ATTR, True)
            vllm_utils._merge_multimodal_embeddings = _omni_merge
            thinker_mod._merge_multimodal_embeddings = _omni_merge
            _patch_vllm_merge_multimodal_embeddings()

        active_merge = vllm_utils._merge_multimodal_embeddings
        orig_check = getattr(thinker_mod, "check_interleaved_audio_video", None)
        if orig_check is not None:

            def _check_interleaved(
                is_video: torch.Tensor,
                is_audio: torch.Tensor,
                num_videos: int,
                num_audios: int,
            ) -> bool:
                if not num_videos or not num_audios:
                    return False
                return orig_check(
                    is_video.cpu(), is_audio.cpu(), num_videos, num_audios
                )

            thinker_mod.check_interleaved_audio_video = _check_interleaved

        if hasattr(thinker_mod, "merge_interleaved_embeddings"):

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

            thinker_mod.merge_interleaved_embeddings = _merge_interleaved

        thinker_cls = getattr(
            thinker_mod, "Qwen3OmniMoeThinkerForConditionalGeneration", None
        )
        if thinker_cls is not None and hasattr(thinker_cls, "embed_input_ids"):
            orig_embed_input_ids = thinker_cls.embed_input_ids

            def _patched_embed_input_ids(
                self: Any,
                input_ids: torch.Tensor,
                multimodal_embeddings: Any = None,
                *,
                is_multimodal: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if multimodal_embeddings:
                    multimodal_embeddings = [
                        _to_cpu(embed) for embed in multimodal_embeddings
                    ]
                    if is_multimodal is not None:
                        is_multimodal = _to_cpu(is_multimodal)
                return orig_embed_input_ids(
                    self,
                    input_ids,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )

            thinker_cls.embed_input_ids = _patched_embed_input_ids

        logger.info("Applied TPU patch: Qwen3-Omni Thinker audio/vision encoders.")
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Thinker patch: %s", e)
