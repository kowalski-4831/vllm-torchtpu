# SPDX-License-Identifier: Apache-2.0
"""TPU runtime patches for vLLM-Omni autoregressive (AR) models."""

from functools import cache as run_once
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Audio buckets in 100-frame (1.0s) chunks; vision buckets in patch counts.
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 500)
_VISION_PATCH_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)


def _select_bucket(
    size: int,
    buckets: tuple[int, ...] = _AUDIO_CHUNK_BUCKETS,
    align: int = 8,
) -> int:
    """Return the smallest bucket >= size, or round up to a multiple of align."""
    return next(
        (b for b in buckets if size <= b),
        ((max(1, size) + align - 1) // align) * align,
    )


def _with_modality(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Copy the vLLM `.modality` tag (e.g. 'audio', 'video') from source to target."""
    if hasattr(source, "modality"):
        target.modality = source.modality
    return target


@run_once
def apply_omni_ar_patches() -> None:
    """Patch Qwen3-Omni MoE Thinker audio/vision encoders and attention for TPU."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.attention.mm_encoder_attention import (
            MMEncoderAttention,
        )
        from vllm.model_executor.layers.linear import (
            QKVParallelLinear,
            RowParallelLinear,
        )
        import vllm.model_executor.models.utils as vllm_utils
        from vllm_omni.model_executor.models.qwen3_omni import (
            qwen3_omni_moe_thinker as thinker_mod,
        )
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeAudioAttention,
            Qwen3OmniMoeAudioEncoder,
        )

        from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        # 1. Audio Attention: disable TP when num_heads (20) is not divisible by TP (8)
        #    and use platform-aware MMEncoderAttention instead of CUDA FlashAttention.
        def _audio_attn_init(
            self: Any, config: Any, quant_config: Any = None, prefix: str = ""
        ) -> None:
            nn.Module.__init__(self)
            self.embed_dim = config.d_model
            self.num_heads = config.encoder_attention_heads
            self.head_dim = self.embed_dim // self.num_heads
            tp_size = get_tensor_model_parallel_world_size()
            disable_tp = self.num_heads % tp_size != 0
            self.num_local_heads = (
                self.num_heads if disable_tp else self.num_heads // tp_size
            )
            self.scaling = self.head_dim**-0.5
            linear_kwargs = {
                "bias": True,
                "quant_config": quant_config,
                "disable_tp": disable_tp,
            }
            self.qkv = QKVParallelLinear(
                self.embed_dim,
                self.head_dim,
                self.num_heads,
                self.num_heads,
                prefix=f"{prefix}.qkv",
                **linear_kwargs,
            )
            self.attn = MMEncoderAttention(
                num_heads=self.num_local_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                prefix=f"{prefix}.attn",
            )
            self.out_proj = RowParallelLinear(
                self.embed_dim,
                self.embed_dim,
                prefix=f"{prefix}.out_proj",
                **linear_kwargs,
            )

        Qwen3OmniMoeAudioAttention.__init__ = _audio_attn_init

        # 2. Audio Encoder: use exact CPU integer chunking and static chunk buckets
        #    to avoid float32 ceil rounding errors and XLA recompilations.
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
            feat_lens = [int(x) for x in feature_lens.tolist()]
            cnn_lens = [int(x) for x in aftercnn_lens.tolist()]
            feature_list = (
                input_features.detach().cpu().to(dtype).split(feat_lens, dim=1)
            )
            positional_embedding = (
                self.positional_embedding.positional_embedding[:tokens_per_chunk]
                .unsqueeze(0)
                .to(device=device, dtype=dtype)
            )
            outputs = []
            for feature, feat_len, cnn_len in zip(feature_list, feat_lens, cnn_lens):
                num_chunks = _select_bucket(
                    (feat_len + window_size - 1) // window_size
                )
                seq_pad_len = (-(num_chunks * tokens_per_chunk)) % 128
                remainder = cnn_len % window_aftercnn
                cu_chunk_lens = (
                    [0]
                    + [window_aftercnn] * (cnn_len // window_aftercnn)
                    + ([remainder] if remainder else [])
                )
                cu_seqlens = torch.tensor(cu_chunk_lens, dtype=torch.int32).cumsum(
                    -1, dtype=torch.int32
                )
                cu_seqlens = F.pad(
                    cu_seqlens,
                    (0, (-cu_seqlens.numel()) % 16),
                    value=num_chunks * tokens_per_chunk + seq_pad_len + 1,
                ).to(device)
                padded_feature = (
                    F.pad(feature, (0, num_chunks * window_size - feat_len))
                    .T.reshape(num_chunks, window_size, -1)
                    .transpose(1, 2)
                    .unsqueeze(1)
                    .contiguous()
                    .to(device)
                )
                padded_embeds = [
                    F.gelu(self.conv2d3(F.gelu(self.conv2d2(F.gelu(self.conv2d1(c))))))
                    for c in padded_feature.split(
                        getattr(self, "conv_chunksize", 500), dim=0
                    )
                ]
                hidden_states, _ = self.conv_out(
                    torch.cat(padded_embeds, dim=0)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .view(num_chunks, tokens_per_chunk, -1)
                )
                hidden_states = F.pad(
                    (hidden_states + positional_embedding).reshape(
                        num_chunks * tokens_per_chunk, -1
                    ),
                    (0, 0, 0, seq_pad_len),
                )
                for encoder_layer in self.layers:
                    hidden_states = encoder_layer(
                        hidden_states, cu_seqlens, max_seqlen=max(cu_chunk_lens)
                    )
                hidden_states = self.proj2(
                    self.act(self.proj1(self.ln_post(hidden_states))[0])
                )[0]
                outputs.append(hidden_states.cpu()[:cnn_len])
            output = torch.cat(outputs, dim=0)
            return output if device.type == "tpu" else output.to(device)

        Qwen3OmniMoeAudioEncoder.forward = _padded_encoder_forward

        # 3. Vision Transformer: chunk long videos at frame boundaries and pad each
        #    chunk to static patch buckets to prevent HBM OOM and recompilation.
        vit_cls = getattr(thinker_mod, "Qwen3Omni_VisionTransformer", None)
        if vit_cls is not None:
            vit_cls.device = property(lambda self: self.pos_embed.weight.device)

            def _padded_vit_forward(
                self: Any, x: torch.Tensor, grid_thw: Any
            ) -> torch.Tensor:
                device = self.patch_embed.proj.weight.device
                dtype = self.dtype
                if device.type == "tpu" and self.pos_embed.weight.device.type == "tpu":
                    self.pos_embed.cpu()
                    self.rotary_pos_emb.cpu()
                grid_cpu = (
                    grid_thw.detach().cpu().to(torch.int32)
                    if isinstance(grid_thw, torch.Tensor)
                    else torch.as_tensor(grid_thw, dtype=torch.int32)
                )
                grid_list = [[int(v) for v in row] for row in grid_cpu.tolist()]
                pos_embeds = (
                    self.fast_pos_embed_interpolate(grid_list).cpu()
                    if self.apply_vit_abs_pos_embed
                    else None
                )
                rotary_cos, rotary_sin = (t.cpu() for t in self.rot_pos_emb(grid_cpu))
                x_cpu = x.detach().cpu().to(dtype)

                frame_seqlens = torch.repeat_interleave(
                    grid_cpu[:, 1] * grid_cpu[:, 2], grid_cpu[:, 0]
                ).tolist()
                frame_chunks: list[list[int]] = []
                for seqlen in frame_seqlens:
                    if (
                        frame_chunks
                        and sum(frame_chunks[-1]) + seqlen <= _VISION_PATCH_BUCKETS[-1]
                    ):
                        frame_chunks[-1].append(seqlen)
                    else:
                        frame_chunks.append([seqlen])

                outputs, offset = [], 0
                deepstack_indexes = self.deepstack_visual_indexes
                for chunk_seqlens in frame_chunks:
                    num_patches = sum(chunk_seqlens)
                    padded_patches = _select_bucket(
                        num_patches, _VISION_PATCH_BUCKETS, align=512
                    )
                    patch_pad = (0, 0, 0, padded_patches - num_patches)
                    cos = F.pad(
                        rotary_cos[offset : offset + num_patches], patch_pad
                    ).to(device)
                    sin = F.pad(
                        rotary_sin[offset : offset + num_patches], patch_pad
                    ).to(device)
                    hidden_states = self.patch_embed(
                        F.pad(x_cpu[offset : offset + num_patches], patch_pad).to(
                            device
                        )
                    )
                    if pos_embeds is not None:
                        hidden_states = hidden_states + F.pad(
                            pos_embeds[offset : offset + num_patches], patch_pad
                        ).to(device=device, dtype=dtype)
                    hidden_states = hidden_states.unsqueeze(1)
                    cu_seqlens = F.pad(
                        torch.tensor(chunk_seqlens, dtype=torch.int32).cumsum(
                            0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    cu_seqlens = F.pad(
                        cu_seqlens,
                        (0, (-cu_seqlens.numel()) % 16),
                        value=padded_patches + 1,
                    ).to(device)
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
                        if (
                            deepstack_indexes is not None
                            and layer_idx in deepstack_indexes
                        ):
                            deepstack_features.append(hidden_states)
                    hidden_states = self.merger(hidden_states)
                    if deepstack_indexes is not None:
                        hidden_states = torch.cat(
                            [hidden_states]
                            + [
                                self.merger_list[i](feat)
                                for i, feat in enumerate(deepstack_features)
                            ],
                            dim=1,
                        )
                    outputs.append(
                        hidden_states.cpu()[: num_patches // self.spatial_merge_unit]
                    )
                    offset += num_patches
                output = torch.cat(outputs, dim=0)
                return output if device.type == "tpu" else output.to(device)

            vit_cls.forward = _padded_vit_forward

        # 4. Multimodal Input Preprocessing: keep metadata & features on CPU before
        #    calling the bucketed encoders to avoid XLA dynamic-split syncs.
        mixin_cls = getattr(
            thinker_mod, "Qwen3OmniMoeConditionalGenerationMixin", None
        )
        if mixin_cls is not None:
            orig_process_audio = mixin_cls._process_audio_input
            mixin_cls._process_audio_input = (
                lambda self, audio_input: orig_process_audio(
                    self,
                    {
                        "input_features": audio_input["input_features"].detach().cpu(),
                        "audio_feature_lengths": audio_input["audio_feature_lengths"]
                        .detach()
                        .cpu(),
                    },
                )
            )
            if orig_process_video := getattr(mixin_cls, "_process_video_input", None):
                mixin_cls._process_video_input = (
                    lambda self, video_input: orig_process_video(
                        self,
                        {
                            "type": video_input["type"],
                            "pixel_values_videos": video_input["pixel_values_videos"]
                            .detach()
                            .cpu(),
                            "video_grid_thw": video_input["video_grid_thw"]
                            .detach()
                            .cpu(),
                        }
                        if video_input["type"] == "pixel_values_videos"
                        else {
                            "type": video_input["type"],
                            "video_embeds": video_input["video_embeds"].detach().cpu(),
                            "video_grid_thw": video_input["video_grid_thw"]
                            .detach()
                            .cpu(),
                        },
                    )
                )
            if orig_process_image := getattr(mixin_cls, "_process_image_input", None):
                mixin_cls._process_image_input = (
                    lambda self, image_input: orig_process_image(
                        self,
                        {
                            "type": image_input["type"],
                            "pixel_values": image_input["pixel_values"].detach().cpu(),
                            "image_grid_thw": image_input["image_grid_thw"]
                            .detach()
                            .cpu(),
                        }
                        if image_input["type"] == "pixel_values"
                        else image_input,
                    )
                )

        # 5. Static Multimodal Embedding Merge: pad flattened multimodal embeddings
        #    to inputs_embeds.shape[0] so torch.gather sees static shapes on TPU.
        _patch_vllm_merge_multimodal_embeddings()
        base_merge = vllm_utils._merge_multimodal_embeddings
        if not getattr(base_merge, "_is_omni_static_padded", False):

            def _omni_merge(
                inputs_embeds: torch.Tensor,
                multimodal_embeddings: Any,
                is_multimodal: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if len(multimodal_embeddings) > 0:
                    cpu_embeds = [
                        embed.detach().cpu() if embed.device.type == "tpu" else embed
                        for embed in multimodal_embeddings
                    ]
                    flattened_embeds = vllm_utils._flatten_embeddings(cpu_embeds)
                    if 0 < flattened_embeds.shape[0] < inputs_embeds.shape[0]:
                        pad_rows = inputs_embeds.shape[0] - flattened_embeds.shape[0]
                        multimodal_embeddings = [
                            F.pad(flattened_embeds, (0, 0, 0, pad_rows))
                        ]
                if (
                    is_multimodal is not None
                    and is_multimodal.device != inputs_embeds.device
                ):
                    is_multimodal = is_multimodal.to(inputs_embeds.device)
                return base_merge(inputs_embeds, multimodal_embeddings, is_multimodal)

            _omni_merge._is_omni_static_padded = True  # type: ignore[attr-defined]
            vllm_utils._merge_multimodal_embeddings = _omni_merge
            for module_name, module in list(sys.modules.items()):
                if (
                    module is not None
                    and module_name.startswith(
                        (
                            "vllm.model_executor.models",
                            "vllm_omni.model_executor.models",
                        )
                    )
                    and hasattr(module, "_merge_multimodal_embeddings")
                ):
                    module._merge_multimodal_embeddings = _omni_merge

        # 6. Interleaved Audio+Video & CPU-Staged embed_input_ids: run non-zero mask
        #    checks and deepstack feature splitting on CPU, preserving `.modality`.
        active_merge = vllm_utils._merge_multimodal_embeddings
        orig_check = getattr(thinker_mod, "check_interleaved_audio_video", None)
        if orig_check is not None:
            thinker_mod.check_interleaved_audio_video = (
                lambda is_video, is_audio, num_videos, num_audios: orig_check(
                    is_video.detach().cpu(),
                    is_audio.detach().cpu(),
                    num_videos,
                    num_audios,
                )
                if num_videos and num_audios
                else False
            )

        if hasattr(thinker_mod, "merge_interleaved_embeddings"):

            def _merge_interleaved(
                inputs_embeds: torch.Tensor,
                multimodal_embeddings: Any,
                is_video: torch.Tensor,
                is_audio: torch.Tensor,
                is_multimodal: torch.Tensor,
            ) -> torch.Tensor:
                from vllm.multimodal.utils import get_mm_embedding_modalities

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

            def _cpu_mm_embed(
                self: Any,
                input_ids: torch.Tensor,
                multimodal_embeddings: Any = None,
                *,
                is_multimodal: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if multimodal_embeddings:
                    multimodal_embeddings = [
                        _with_modality(
                            embed.detach().cpu()
                            if embed.device.type == "tpu"
                            else embed,
                            embed,
                        )
                        for embed in multimodal_embeddings
                    ]
                    if is_multimodal is not None and is_multimodal.device.type == "tpu":
                        is_multimodal = is_multimodal.detach().cpu()
                return orig_embed_input_ids(
                    self,
                    input_ids,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )

            thinker_cls.embed_input_ids = _cpu_mm_embed

        # 7. Mixed-Dim Multimodal Chunk Slicing: prevent TPUModelRunner.execute_model
        #    from calling torch.cat on mixed 8192-d video and 2048-d audio tensors.
        if not getattr(
            TPUModelRunner._gather_mm_embeddings, "_is_omni_patched", False
        ):
            orig_gather = TPUModelRunner._gather_mm_embeddings
            orig_get_inputs = TPUModelRunner._get_model_inputs

            def _omni_gather_mm(
                self: Any, *args: Any, **kwargs: Any
            ) -> tuple[list[torch.Tensor], torch.Tensor]:
                mm_embeds, is_multimodal = orig_gather(self, *args, **kwargs)
                self._omni_mm_embeds_list = mm_embeds
                if mm_embeds:
                    total_tokens = sum(int(embed.shape[0]) for embed in mm_embeds)
                    return (
                        [torch.arange(total_tokens, dtype=torch.int64)],
                        is_multimodal,
                    )
                return mm_embeds, is_multimodal

            def _omni_get_model_inputs(
                self: Any,
                input_ids: torch.Tensor,
                mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
            ) -> Any:
                if mm_embed_inputs is not None:
                    chunk_embeds, is_multimodal = mm_embed_inputs
                    mm_embeds = getattr(self, "_omni_mm_embeds_list", None)
                    if chunk_embeds and mm_embeds and chunk_embeds[0].ndim == 1:
                        chunk_start = int(chunk_embeds[0][0].item())
                        chunk_end = chunk_start + int(chunk_embeds[0].shape[0])
                        sliced_embeds, offset = [], 0
                        for embed in mm_embeds:
                            num_rows = int(embed.shape[0])
                            start = max(0, chunk_start - offset)
                            end = min(num_rows, chunk_end - offset)
                            if start < end:
                                embed_slice = (
                                    embed
                                    if (start == 0 and end == num_rows)
                                    else embed[start:end]
                                )
                                sliced_embeds.append(
                                    _with_modality(embed_slice, embed)
                                )
                            offset += num_rows
                        mm_embed_inputs = (sliced_embeds, is_multimodal)
                return orig_get_inputs(self, input_ids, mm_embed_inputs)

            _omni_gather_mm._is_omni_patched = True  # type: ignore[attr-defined]
            TPUModelRunner._gather_mm_embeddings = _omni_gather_mm
            TPUModelRunner._get_model_inputs = _omni_get_model_inputs

        logger.info(
            "Applied TPU patch: Qwen3-Omni Thinker audio/vision encoders and attention."
        )
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Thinker patch: %s", e)
