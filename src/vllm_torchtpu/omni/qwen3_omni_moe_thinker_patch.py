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


def _with_modality(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Copy the vLLM `.modality` tag (e.g. 'audio', 'video') from src to dst."""
    if hasattr(src, "modality"):
        dst.modality = src.modality
    return dst


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
            no_tp = self.num_heads % tp_size != 0
            self.num_local_heads = (
                self.num_heads if no_tp else self.num_heads // tp_size
            )
            self.scaling = self.head_dim**-0.5
            linear_kwargs = {
                "bias": True,
                "quant_config": quant_config,
                "disable_tp": no_tp,
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
            win = self.n_window * 2
            t_chunk = (((win + 1) // 2 + 1) // 2 + 1) // 2
            win_cnn = t_chunk * (self.n_window_infer // win)
            dev = self.conv2d1.weight.device
            dtype = self.conv2d1.weight.dtype
            flens = [int(v) for v in feature_lens.tolist()]
            alens = [int(v) for v in aftercnn_lens.tolist()]
            feats = input_features.detach().cpu().to(dtype).split(flens, dim=1)
            pos_emb = (
                self.positional_embedding.positional_embedding[:t_chunk]
                .unsqueeze(0)
                .to(device=dev, dtype=dtype)
            )
            outputs = []
            for feat, flen, alen in zip(feats, flens, alens):
                bucket = _select_bucket((flen + win - 1) // win)
                seq_pad = (-(bucket * t_chunk)) % 128
                rem = alen % win_cnn
                cu_lens = (
                    [0] + [win_cnn] * (alen // win_cnn) + ([rem] if rem else [])
                )
                cu = torch.tensor(cu_lens, dtype=torch.int32).cumsum(
                    -1, dtype=torch.int32
                )
                cu = F.pad(
                    cu,
                    (0, (-cu.numel()) % 16),
                    value=bucket * t_chunk + seq_pad + 1,
                ).to(dev)
                conv_in = (
                    F.pad(feat, (0, bucket * win - flen))
                    .T.reshape(bucket, win, -1)
                    .transpose(1, 2)
                    .unsqueeze(1)
                    .contiguous()
                    .to(dev)
                )
                conv_chunks = [
                    F.gelu(self.conv2d3(F.gelu(self.conv2d2(F.gelu(self.conv2d1(c))))))
                    for c in conv_in.split(getattr(self, "conv_chunksize", 500), dim=0)
                ]
                hs, _ = self.conv_out(
                    torch.cat(conv_chunks, dim=0)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .view(bucket, t_chunk, -1)
                )
                hs = F.pad(
                    (hs + pos_emb).reshape(bucket * t_chunk, -1),
                    (0, 0, 0, seq_pad),
                )
                for layer in self.layers:
                    hs = layer(hs, cu, max_seqlen=max(cu_lens))
                hs = self.proj2(self.act(self.proj1(self.ln_post(hs))[0]))[0]
                outputs.append(hs.cpu()[:alen])
            res = torch.cat(outputs, dim=0)
            return res if dev.type == "tpu" else res.to(dev)

        Qwen3OmniMoeAudioEncoder.forward = _padded_encoder_forward

        # 3. Vision Transformer: chunk long videos at frame boundaries and pad each
        #    chunk to static patch buckets to prevent HBM OOM and recompilation.
        vit_cls = getattr(thinker_mod, "Qwen3Omni_VisionTransformer", None)
        if vit_cls is not None:
            vit_cls.device = property(lambda self: self.pos_embed.weight.device)

            def _padded_vit_forward(
                self: Any, x: torch.Tensor, grid_thw: Any
            ) -> torch.Tensor:
                dev = self.patch_embed.proj.weight.device
                dtype = self.dtype
                if dev.type == "tpu" and self.pos_embed.weight.device.type == "tpu":
                    self.pos_embed.cpu()
                    self.rotary_pos_emb.cpu()
                grid_cpu = (
                    grid_thw.detach().cpu().to(torch.int32)
                    if isinstance(grid_thw, torch.Tensor)
                    else torch.as_tensor(grid_thw, dtype=torch.int32)
                )
                grid_list = [[int(v) for v in row] for row in grid_cpu.tolist()]
                pos_all = (
                    self.fast_pos_embed_interpolate(grid_list).cpu()
                    if self.apply_vit_abs_pos_embed
                    else None
                )
                cos_all, sin_all = (t.cpu() for t in self.rot_pos_emb(grid_cpu))
                x_cpu = x.detach().cpu().to(dtype)

                frame_seqlens = torch.repeat_interleave(
                    grid_cpu[:, 1] * grid_cpu[:, 2], grid_cpu[:, 0]
                ).tolist()
                chunks: list[list[int]] = []
                for seqlen in frame_seqlens:
                    if chunks and sum(chunks[-1]) + seqlen <= _VISION_PATCH_BUCKETS[-1]:
                        chunks[-1].append(seqlen)
                    else:
                        chunks.append([seqlen])

                outputs, offset = [], 0
                ds_indexes = self.deepstack_visual_indexes
                for chunk_lens in chunks:
                    num_p = sum(chunk_lens)
                    bucket_p = _select_bucket(
                        num_p, _VISION_PATCH_BUCKETS, align=512
                    )
                    pad = (0, 0, 0, bucket_p - num_p)
                    cos = F.pad(cos_all[offset : offset + num_p], pad).to(dev)
                    sin = F.pad(sin_all[offset : offset + num_p], pad).to(dev)
                    hs = self.patch_embed(
                        F.pad(x_cpu[offset : offset + num_p], pad).to(dev)
                    )
                    if pos_all is not None:
                        hs = hs + F.pad(
                            pos_all[offset : offset + num_p], pad
                        ).to(device=dev, dtype=dtype)
                    hs = hs.unsqueeze(1)
                    cu = F.pad(
                        torch.tensor(chunk_lens, dtype=torch.int32).cumsum(
                            0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    cu = F.pad(
                        cu, (0, (-cu.numel()) % 16), value=bucket_p + 1
                    ).to(dev)
                    ds_feats = []
                    for idx, blk in enumerate(self.blocks):
                        hs = blk(
                            hs,
                            cu_seqlens=cu,
                            rotary_pos_emb_cos=cos,
                            rotary_pos_emb_sin=sin,
                            max_seqlen=max(chunk_lens),
                            sequence_lengths=None,
                        )
                        if ds_indexes is not None and idx in ds_indexes:
                            ds_feats.append(hs)
                    hs = self.merger(hs)
                    if ds_indexes is not None:
                        hs = torch.cat(
                            [hs]
                            + [
                                self.merger_list[i](feat)
                                for i, feat in enumerate(ds_feats)
                            ],
                            dim=1,
                        )
                    outputs.append(hs.cpu()[: num_p // self.spatial_merge_unit])
                    offset += num_p
                res = torch.cat(outputs, dim=0)
                return res if dev.type == "tpu" else res.to(dev)

            vit_cls.forward = _padded_vit_forward

        # 4. Multimodal Input Preprocessing: keep metadata & features on CPU before
        #    calling the bucketed encoders to avoid XLA dynamic-split syncs.
        mixin_cls = getattr(
            thinker_mod, "Qwen3OmniMoeConditionalGenerationMixin", None
        )
        if mixin_cls is not None:
            orig_aud = mixin_cls._process_audio_input
            mixin_cls._process_audio_input = lambda self, ai: orig_aud(
                self,
                {
                    "input_features": ai["input_features"].detach().cpu(),
                    "audio_feature_lengths": ai["audio_feature_lengths"]
                    .detach()
                    .cpu(),
                },
            )
            if orig_vid := getattr(mixin_cls, "_process_video_input", None):
                mixin_cls._process_video_input = lambda self, vi: orig_vid(
                    self,
                    {
                        "type": vi["type"],
                        "pixel_values_videos": vi["pixel_values_videos"]
                        .detach()
                        .cpu(),
                        "video_grid_thw": vi["video_grid_thw"].detach().cpu(),
                    }
                    if vi["type"] == "pixel_values_videos"
                    else {
                        "type": vi["type"],
                        "video_embeds": vi["video_embeds"].detach().cpu(),
                        "video_grid_thw": vi["video_grid_thw"].detach().cpu(),
                    },
                )
            if orig_img := getattr(mixin_cls, "_process_image_input", None):
                mixin_cls._process_image_input = lambda self, ii: orig_img(
                    self,
                    {
                        "type": ii["type"],
                        "pixel_values": ii["pixel_values"].detach().cpu(),
                        "image_grid_thw": ii["image_grid_thw"].detach().cpu(),
                    }
                    if ii["type"] == "pixel_values"
                    else ii,
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
                        e.detach().cpu() if e.device.type == "tpu" else e
                        for e in multimodal_embeddings
                    ]
                    flat = vllm_utils._flatten_embeddings(cpu_embeds)
                    if 0 < flat.shape[0] < inputs_embeds.shape[0]:
                        pad_rows = inputs_embeds.shape[0] - flat.shape[0]
                        multimodal_embeddings = [
                            F.pad(flat, (0, 0, 0, pad_rows))
                        ]
                if (
                    is_multimodal is not None
                    and is_multimodal.device != inputs_embeds.device
                ):
                    is_multimodal = is_multimodal.to(inputs_embeds.device)
                return base_merge(
                    inputs_embeds, multimodal_embeddings, is_multimodal
                )

            _omni_merge._is_omni_static_padded = True  # type: ignore[attr-defined]
            vllm_utils._merge_multimodal_embeddings = _omni_merge
            for name, mod in list(sys.modules.items()):
                if (
                    mod is not None
                    and name.startswith(
                        (
                            "vllm.model_executor.models",
                            "vllm_omni.model_executor.models",
                        )
                    )
                    and hasattr(mod, "_merge_multimodal_embeddings")
                ):
                    mod._merge_multimodal_embeddings = _omni_merge

        # 6. Interleaved Audio+Video & CPU-Staged embed_input_ids: run non-zero mask
        #    checks and deepstack feature splitting on CPU, preserving `.modality`.
        _active_merge = vllm_utils._merge_multimodal_embeddings
        orig_check = getattr(thinker_mod, "check_interleaved_audio_video", None)
        if orig_check is not None:
            thinker_mod.check_interleaved_audio_video = (
                lambda iv, ia, nv, na: orig_check(
                    iv.detach().cpu(), ia.detach().cpu(), nv, na
                )
                if nv and na
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
                mods = get_mm_embedding_modalities(multimodal_embeddings)
                for mask, mod_name in (
                    (is_video, "video"),
                    (is_audio, "audio"),
                    (is_image, "image"),
                ):
                    group = [
                        e
                        for e, m in zip(multimodal_embeddings, mods)
                        if m == mod_name
                    ]
                    if group:
                        inputs_embeds = _active_merge(
                            inputs_embeds, group, mask
                        )
                return inputs_embeds

            thinker_mod.merge_interleaved_embeddings = _merge_interleaved

        thinker_cls = getattr(
            thinker_mod, "Qwen3OmniMoeThinkerForConditionalGeneration", None
        )
        if thinker_cls is not None and hasattr(thinker_cls, "embed_input_ids"):
            orig_embed = thinker_cls.embed_input_ids

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
                            e.detach().cpu() if e.device.type == "tpu" else e, e
                        )
                        for e in multimodal_embeddings
                    ]
                    if (
                        is_multimodal is not None
                        and is_multimodal.device.type == "tpu"
                    ):
                        is_multimodal = is_multimodal.detach().cpu()
                return orig_embed(
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
                mm_list, is_mm = orig_gather(self, *args, **kwargs)
                self._omni_mm_embeds_list = mm_list
                if mm_list:
                    total_tokens = sum(int(e.shape[0]) for e in mm_list)
                    return [torch.arange(total_tokens, dtype=torch.int64)], is_mm
                return mm_list, is_mm

            def _omni_get_model_inputs(
                self: Any,
                input_ids: torch.Tensor,
                mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
            ) -> Any:
                if mm_embed_inputs is not None:
                    c_embeds, is_mm = mm_embed_inputs
                    mm_list = getattr(self, "_omni_mm_embeds_list", None)
                    if c_embeds and mm_list and c_embeds[0].ndim == 1:
                        mm_start = int(c_embeds[0][0].item())
                        mm_end = mm_start + int(c_embeds[0].shape[0])
                        real_embeds, pos = [], 0
                        for e in mm_list:
                            num_rows = int(e.shape[0])
                            s = max(0, mm_start - pos)
                            t = min(num_rows, mm_end - pos)
                            if s < t:
                                sub = e if (s == 0 and t == num_rows) else e[s:t]
                                real_embeds.append(_with_modality(sub, e))
                            pos += num_rows
                        mm_embed_inputs = (real_embeds, is_mm)
                return orig_get_inputs(self, input_ids, mm_embed_inputs)

            _omni_gather_mm._is_omni_patched = True  # type: ignore[attr-defined]
            TPUModelRunner._gather_mm_embeddings = _omni_gather_mm
            TPUModelRunner._get_model_inputs = _omni_get_model_inputs

        logger.info(
            "Applied TPU patch: Qwen3-Omni Thinker audio/vision encoders and attention."
        )
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Thinker patch: %s", e)
