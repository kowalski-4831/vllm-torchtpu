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

# Bucket sizes in 100-frame (1.0s) chunks, aligned to 8 chunks (8.0s = 104 tokens).
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 500)
_VISION_PATCH_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)


def _select_bucket(
    size: int, buckets: tuple[int, ...] = _AUDIO_CHUNK_BUCKETS, align: int = 8
) -> int:
    """Return the smallest bucket >= size, or round up to a multiple of align."""
    return next(
        (b for b in buckets if size <= b),
        ((max(1, size) + align - 1) // align) * align,
    )


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
        import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as thinker_mod
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeAudioAttention,
            Qwen3OmniMoeAudioEncoder,
        )

        from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings

        def _init(
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
            self.qkv = QKVParallelLinear(
                self.embed_dim,
                self.head_dim,
                self.num_heads,
                self.num_heads,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv",
                disable_tp=no_tp,
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
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
                disable_tp=no_tp,
            )

        Qwen3OmniMoeAudioAttention.__init__ = _init

        def _padded_encoder_forward(
            self: Any,
            input_features: torch.Tensor,
            feature_lens: torch.Tensor,
            aftercnn_lens: torch.Tensor,
        ) -> torch.Tensor:
            win = self.n_window * 2
            t_chunk = (((win + 1) // 2 + 1) // 2 + 1) // 2
            win_cnn = t_chunk * (self.n_window_infer // win)
            dev, dtype = self.conv2d1.weight.device, self.conv2d1.weight.dtype
            flens = [int(x) for x in feature_lens.tolist()]
            alens = [int(x) for x in aftercnn_lens.tolist()]
            feats = input_features.detach().cpu().to(dtype).split(flens, dim=1)
            pos_emb = (
                self.positional_embedding.positional_embedding[:t_chunk, :]
                .unsqueeze(0)
                .to(device=dev, dtype=dtype)
            )
            outs = []
            for feat, flen, alen in zip(feats, flens, alens):
                b = _select_bucket((flen + win - 1) // win)
                seq_pad = (-(b * t_chunk)) % 128
                rem = alen % win_cnn
                cu_lens = (
                    [0]
                    + [win_cnn] * (alen // win_cnn)
                    + ([rem] if rem else [])
                )
                cu = torch.tensor(cu_lens, dtype=torch.int32).cumsum(
                    -1, dtype=torch.int32
                )
                cu = F.pad(
                    cu, (0, (-cu.numel()) % 16), value=b * t_chunk + seq_pad + 1
                ).to(dev)
                x = (
                    F.pad(feat, (0, b * win - flen))
                    .T.reshape(b, win, -1)
                    .transpose(1, 2)
                    .unsqueeze(1)
                    .contiguous()
                    .to(dev)
                )
                x = torch.cat(
                    [
                        F.gelu(
                            self.conv2d3(
                                F.gelu(self.conv2d2(F.gelu(self.conv2d1(c))))
                            )
                        )
                        for c in x.split(
                            getattr(self, "conv_chunksize", 500), dim=0
                        )
                    ],
                    dim=0,
                )
                x, _ = self.conv_out(
                    x.permute(0, 3, 1, 2).contiguous().view(b, t_chunk, -1)
                )
                hs = F.pad(
                    (x + pos_emb).reshape(b * t_chunk, -1), (0, 0, 0, seq_pad)
                )
                for layer in self.layers:
                    hs = layer(hs, cu, max_seqlen=max(cu_lens))
                hs = self.proj2(self.act(self.proj1(self.ln_post(hs))[0]))[0]
                outs.append(hs.cpu()[:alen])
            res = torch.cat(outs, dim=0)
            return res if dev.type == "tpu" else res.to(dev)

        Qwen3OmniMoeAudioEncoder.forward = _padded_encoder_forward

        vit_cls = getattr(thinker_mod, "Qwen3Omni_VisionTransformer", None)
        if vit_cls is not None:
            vit_cls.device = property(lambda self: self.pos_embed.weight.device)

            def _padded_vit_forward(
                self: Any, x: torch.Tensor, grid_thw: Any
            ) -> torch.Tensor:
                dev, dtype = self.patch_embed.proj.weight.device, self.dtype
                if (
                    dev.type == "tpu"
                    and self.pos_embed.weight.device.type == "tpu"
                ):
                    self.pos_embed.cpu()
                    self.rotary_pos_emb.cpu()
                g_cpu = (
                    grid_thw.detach().cpu().to(torch.int32)
                    if isinstance(grid_thw, torch.Tensor)
                    else torch.as_tensor(grid_thw, dtype=torch.int32)
                )
                g_list = [[int(v) for v in r] for r in g_cpu.tolist()]
                n_p = int(x.shape[0])
                b_p = _select_bucket(n_p, _VISION_PATCH_BUCKETS, align=512)
                pad_p = b_p - n_p
                pos = F.pad(
                    self.fast_pos_embed_interpolate(g_list).cpu(),
                    (0, 0, 0, pad_p),
                ).to(device=dev, dtype=dtype)
                cos, sin = self.rot_pos_emb(g_cpu)
                cos = F.pad(cos.cpu(), (0, 0, 0, pad_p)).to(dev)
                sin = F.pad(sin.cpu(), (0, 0, 0, pad_p)).to(dev)
                hs = self.patch_embed(
                    F.pad(x.detach().cpu().to(dtype), (0, 0, 0, pad_p)).to(dev)
                )
                if self.apply_vit_abs_pos_embed:
                    hs = hs + pos
                hs = hs.unsqueeze(1)
                seqlens = torch.repeat_interleave(
                    g_cpu[:, 1] * g_cpu[:, 2], g_cpu[:, 0]
                )
                max_s = int(seqlens.max().item())
                cu = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
                cu = F.pad(cu, (0, (-cu.numel()) % 16), value=b_p + 1).to(dev)
                ds_idx, hs_list = self.deepstack_visual_indexes, []
                for i, blk in enumerate(self.blocks):
                    hs = blk(
                        hs,
                        cu_seqlens=cu,
                        rotary_pos_emb_cos=cos,
                        rotary_pos_emb_sin=sin,
                        max_seqlen=max_s,
                        sequence_lengths=None,
                    )
                    if ds_idx is not None and i in ds_idx:
                        hs_list.append(hs)
                hs = self.merger(hs)
                if ds_idx is not None:
                    hs = torch.cat(
                        [hs]
                        + [
                            self.merger_list[i](d)
                            for i, d in enumerate(hs_list)
                        ],
                        dim=1,
                    )
                out = hs.cpu()[: n_p // self.spatial_merge_unit]
                return out if dev.type == "tpu" else out.to(dev)

            vit_cls.forward = _padded_vit_forward

        mixin_cls = getattr(
            thinker_mod, "Qwen3OmniMoeConditionalGenerationMixin", None
        )
        if mixin_cls is not None:
            orig_proc = mixin_cls._process_audio_input
            mixin_cls._process_audio_input = lambda self, ai: orig_proc(
                self,
                {
                    "input_features": ai["input_features"].detach().cpu(),
                    "audio_feature_lengths": ai["audio_feature_lengths"]
                    .detach()
                    .cpu(),
                },
            )
            if hasattr(mixin_cls, "_process_video_input"):
                orig_vid = mixin_cls._process_video_input
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
                    else vi,
                )
            if hasattr(mixin_cls, "_process_image_input"):
                orig_img = mixin_cls._process_image_input
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

        _patch_vllm_merge_multimodal_embeddings()
        base_merge = vllm_utils._merge_multimodal_embeddings
        if not getattr(base_merge, "_is_omni_static_padded", False):

            def _omni_merge(
                inputs_embeds: torch.Tensor,
                multimodal_embeddings: Any,
                is_multimodal: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if len(multimodal_embeddings) > 0:
                    if inputs_embeds.device.type == "tpu":
                        multimodal_embeddings = [
                            e.detach().cpu() if e.device.type == "tpu" else e
                            for e in multimodal_embeddings
                        ]
                    mm = vllm_utils._flatten_embeddings(multimodal_embeddings)
                    if 0 < mm.shape[0] < inputs_embeds.shape[0]:
                        pad = inputs_embeds.shape[0] - mm.shape[0]
                        src = (
                            mm.detach().cpu()
                            if inputs_embeds.device.type == "tpu"
                            else mm
                        )
                        multimodal_embeddings = [F.pad(src, (0, 0, 0, pad))]
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
                    grp = [
                        e
                        for e, m in zip(multimodal_embeddings, mods)
                        if m == mod_name
                    ]
                    if grp:
                        inputs_embeds = _active_merge(inputs_embeds, grp, mask)
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
                    mm_cpu = []
                    for e in multimodal_embeddings:
                        ec = e.detach().cpu() if e.device.type == "tpu" else e
                        if hasattr(e, "modality"):
                            ec.modality = e.modality
                        mm_cpu.append(ec)
                    multimodal_embeddings = mm_cpu
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

        logger.info(
            "Applied TPU patch: Qwen3-Omni Thinker audio/vision encoders and attention."
        )
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Thinker patch: %s", e)
