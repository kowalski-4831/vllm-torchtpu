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

_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 500)
_VISION_PATCH_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)


def _select_bucket(
    size: int, buckets: tuple[int, ...] = _AUDIO_CHUNK_BUCKETS, align: int = 8
) -> int:
    """Return the smallest bucket >= size, or round up to a multiple of align."""
    return next((b for b in buckets if size <= b), ((max(1, size) + align - 1) // align) * align)


def _with_mod(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if hasattr(src, "modality"):
        dst.modality = src.modality
    return dst


@run_once
def apply_omni_ar_patches() -> None:
    """Patch Qwen3-Omni MoE Thinker audio/vision encoders and attention for TPU."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
        from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
        import vllm.model_executor.models.utils as vllm_utils
        import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as thinker_mod
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeAudioAttention,
            Qwen3OmniMoeAudioEncoder,
        )

        from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        def _init(self: Any, config: Any, quant_config: Any = None, prefix: str = "") -> None:
            nn.Module.__init__(self)
            d, h = config.d_model, config.encoder_attention_heads
            hd, tp = d // h, get_tensor_model_parallel_world_size()
            no_tp = h % tp != 0
            self.embed_dim, self.num_heads, self.head_dim = d, h, hd
            self.num_local_heads, self.scaling = (h if no_tp else h // tp), hd**-0.5
            kw = {"bias": True, "quant_config": quant_config, "disable_tp": no_tp}
            self.qkv = QKVParallelLinear(d, hd, h, h, prefix=f"{prefix}.qkv", **kw)
            self.attn = MMEncoderAttention(self.num_local_heads, hd, self.scaling, prefix=f"{prefix}.attn")
            self.out_proj = RowParallelLinear(d, d, prefix=f"{prefix}.out_proj", **kw)

        Qwen3OmniMoeAudioAttention.__init__ = _init

        def _padded_encoder_forward(
            self: Any, x: torch.Tensor, flens: torch.Tensor, alens: torch.Tensor
        ) -> torch.Tensor:
            win, dev, dt = self.n_window * 2, self.conv2d1.weight.device, self.conv2d1.weight.dtype
            tc = (((win + 1) // 2 + 1) // 2 + 1) // 2
            w_cnn, fls, als = tc * (self.n_window_infer // win), [int(v) for v in flens.tolist()], [int(v) for v in alens.tolist()]
            pos = self.positional_embedding.positional_embedding[:tc].unsqueeze(0).to(dev, dt)
            outs = []
            for feat, fl, al in zip(x.detach().cpu().to(dt).split(fls, dim=1), fls, als):
                b = _select_bucket((fl + win - 1) // win)
                sp, rem = (-(b * tc)) % 128, al % w_cnn
                cu_l = [0] + [w_cnn] * (al // w_cnn) + ([rem] if rem else [])
                cu = torch.tensor(cu_l, dtype=torch.int32).cumsum(-1, dtype=torch.int32)
                cu = F.pad(cu, (0, (-cu.numel()) % 16), value=b * tc + sp + 1).to(dev)
                c_in = F.pad(feat, (0, b * win - fl)).T.reshape(b, win, -1).transpose(1, 2).unsqueeze(1).contiguous().to(dev)
                c_out = [
                    F.gelu(self.conv2d3(F.gelu(self.conv2d2(F.gelu(self.conv2d1(c))))))
                    for c in c_in.split(getattr(self, "conv_chunksize", 500), 0)
                ]
                hs, _ = self.conv_out(torch.cat(c_out, 0).permute(0, 3, 1, 2).contiguous().view(b, tc, -1))
                hs = F.pad((hs + pos).reshape(b * tc, -1), (0, 0, 0, sp))
                for lyr in self.layers:
                    hs = lyr(hs, cu, max_seqlen=max(cu_l))
                outs.append(self.proj2(self.act(self.proj1(self.ln_post(hs))[0]))[0].cpu()[:al])
            res = torch.cat(outs, dim=0)
            return res if dev.type == "tpu" else res.to(dev)

        Qwen3OmniMoeAudioEncoder.forward = _padded_encoder_forward

        if (vit_cls := getattr(thinker_mod, "Qwen3Omni_VisionTransformer", None)) is not None:
            vit_cls.device = property(lambda self: self.pos_embed.weight.device)

            def _padded_vit_forward(self: Any, x: torch.Tensor, grid_thw: Any) -> torch.Tensor:
                dev, dt = self.patch_embed.proj.weight.device, self.dtype
                if dev.type == "tpu" and self.pos_embed.weight.device.type == "tpu":
                    self.pos_embed.cpu(), self.rotary_pos_emb.cpu()
                g = grid_thw.detach().cpu().to(torch.int32) if isinstance(grid_thw, torch.Tensor) else torch.as_tensor(grid_thw, dtype=torch.int32)
                pos = self.fast_pos_embed_interpolate([[int(v) for v in r] for r in g.tolist()]).cpu() if self.apply_vit_abs_pos_embed else None
                cos_a, sin_a = (t.cpu() for t in self.rot_pos_emb(g))
                x_c, chunks = x.detach().cpu().to(dt), []
                for s in torch.repeat_interleave(g[:, 1] * g[:, 2], g[:, 0]).tolist():
                    if chunks and sum(chunks[-1]) + s <= _VISION_PATCH_BUCKETS[-1]:
                        chunks[-1].append(s)
                    else:
                        chunks.append([s])
                outs, off, ds_idx = [], 0, self.deepstack_visual_indexes
                for c_l in chunks:
                    np_, bp = sum(c_l), _select_bucket(sum(c_l), _VISION_PATCH_BUCKETS, 512)
                    pad = (0, 0, 0, bp - np_)
                    cos, sin = F.pad(cos_a[off : off + np_], pad).to(dev), F.pad(sin_a[off : off + np_], pad).to(dev)
                    hs = self.patch_embed(F.pad(x_c[off : off + np_], pad).to(dev))
                    if pos is not None:
                        hs = hs + F.pad(pos[off : off + np_], pad).to(dev, dt)
                    hs, cu = hs.unsqueeze(1), F.pad(torch.tensor(c_l, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
                    cu, ds = F.pad(cu, (0, (-cu.numel()) % 16), value=bp + 1).to(dev), []
                    for i, blk in enumerate(self.blocks):
                        hs = blk(hs, cu_seqlens=cu, rotary_pos_emb_cos=cos, rotary_pos_emb_sin=sin, max_seqlen=max(c_l), sequence_lengths=None)
                        if ds_idx is not None and i in ds_idx:
                            ds.append(hs)
                    hs = self.merger(hs)
                    if ds_idx is not None:
                        hs = torch.cat([hs] + [self.merger_list[i](d) for i, d in enumerate(ds)], dim=1)
                    outs.append(hs.cpu()[: np_ // self.spatial_merge_unit])
                    off += np_
                res = torch.cat(outs, dim=0)
                return res if dev.type == "tpu" else res.to(dev)

            vit_cls.forward = _padded_vit_forward

        if (mixin_cls := getattr(thinker_mod, "Qwen3OmniMoeConditionalGenerationMixin", None)) is not None:
            orig_aud = mixin_cls._process_audio_input
            mixin_cls._process_audio_input = lambda self, ai: orig_aud(
                self, {k: ai[k].detach().cpu() for k in ("input_features", "audio_feature_lengths")}
            )

        _patch_vllm_merge_multimodal_embeddings()
        base_merge = vllm_utils._merge_multimodal_embeddings
        if not getattr(base_merge, "_is_omni_static_padded", False):

            def _omni_merge(ie: torch.Tensor, mm: Any, is_mm: Any = None) -> torch.Tensor:
                if len(mm) > 0:
                    flat = vllm_utils._flatten_embeddings([e.detach().cpu() if e.device.type == "tpu" else e for e in mm])
                    if 0 < flat.shape[0] < ie.shape[0]:
                        mm = [F.pad(flat, (0, 0, 0, ie.shape[0] - flat.shape[0]))]
                if is_mm is not None and is_mm.device != ie.device:
                    is_mm = is_mm.to(ie.device)
                return base_merge(ie, mm, is_mm)

            _omni_merge._is_omni_static_padded = True  # type: ignore[attr-defined]
            vllm_utils._merge_multimodal_embeddings = _omni_merge
            for name, mod in list(sys.modules.items()):
                if mod is not None and name.startswith(("vllm.model_executor.models", "vllm_omni.model_executor.models")) and hasattr(mod, "_merge_multimodal_embeddings"):
                    mod._merge_multimodal_embeddings = _omni_merge

        _active_merge = vllm_utils._merge_multimodal_embeddings
        if (orig_chk := getattr(thinker_mod, "check_interleaved_audio_video", None)) is not None:
            thinker_mod.check_interleaved_audio_video = (
                lambda iv, ia, nv, na: orig_chk(iv.detach().cpu(), ia.detach().cpu(), nv, na) if nv and na else False
            )

        if hasattr(thinker_mod, "merge_interleaved_embeddings"):

            def _merge_interleaved(ie: torch.Tensor, mm: Any, iv: torch.Tensor, ia: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
                from vllm.multimodal.utils import get_mm_embedding_modalities

                mods = get_mm_embedding_modalities(mm)
                for msk, name in ((iv, "video"), (ia, "audio"), (im & ~iv & ~ia, "image")):
                    if grp := [e for e, m in zip(mm, mods) if m == name]:
                        ie = _active_merge(ie, grp, msk)
                return ie

            thinker_mod.merge_interleaved_embeddings = _merge_interleaved

        if (tc := getattr(thinker_mod, "Qwen3OmniMoeThinkerForConditionalGeneration", None)) and hasattr(tc, "embed_input_ids"):
            orig_embed = tc.embed_input_ids

            def _cpu_mm_embed(self: Any, ids: torch.Tensor, mm: Any = None, *, is_multimodal: Any = None) -> torch.Tensor:
                if mm:
                    mm = [_with_mod(e.detach().cpu() if e.device.type == "tpu" else e, e) for e in mm]
                    is_multimodal = is_multimodal.detach().cpu() if is_multimodal is not None and is_multimodal.device.type == "tpu" else is_multimodal
                return orig_embed(self, ids, multimodal_embeddings=mm, is_multimodal=is_multimodal)

            tc.embed_input_ids = _cpu_mm_embed

        if not getattr(TPUModelRunner._gather_mm_embeddings, "_is_omni_patched", False):
            orig_gather, orig_get = TPUModelRunner._gather_mm_embeddings, TPUModelRunner._get_model_inputs

            def _omni_gather_mm(self: Any, *a: Any, **kw: Any) -> Any:
                ml, ism = orig_gather(self, *a, **kw)
                self._omni_mm_embeds_list = ml
                return ([torch.arange(sum(int(e.shape[0]) for e in ml), dtype=torch.int64)] if ml else ml), ism

            def _omni_get_model_inputs(self: Any, ids: torch.Tensor, mmi: Any) -> Any:
                if mmi and mmi[0] and (ml := getattr(self, "_omni_mm_embeds_list", None)) and mmi[0][0].ndim == 1:
                    st, en, pos, real = int(mmi[0][0][0].item()), int(mmi[0][0][0].item()) + int(mmi[0][0].shape[0]), 0, []
                    for e in ml:
                        s, t = max(0, st - pos), min(n := int(e.shape[0]), en - pos)
                        if s < t:
                            real.append(_with_mod(e if (s == 0 and t == n) else e[s:t], e))
                        pos += n
                    mmi = (real, mmi[1])
                return orig_get(self, ids, mmi)

            _omni_gather_mm._is_omni_patched = True  # type: ignore[attr-defined]
            TPUModelRunner._gather_mm_embeddings, TPUModelRunner._get_model_inputs = _omni_gather_mm, _omni_get_model_inputs

        logger.info("Applied TPU patch: Qwen3-Omni Thinker audio/vision encoders and attention.")
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Thinker patch: %s", e)
