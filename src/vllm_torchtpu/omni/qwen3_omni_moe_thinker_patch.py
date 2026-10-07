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
    """Patch Qwen3-Omni MoE Thinker audio encoder and attention for TPU."""
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

        mixin_cls = getattr(
            thinker_mod, "Qwen3OmniMoeConditionalGenerationMixin", None
        )
        if mixin_cls is not None:
            orig_proc = mixin_cls._process_audio_input
            mixin_cls._process_audio_input = lambda self, ai: orig_proc(
                self, {k: v.detach().cpu() for k, v in ai.items()}
            )

        _patch_vllm_merge_multimodal_embeddings()
        base_merge = vllm_utils._merge_multimodal_embeddings
        if not getattr(base_merge, "_is_omni_static_padded", False):

            def _omni_merge(
                inputs_embeds: torch.Tensor,
                mm_embeds: Any,
                is_mm: torch.Tensor,
            ) -> torch.Tensor:
                if len(mm_embeds) > 0:
                    mm = vllm_utils._flatten_embeddings(mm_embeds)
                    if 0 < mm.shape[0] < inputs_embeds.shape[0]:
                        pad = inputs_embeds.shape[0] - mm.shape[0]
                        src = (
                            mm.detach().cpu()
                            if inputs_embeds.device.type == "tpu"
                            else mm
                        )
                        mm_embeds = [F.pad(src, (0, 0, 0, pad))]
                return base_merge(inputs_embeds, mm_embeds, is_mm)

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

        logger.info(
            "Applied TPU patch: Qwen3-Omni Thinker audio attention and padded encoder."
        )
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Thinker patch: %s", e)
