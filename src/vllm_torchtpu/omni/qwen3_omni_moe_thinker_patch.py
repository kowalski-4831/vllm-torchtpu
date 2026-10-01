# SPDX-License-Identifier: Apache-2.0
"""TPU runtime patches for vLLM-Omni autoregressive (AR) models."""

from functools import cache as run_once
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def apply_omni_ar_patches() -> None:
    """Patch Qwen3-Omni MoE Thinker audio attention for TPU."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.linear import (
            QKVParallelLinear,
            RowParallelLinear,
        )
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeAudioAttention,
        )

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
            self.out_proj = RowParallelLinear(
                self.embed_dim,
                self.embed_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
                disable_tp=no_tp,
            )

        def _forward(
            self: Any,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_seqlen: Any = None,
        ) -> torch.Tensor:
            seq_len, _ = hidden_states.size()
            q, k, v = self.qkv(hidden_states)[0].chunk(3, dim=-1)
            q = q.view(1, seq_len, self.num_local_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, seq_len, self.num_local_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, seq_len, self.num_local_heads, self.head_dim).transpose(1, 2)
            mask = None
            if cu_seqlens is not None and cu_seqlens.numel() > 2:
                pos = torch.arange(seq_len, device=hidden_states.device)
                seg = (pos.unsqueeze(1) >= cu_seqlens.to(pos.dtype).unsqueeze(0)).sum(1)
                mask = (seg.unsqueeze(0) == seg.unsqueeze(1))[None, None]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.scaling
            )
            return self.out_proj(out.transpose(1, 2).reshape(seq_len, -1))[0]

        Qwen3OmniMoeAudioAttention.__init__ = _init
        Qwen3OmniMoeAudioAttention.forward = _forward
        logger.info("Applied TPU patch: Qwen3-Omni Thinker audio attention.")
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Thinker patch: %s", e)
