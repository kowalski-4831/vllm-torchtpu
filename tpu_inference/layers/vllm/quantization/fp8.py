# Copyright 2025 Google LLC
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
"""
FP8 quantization support for TPU.

This module provides TPU-compatible FP8 quantization for MoE models like
Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8. The target model uses FP8 e4m3fn
weights with block-wise scales (weight_block_size: [128, 128]).

== MoE Weights ==

Kept in FP8 and passed with block-wise scales to the GMM kernel, which
natively supports FP8 rhs with rhs_scale via gmm_v2.

== Linear Weights (attention projections) ==

Dequantized from FP8 to BF16 at load time since TPU doesn't have FP8
linear kernels. The vLLM Fp8LinearMethod.create_weights() is reused for
correct parameter allocation (weight_scale_inv), then process_weights
dequantizes using block scales.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         Fp8LinearMethod,
                                                         Fp8MoEMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped
from vllm.model_executor.utils import replace_parameter

from tpu_inference.layers.common.quant_methods import FP8, get_tpu_quant_method
from tpu_inference.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 prebuild_fused_moe_kernel)
from tpu_inference.layers.vllm.moe_routing import (select_experts,
                                                   select_experts_ep)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(FP8))
class VllmFp8Config(Fp8Config, VllmQuantConfig):

    @classmethod
    def get_name(cls) -> str:
        return FP8

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                from tpu_inference.layers.vllm.quantization.unquantized import \
                    VllmUnquantizedFusedMoEMethod
                moe_config = self.get_moe_config(layer)
                return VllmUnquantizedFusedMoEMethod(moe_config)
            if self.is_checkpoint_fp8_serialized:
                moe_config = self.get_moe_config(layer)
                return VllmFp8MoEMethodTPU(self, moe_config)
            raise NotImplementedError(
                "Online FP8 quantization not supported on TPU.")

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if self.is_checkpoint_fp8_serialized:
                return VllmFp8LinearMethodTPU(self)
            raise NotImplementedError(
                "Online FP8 quantization not supported on TPU.")

        if isinstance(layer, Attention):
            return None

        return None


class VllmFp8MoEMethodTPU(FusedMoEMethodBase):
    """
    TPU-native FP8 MoE method for block-quantized FP8 models.

    Keeps weights in FP8 e4m3fn and passes block-wise scales to the GMM
    kernel. Delegates create_weights() to Fp8MoEMethod for correct FP8
    weight and scale allocation, but overrides processing and forward
    to use TPU-native fused_moe_gmm().

    Uses is_monolithic=True so vLLM's DefaultMoERunner calls
    apply_monolithic(layer, x, router_logits) directly, bypassing the
    CUDA-only router.select_experts() path.
    """

    def __init__(self, quant_config: Fp8Config, moe_config):
        super().__init__(moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")
        self._is_monolithic = True
        self.apply_monolithic = self._forward_monolithic_tpu

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        Fp8MoEMethod.create_weights(self, layer, num_experts, hidden_size,
                                    intermediate_size_per_partition,
                                    params_dtype, **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Process FP8 block-quantized weights for GMM kernel.

        For block quant [128, 128]:
        - w13_weight_scale_inv: [E, 2*I/128, H/128]
        - w2_weight_scale_inv: [E, H/128, I/128]

        After processing:
        - Weights transposed: [E, out, in] -> [E, in, out]
        - Scales: [E, out/B, in/B] -> [E, in/B, 1, out] (expanded)
        """
        assert isinstance(layer, FusedMoE)

        activation_str = (layer.activation.value if hasattr(
            layer.activation, 'value') else str(layer.activation))
        layer._tpu_activation_str = activation_str

        w13_weight = layer.w13_weight.data
        w2_weight = layer.w2_weight.data

        # Get scales (named weight_scale_inv for block quant)
        w13_weight_scale = getattr(layer, f"w13_{self.weight_scale_name}").data
        w2_weight_scale = getattr(layer, f"w2_{self.weight_scale_name}").data

        # 1. Handle w13 interleaving for swigluoai activation
        w13_interleave = activation_str == "swigluoai"
        if w13_interleave:
            w1_weight = w13_weight[:, ::2, :]
            w3_weight = w13_weight[:, 1::2, :]
            w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

            w1_scale = w13_weight_scale[:, ::2, :]
            w3_scale = w13_weight_scale[:, 1::2, :]
            w13_weight_scale = torch.cat([w1_scale, w3_scale], dim=1)

        # 2. Transpose weights: [E, out, in] -> [E, in, out]
        w13_weight = w13_weight.transpose(1, 2).contiguous()
        w2_weight = w2_weight.transpose(1, 2).contiguous()

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        # 3. Reshape scales for GMM kernel
        # GMM expects rhs_scale shape [E, in_blocks, 1, out_dim]
        # Input: [E, out/block_h, in/block_w]
        block_h = self.weight_block_size[0] if self.block_quant else 1
        w13_weight_scale = w13_weight_scale.transpose(1, 2)
        w13_weight_scale = w13_weight_scale.unsqueeze(-1).expand(
            *w13_weight_scale.shape,
            block_h).reshape(w13_weight_scale.shape[0],
                             w13_weight_scale.shape[1], -1)
        w13_weight_scale = w13_weight_scale.unsqueeze(2).to(torch.float32)

        w2_weight_scale = w2_weight_scale.transpose(1, 2)
        w2_weight_scale = w2_weight_scale.unsqueeze(-1).expand(
            *w2_weight_scale.shape,
            block_h).reshape(w2_weight_scale.shape[0],
                             w2_weight_scale.shape[1], -1)
        w2_weight_scale = w2_weight_scale.unsqueeze(2).to(torch.float32)

        setattr(layer, f"w13_{self.weight_scale_name}",
                torch.nn.Parameter(w13_weight_scale, requires_grad=False))
        setattr(layer, f"w2_{self.weight_scale_name}",
                torch.nn.Parameter(w2_weight_scale, requires_grad=False))

        # 4. Handle biases if present
        if self.moe.has_bias:
            w13_bias = layer.w13_bias.data
            w2_bias = layer.w2_bias.data

            if w13_interleave:
                w1_bias = w13_bias[:, ::2]
                w3_bias = w13_bias[:, 1::2]
                w13_bias = torch.cat([w1_bias, w3_bias], dim=1)

            layer.w13_bias = torch.nn.Parameter(
                w13_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
            layer.w2_bias = torch.nn.Parameter(
                w2_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )

        logger.info_once(
            "FP8 weights transposed for GMM kernel: "
            f"w13={list(layer.w13_weight.shape)}, "
            f"w2={list(layer.w2_weight.shape)}, "
            f"w13_scale={list(getattr(layer, f'w13_{self.weight_scale_name}').shape)}, "
            f"w2_scale={list(getattr(layer, f'w2_{self.weight_scale_name}').shape)}"
        )
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
        )

    def _forward_monolithic_tpu(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using TPU-native GMM kernel with FP8 weights."""
        activation_str = layer._tpu_activation_str
        if layer.moe_config.moe_parallel_config.use_ep:
            if layer.expert_map is None:
                raise ValueError("EP path requires layer.expert_map.")
            topk_weights, topk_ids = select_experts_ep(
                hidden_states=x,
                router_logits=router_logits,
                expert_map=layer.expert_map,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
                scoring_fn=getattr(layer, "scoring_func", "softmax"),
            )
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
                scoring_fn=getattr(layer, "scoring_func", "softmax"),
            )
        return fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=getattr(layer, f"w13_{self.weight_scale_name}"),
            w2_scale=getattr(layer, f"w2_{self.weight_scale_name}"),
            w1_bias=getattr(layer, 'w13_bias', None),
            w2_bias=getattr(layer, 'w2_bias', None),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
        )


class VllmFp8LinearMethodTPU(Fp8LinearMethod):
    """
    TPU FP8 linear method that dequantizes FP8 weights to BF16 at load time.

    Reuses vLLM's Fp8LinearMethod.create_weights() for correct FP8 weight and
    scale parameter allocation (needed by the weight loader), but dequantizes
    to BF16 after loading since TPU doesn't have FP8 linear kernels.

    Inherits from Fp8LinearMethod but skips GPU-specific __init__. Only
    sets attributes needed by create_weights().
    """

    def __init__(self, quant_config: Fp8Config):
        # Skip Fp8LinearMethod.__init__ which has GPU-specific code
        # (CUDA capability, Marlin, cutlass, W8A8BlockFp8LinearOp).
        # Set only the attributes that create_weights() reads.
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = quant_config.activation_scheme == "static"
        # These are needed by create_weights but are GPU-specific.
        # Set safe defaults so the parent's create_weights doesn't crash.
        self.use_marlin = False
        self.cutlass_block_fp8_supported = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Dequantize FP8 block-quantized weights to BF16."""
        weight = layer.weight.data  # float8_e4m3fn [out, in]
        weight_bf16 = weight.to(torch.bfloat16)

        if self.block_quant:
            weight_scale_inv = layer.weight_scale_inv.data  # [out/B, in/B]
            block_h, block_w = self.weight_block_size

            out_dim, in_dim = weight_bf16.shape
            weight_bf16 = weight_bf16.reshape(out_dim // block_h, block_h,
                                              in_dim // block_w, block_w)
            # scale_inv is [out/B, in/B] -> broadcast over block dims
            weight_bf16 = weight_bf16 * weight_scale_inv.to(
                torch.bfloat16).unsqueeze(1).unsqueeze(3)
            weight_bf16 = weight_bf16.reshape(out_dim, in_dim)
        else:
            weight_scale = layer.weight_scale.data
            weight_bf16 = weight_bf16 * weight_scale.to(torch.bfloat16)

        replace_parameter(layer, "weight",
                          torch.nn.Parameter(weight_bf16, requires_grad=False))

        logger.info_once("FP8 linear weights dequantized to BF16: "
                         f"shape={list(weight_bf16.shape)}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
