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

The checkpoint stores FP8 weights with 2D block scales. At load time we
dequantize those weights, then requantize them into the runtime format used by
the TPU GMM kernel. By default the runtime rhs uses per-channel scales over the
contracting dimension.

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

from tpu_inference.layers.common.process_weights.linear_weights import \
    process_blockwise_fp8_linear_weights
from tpu_inference.layers.common.process_weights.moe_weights import (
    log_and_prebuild_fp8_moe, materialize_moe_weights, process_fp8_moe_weights)
from tpu_inference.layers.common.quant_methods import FP8, get_tpu_quant_method
from tpu_inference.layers.vllm.fused_moe import fused_moe_gmm
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

    Delegates create_weights() to Fp8MoEMethod for correct FP8 parameter
    allocation, then converts checkpoint MoE weights into the runtime FP8 rhs
    format expected by the TPU GMM kernel.

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
        Process FP8 MoE weights into the runtime layout used by the GMM kernel.

        The checkpoint stores FP8 weights with 2D block scales. We first
        dequantize those weights, then requantize them for the GMM runtime
        format. By default the requantized rhs uses a single scale over the
        contracting dimension.
        """
        assert isinstance(layer, FusedMoE)
        assert self.block_quant, "TPU FP8 MoE path expects block-quantized weights."

        activation_str = (layer.activation.value if hasattr(
            layer.activation, 'value') else str(layer.activation))
        layer._tpu_activation_str = activation_str

        weights, requant_dtype_name, requant_block_size = process_fp8_moe_weights(
            layer,
            weight_scale_name=self.weight_scale_name,
            checkpoint_block_size=tuple(self.weight_block_size),
            activation=activation_str,
        )

        layer.w13_weight = torch.nn.Parameter(weights.w13_weight,
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(weights.w2_weight,
                                             requires_grad=False)
        setattr(
            layer, f"w13_{self.weight_scale_name}",
            torch.nn.Parameter(weights.w13_weight_scale, requires_grad=False))
        setattr(
            layer, f"w2_{self.weight_scale_name}",
            torch.nn.Parameter(weights.w2_weight_scale, requires_grad=False))

        if self.moe.has_bias:
            layer.w13_bias = torch.nn.Parameter(weights.w13_bias,
                                                requires_grad=False)
            layer.w2_bias = torch.nn.Parameter(weights.w2_bias,
                                               requires_grad=False)

        materialize_moe_weights(layer, self.weight_scale_name)
        log_and_prebuild_fp8_moe(
            layer,
            weight_scale_name=self.weight_scale_name,
            requant_dtype_name=requant_dtype_name,
            requant_block_size=requant_block_size,
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
        weights = process_blockwise_fp8_linear_weights(
            layer.weight.data,
            weight_scale=getattr(layer, "weight_scale", None).data if hasattr(
                layer, "weight_scale") else None,
            weight_scale_inv=getattr(layer, "weight_scale_inv", None).data
            if hasattr(layer, "weight_scale_inv") else None,
            block_quant=self.block_quant,
            weight_block_size=tuple(self.weight_block_size)
            if self.weight_block_size is not None else None,
            bias=layer.bias.data
            if getattr(layer, "bias", None) is not None else None,
        )

        replace_parameter(
            layer, "weight",
            torch.nn.Parameter(weights.weight, requires_grad=False))

        logger.info_once("FP8 linear weights dequantized to BF16: "
                         f"shape={list(weights.weight.shape)}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
