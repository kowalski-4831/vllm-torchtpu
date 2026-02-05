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
TPU-native unquantized (bfloat16/float16) support for vLLM.

This module provides TPU-compatible implementations for unquantized models
like Qwen3-Coder. It bypasses vLLM's default `torch_xla`-based TPU path
and uses our native TorchTPU + Pallas kernels.

== Supported Models ==
- Qwen3-Coder-30B-A3B (unquantized MoE)
- Any MoE model without quantization

== Key Classes ==
- VllmUnquantizedConfig: Quantization config for unquantized models
- VllmUnquantizedFusedMoEMethod: MoE forward pass using GMM kernel
- VllmUnquantizedLinearMethod: Linear layer using standard matmul

== Execution Flow ==
1. vLLM loads model with quantization=None (unquantized)
2. get_tpu_quantization_config() returns VllmUnquantizedConfig
3. For FusedMoE layers, VllmUnquantizedFusedMoEMethod.apply() is called
4. apply() routes to fused_moe_gmm() which uses our TPU Pallas kernels
"""

from typing import Any, Optional

import torch
from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from tpu_inference.layers.common.quant_methods import (UNQUANTIZED,
                                                       get_tpu_quant_method)
from tpu_inference.layers.vllm.fused_moe_torch import fused_moe_gmm
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(UNQUANTIZED))
class VllmUnquantizedConfig(QuantizationConfig, VllmQuantConfig):
    """
    TPU-specific configuration for unquantized models.

    This config is registered under "tpu-unquantized" and provides
    TPU-compatible methods for FusedMoE and Linear layers.
    """

    @classmethod
    def get_name(cls) -> str:
        return UNQUANTIZED

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # Always supported on TPU

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # No extra configs required

    @classmethod
    def from_config(cls, _: dict[str, Any]) -> "VllmUnquantizedConfig":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        """
        Return the appropriate quantization method for a layer.

        Args:
            layer: The layer to handle.
            prefix: The parameter prefix (e.g., "model.layers.0.mlp").

        Returns:
            A QuantizeMethodBase implementation, or None to use defaults.
        """
        if isinstance(layer, LinearBase):
            # Return None to let vLLM use its default UnquantizedLinearMethod
            # which uses torch.matmul - works fine on TPU
            # TODO: Do we need to use our own linear method?
            logger.warning_once(
                "Using vLLM's unquantized linear method for layer, performance may be affected."
            )
            return UnquantizedLinearMethod()

        if isinstance(layer, FusedMoE):
            # For MoE layers, use our custom TPU implementation
            moe_config = self.get_moe_config(layer)
            return VllmUnquantizedFusedMoEMethod(moe_config)

        if isinstance(layer, Attention):
            return None
        return None


class VllmUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """
    TPU-native implementation of unquantized FusedMoE.

    This class overrides the default forward_tpu() behavior which uses
    torch_xla, and instead uses our native TorchTPU + Pallas GMM kernels.

    Key Design:
    - Inherits from UnquantizedFusedMoEMethod for weight creation/processing
    - Overrides apply() to use fused_moe_gmm() instead of fused_moe_pallas()
    """

    def __init__(self, moe: FusedMoEConfig):
        """
        Initialize the unquantized MoE method for TPU.

        Args:
            moe: The MoE configuration from vLLM.
        """
        super().__init__(moe)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Process weights after loading for GMM kernel compatibility.

        Transforms weights from vLLM's default layout to GMM kernel layout:
        - Transpose: [E, out_dim, in_dim] → [E, in_dim, out_dim]
        - Handle interleaving for swigluoai activation
        - Transform biases: [E, out_dim] → [E, 1, out_dim]
        TODO: This is similiar to process_weights_after_loading in mxfp4.py, consider refactoring
        """
        assert isinstance(layer, FusedMoE)
        # Get weight tensors
        w13_weight = layer.w13_weight.data
        w2_weight = layer.w2_weight.data

        # Handle w13 interleaving for swigluoai activation
        # Some models store w13 interleaved: even indices are w1 (gate), odd are w3 (up)
        # We un-interleave so first half is w1 and second half is w3
        w13_interleave = layer.activation == "swigluoai"
        if w13_interleave:
            # w13_weight shape: [E, out_dim, in_dim] where out_dim = 2 * intermediate
            w1_weight = w13_weight[:, ::2, :]  # even indices
            w3_weight = w13_weight[:, 1::2, :]  # odd indices
            w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

        # Transpose: [E, out_dim, in_dim] → [E, in_dim, out_dim]
        # GMM kernel expects contracting dim on axis 1
        w13_weight = w13_weight.transpose(1, 2)  # [E, hidden, 2*intermediate]
        w2_weight = w2_weight.transpose(1, 2)  # [E, intermediate, hidden]

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        # Handle biases if present
        if self.moe.has_bias:
            w13_bias = layer.w13_bias.data
            w2_bias = layer.w2_bias.data

            # Handle w13 bias interleaving
            if w13_interleave:
                w1_bias = w13_bias[:, ::2]
                w3_bias = w13_bias[:, 1::2]
                w13_bias = torch.cat([w1_bias, w3_bias], dim=1)

            # Transform: [E, out_dim] → [E, 1, out_dim]
            layer.w13_bias = torch.nn.Parameter(
                w13_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
            layer.w2_bias = torch.nn.Parameter(
                w2_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )

        logger.info_once(
            "Unquantized weights transposed for GMM kernel: "
            f"w13={list(layer.w13_weight.shape)}, w2={list(layer.w2_weight.shape)}"
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using TPU-native GMM kernel.

        This bypasses vLLM's forward_tpu() which uses torch_xla and instead
        uses our fused_moe_gmm() which uses TorchTPU + Pallas kernels.

        Args:
            layer: The FusedMoE layer.
            x: Input tensor [num_tokens, hidden_size].
            router_logits: Router logits [num_tokens, num_experts].

        Returns:
            Output tensor [num_tokens, hidden_size].
        """
        output = fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=None,  # Unquantized = no scales
            w2_scale=None,
            w1_bias=getattr(layer, 'w13_bias', None),
            w2_bias=getattr(layer, 'w2_bias', None),
            gating_output=router_logits,
            topk=layer.moe_config.experts_per_token,
            renormalize=layer.renormalize,
            activation=layer.activation,
        )
        return output
