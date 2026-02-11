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
MXFP4 Quantization Support for TPU.

This module provides TPU-compatible MXFP4 quantization for MoE models like
openai/gpt-oss-20b. MXFP4 is a 4-bit floating point format with block-wise
scaling.

== Execution Flow ==

1. MODEL LOADING:
   - vLLM detects quantization="mxfp4" in model config
   - get_tpu_quantization_config() returns VllmMxfp4Config
   - VllmMxfp4Config is registered with vLLM's quantization registry

2. LAYER CREATION:
   - For each FusedMoE layer, vLLM calls quant_config.get_quant_method(layer)
   - VllmMxfp4Config.get_quant_method() returns VllmMxfp4MoEMethod

3. WEIGHT CREATION:
   - VllmMxfp4MoEMethod.create_weights() is called
   - Allocates w13_weight, w2_weight, scales, and biases

4. WEIGHT LOADING:
   - Weights are loaded from safetensors/checkpoint
   - VllmMxfp4MoEMethod.process_weights_after_loading() is called
   - Dequantizes MXFP4 -> FP32 -> re-quantizes for TPU kernel

5. INFERENCE:
   - VllmMxfp4MoEMethod.apply() is called with input tensor
   - Routes tokens to experts using gating output
   - Calls TPU MoE kernel (GMM or fused MoE)
   - Returns output tensor
"""

from typing import Optional

import torch
from torch_tpu._internal import sync
from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig, mxfp4_w4a16_moe_quant_config)
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import (Mxfp4Backend,
                                                           Mxfp4Config,
                                                           Mxfp4MoEMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.quant_methods import (MXFP4,
                                                       get_tpu_quant_method)
from tpu_inference.layers.common.quantization import dequantize_mxfp4_packed
from tpu_inference.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 prebuild_fused_moe_kernel)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(MXFP4))
class VllmMxfp4Config(Mxfp4Config, VllmQuantConfig):
    """
    TPU-specific MXFP4 quantization configuration.

    This class is registered with vLLM's quantization system under the name
    "tpu-mxfp4" and provides TPU-compatible quantization methods for each
    layer type.
    """

    @classmethod
    def get_name(cls) -> str:
        """Return the base quantization method name."""
        return MXFP4

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        """
        Return the appropriate quantization method for a layer.

        This is called by vLLM during model construction to get the
        quantization handler for each layer.

        Args:
            layer: The layer to quantize.
            prefix: The parameter prefix (e.g., "model.layers.0.mlp").

        Returns:
            A QuantizeMethodBase implementation, or None to skip quantization.
        """
        if isinstance(layer, LinearBase):
            # MXFP4 only quantizes MoE layers, not linear layers.
            # For linear layers (attention projections, etc.), use unquantized method.
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            # MXFP4 linear layer is not implemented - use unquantized
            logger.warning_once(
                "MXFP4 linear layer is not implemented on TPU - "
                "using unquantized linear method.")
            return UnquantizedLinearMethod()

        elif isinstance(layer, FusedMoE):
            # FusedMoE is the main use case for MXFP4 (MoE models like GPT-OSS)
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config)

        elif isinstance(layer, Attention):
            logger.warning_once(
                "MXFP4 attention layer is not implemented on TPU. "
                "Skipping quantization for this layer.")

        return None


class VllmMxfp4MoEMethod(Mxfp4MoEMethod):
    """
    TPU-specific implementation of MXFP4 MoE quantization.

    This class handles:
    1. Weight creation with correct shapes for MXFP4 (inherited from Mxfp4MoEMethod)
    2. Weight processing after loading (dequant -> requant for TPU)
    3. Forward pass using TPU MoE kernels

    Key Design Decision:
        We inherit from Mxfp4MoEMethod to reuse its create_weights() method,
        but bypass its __init__ to avoid the GPU backend assertion.
        We set mxfp4_backend = TRITON as a marker for minimal weight processing.
    """

    def __init__(self, moe: FusedMoEConfig):
        """
        Initialize the MXFP4 MoE method for TPU.

        Args:
            moe: The MoE configuration from vLLM.
        """
        # Call FusedMoEMethodBase.__init__ directly to skip Mxfp4MoEMethod's
        # assertion that requires a GPU backend
        FusedMoEMethodBase.__init__(self, moe)

        # We use TRITON backend marker because it has minimal weight
        # post-processing requirements. This is just for compatibility,
        # actual execution uses TPU kernels.
        self.mxfp4_backend = Mxfp4Backend.TRITON

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Process weights: dequant → layout transform for GMM kernel.

        Pipeline:
        1. Dequantize MXFP4 packed → bfloat16
        2. Transpose: [E, out_dim, in_dim] → [E, in_dim, out_dim] for GMM
        3. Transform biases: [E, out_dim] → [E, 1, out_dim]

        TODO(requantization): For memory optimization, add:
            - quantize_tensor_to_fp4() with block_size=512
            - pack_fp4_indices() to pack back to uint8
            - Transform scales: swap axes 1↔2, expand_dims(axis=2)
        """

        assert isinstance(layer, FusedMoE)
        assert layer.moe_config.has_bias, "MXFP4 quantization requires bias."

        # 1. Dequantize MXFP4 packed → bfloat16
        # Compute on CPU for faster dequantization, then move back to TPU.
        # Compile time is an issue if we do this on TPU.
        w13_weight = dequantize_mxfp4_packed(
            layer.w13_weight.data.to("cpu"),
            layer.w13_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w13_weight.device)
        w2_weight = dequantize_mxfp4_packed(
            layer.w2_weight.data.to("cpu"),
            layer.w2_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w2_weight.device)

        # 2. Handle w13 interleaving for swigluoai activation
        # GPT-OSS stores w13 interleaved: even indices are w1 (gate), odd are w3 (up)
        # We un-interleave so first half is w1 and second half is w3
        w13_interleave = layer.activation == "swigluoai"
        if w13_interleave:
            # w13_weight shape: [E, out_dim, in_dim] where out_dim = 2 * intermediate
            w1_weight = w13_weight[:, ::2, :]  # even indices
            w3_weight = w13_weight[:, 1::2, :]  # odd indices
            w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

        # 3. Transpose: [E, out_dim, in_dim] → [E, in_dim, out_dim]
        # Keep weights contiguous after transpose. Non-contiguous parameter views
        # can trigger large as_strided/copy_from_as_strided_inverse graphs.
        w13_weight = w13_weight.transpose(
            1, 2).contiguous()  # [E, hidden, 2*intermediate]
        w2_weight = w2_weight.transpose(
            1, 2).contiguous()  # [E, intermediate, hidden]

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        # 4. Scales not needed for bfloat16 weights
        # TODO(requantization): Transform scales when using fp4:
        #   scale = scale.transpose(1, 2).unsqueeze(2)  # [E, num_blocks, 1, out_dim]
        layer.w13_weight_scale = None
        layer.w2_weight_scale = None

        # 5. Transform biases: [E, out_dim] → [E, 1, out_dim]
        # Also handle interleaving for swigluoai activation
        if layer.w13_bias is not None:
            w13_bias = layer.w13_bias.data
            if w13_interleave:
                # Un-interleave bias: even indices are w1, odd are w3
                w1_bias = w13_bias[:, ::2]
                w3_bias = w13_bias[:, 1::2]
                w13_bias = torch.cat([w1_bias, w3_bias], dim=1)

            layer.w13_bias = torch.nn.Parameter(
                w13_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
        if layer.w2_bias is not None:
            layer.w2_bias = torch.nn.Parameter(
                layer.w2_bias.data.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )

        # Materialize transformed weights during load to avoid deferred execution
        # surfacing later during runtime host copies.
        if layer.w13_weight.device.type == "tpu":
            # Sync one tensor at a time to avoid creating oversized combined
            # synchronization graphs across layers.
            sync.synchronize(layer.w13_weight, wait=True)
            sync.synchronize(layer.w2_weight, wait=True)
            if layer.w13_bias is not None:
                sync.synchronize(layer.w13_bias, wait=True)
            if layer.w2_bias is not None:
                sync.synchronize(layer.w2_bias, wait=True)

        # TODO, for distributed senario, the weight layout handling should be more complex

        logger.info_once(
            "MXFP4 weights dequantized to bfloat16 and transposed for GMM kernel."
        )
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            renormalize=layer.renormalize,
            activation=layer.activation,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> Optional[FusedMoEQuantConfig]:
        """Get the quantization config for fused MoE operations.

        Returns config with scale and bias tensors for the MoE kernel.
        Note: scales are None when using bfloat16 weights.
        """
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using GMM kernel."""
        return fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            gating_output=router_logits,
            topk=layer.moe_config.experts_per_token,
            renormalize=layer.renormalize,
            activation=layer.activation,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )
