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
        """
        Process weights after they've been loaded from checkpoint.

        This dequantizes MXFP4 weights to bfloat16 for TPU inference.
        MXFP4 format:
        - Weights are packed as uint8 (2 x fp4 per byte)
        - Scales are e8m0 stored as uint8

        SIMPLIFICATION vs Old tpu_inference Implementation:
        =====================================================
        This is a simplified implementation that differs from old tpu_inference:

        1. NO RE-QUANTIZATION: Old tpu_inference re-quantizes dequantized weights
           to fp4 with REQUANTIZED_BLOCK_SIZE for optimized TPU kernels.
           We keep weights as bfloat16 for simpler apply() logic.

        2. NO SWIGLU REORDERING: Old tpu_inference calls process_moe_weights() to
           interleave gate/up projections for efficient SwiGLU.
           We keep original layout and split manually in apply().

        3. NO SHARDING: Old tpu_inference calls shard_moe_weights() for tensor
           parallelism. We assume single-device execution.

        For production performance, port the full pipeline from old tpu_inference.

        Args:
            layer: The FusedMoE layer with loaded weights.
        """
        assert isinstance(layer, FusedMoE)
        assert layer.moe_config.has_bias, "MXFP4 quantization requires bias."

        # Dequantize w13 (gate_up_proj)
        w13_dequant = self._dequantize_mxfp4_packed(
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
        )

        # Dequantize w2 (down_proj)
        w2_dequant = self._dequantize_mxfp4_packed(
            layer.w2_weight.data,
            layer.w2_weight_scale.data,
        )

        # Update layer weights with dequantized values
        # Store as bfloat16 for TPU inference
        layer.w13_weight = torch.nn.Parameter(w13_dequant.to(torch.bfloat16),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_dequant.to(torch.bfloat16),
                                             requires_grad=False)

        # Scales are no longer needed after dequantization, but we keep them
        # for compatibility with get_fused_moe_quant_config
        # Convert to float32 for any downstream use
        layer.w13_weight_scale = torch.nn.Parameter(self._e8m0_to_fp32(
            layer.w13_weight_scale.data),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(self._e8m0_to_fp32(
            layer.w2_weight_scale.data),
                                                   requires_grad=False)

        logger.info_once(
            "MXFP4 weights dequantized to bfloat16 for TPU inference.")

    def _dequantize_mxfp4_packed(
        self,
        weight_packed: torch.Tensor,
        scale_u8: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize MXFP4 packed weights to float32.

        MXFP4 format:
        - Each uint8 contains 2 x 4-bit float values (e2m1 format)
        - Scales are e8m0 (8-bit exponent only) stored as uint8
        - Block size is 32 (32 values share one scale)

        Args:
            weight_packed: Packed weights [E, out_dim, in_dim/2] as uint8
            scale_u8: Scales [E, out_dim, in_dim/32] as uint8

        Returns:
            Dequantized weights [E, out_dim, in_dim] as float32
        """
        # Step 1: Unpack uint8 -> two fp4 values
        # Each uint8 byte contains: [fp4_low (bits 0-3), fp4_high (bits 4-7)]
        weight_unpacked = self._unpack_uint8_to_fp4(weight_packed)

        # Step 2: Convert e8m0 scales to float32
        scale_fp32 = self._e8m0_to_fp32(scale_u8)

        # Step 3: Apply block-wise dequantization
        # weight shape: [E, out_dim, in_dim]
        # scale shape: [E, out_dim, in_dim/32]
        # We need to broadcast scale to match weight dimensions
        block_size = weight_unpacked.shape[-1] // scale_fp32.shape[-1]

        # Expand scale to match weight shape
        # [E, out_dim, num_blocks] -> [E, out_dim, num_blocks, 1]
        scale_expanded = scale_fp32.unsqueeze(-1)
        # [E, out_dim, num_blocks, 1] -> [E, out_dim, num_blocks, block_size]
        scale_expanded = scale_expanded.expand(*scale_fp32.shape, block_size)
        # [E, out_dim, num_blocks, block_size] -> [E, out_dim, in_dim]
        scale_expanded = scale_expanded.reshape(weight_unpacked.shape)

        # Dequantize: weight * scale
        weight_dequant = weight_unpacked * scale_expanded

        return weight_dequant

    def _unpack_uint8_to_fp4(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack uint8 tensor containing two fp4 (e2m1) values per byte.

        FP4 e2m1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
        Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

        Args:
            packed: uint8 tensor [..., N/2]

        Returns:
            Unpacked float32 tensor [..., N]
        """
        # Extract low and high nibbles
        low_nibble = (packed & 0x0F).to(torch.int8)  # bits 0-3
        high_nibble = ((packed >> 4) & 0x0F).to(torch.int8)  # bits 4-7

        # Convert fp4 nibbles to float32
        low_fp32 = self._fp4_e2m1_to_fp32(low_nibble)
        high_fp32 = self._fp4_e2m1_to_fp32(high_nibble)

        # Interleave: [low0, high0, low1, high1, ...]
        # Shape: [..., N/2] -> [..., N/2, 2] -> [..., N]
        result = torch.stack([low_fp32, high_fp32], dim=-1)
        return result.reshape(*packed.shape[:-1], -1)

    def _fp4_e2m1_to_fp32(self, fp4: torch.Tensor) -> torch.Tensor:
        """
        Convert 4-bit float (e2m1) to float32.

        FP4 e2m1 encoding (4 bits):
        - Bit 3: sign (0=positive, 1=negative)
        - Bits 1-2: exponent (2 bits, bias=1)
        - Bit 0: mantissa (1 bit)

        Lookup table for all 16 possible values:
        0000=0, 0001=0.5, 0010=1, 0011=1.5, 0100=2, 0101=3, 0110=4, 0111=6
        1000=-0, 1001=-0.5, 1010=-1, 1011=-1.5, 1100=-2, 1101=-3, 1110=-4, 1111=-6
        """
        # Use lookup table for conversion (most efficient)
        fp4_lut = torch.tensor(
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5,
                -2.0, -3.0, -4.0, -6.0
            ],
            dtype=torch.float32,
            device=fp4.device,
        )
        return fp4_lut[fp4.to(torch.long)]

    def _e8m0_to_fp32(self, u8: torch.Tensor) -> torch.Tensor:
        """
        Convert e8m0 (8-bit exponent only) to float32.

        e8m0 format: 8-bit unsigned exponent with bias 127
        Value = 2^(exponent - 127)

        Args:
            u8: uint8 tensor representing e8m0 exponents

        Returns:
            float32 tensor with actual scale values
        """
        # e8m0 bias is 127 (same as fp32 exponent bias)
        E8M0_BIAS = 127
        exponents = u8.to(torch.int32) - E8M0_BIAS
        # 2^exponent = ldexp(1.0, exponent)
        return torch.ldexp(torch.ones_like(u8, dtype=torch.float32), exponents)

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> Optional[FusedMoEQuantConfig]:
        """
        Get the quantization config for fused MoE operations.

        Returns config with scale and bias tensors for the MoE kernel.

        Args:
            layer: The FusedMoE layer.

        Returns:
            FusedMoEQuantConfig with w1/w2 scales and biases.
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
        """
        Forward pass for MXFP4 MoE on TPU.

        SIMPLIFICATION vs Old tpu_inference Implementation:
        =====================================================
        This is a simplified implementation using bfloat16 weights and
        standard PyTorch operations. Old tpu_inference uses:
        - Quantized fp4 kernels via fused_moe_apply()
        - JAX GMM (Group Matrix Multiply) for batched expert computation
        - Pre-computed routing tables for efficiency

        For production performance, integrate with fused_moe.py or port
        the optimized kernels from old tpu_inference.

        Args:
            layer: The FusedMoE layer with dequantized bfloat16 weights.
            x: Input tensor of shape [num_tokens, hidden_size].
            router_logits: Router logits of shape [num_tokens, num_experts].

        Returns:
            Output tensor of shape [num_tokens, hidden_size].
        """
        num_tokens, hidden_size = x.shape
        num_experts = layer.moe_config.num_experts
        top_k = layer.moe_config.experts_per_token
        logger.debug(
            "Forward pass for MXFP4 MoE: num_tokens=%d, hidden_size=%d, num_experts=%d, top_k=%d",
            num_tokens, hidden_size, num_experts, top_k)


        # Step 1: Compute top-k routing
        # router_logits: [num_tokens, num_experts]
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)

        # Renormalize top-k weights
        if layer.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                           keepdim=True)

        # Step 2: Prepare output tensor
        output = torch.zeros_like(x)

        # Step 3: Process each expert
        # w13_weight: [E, 2*intermediate, hidden] (gate and up projections fused)
        # w2_weight: [E, hidden, intermediate] (down projection)
        # w13_bias: [E, 2*intermediate]
        # w2_bias: [E, hidden]

        for k in range(top_k):
            expert_indices = topk_indices[:, k]  # [num_tokens]
            expert_weights = topk_weights[:, k]  # [num_tokens]

            # For each unique expert, process its assigned tokens
            for expert_id in range(num_experts):
                # Find tokens assigned to this expert using nonzero (TPU-compatible)
                # NOTE: Boolean indexing (x[mask]) is not currently supported on TPU.
                # If supported in the future, we can revert to simpler boolean masking.
                token_mask = expert_indices == expert_id
                token_indices = torch.nonzero(token_mask, as_tuple=True)[0]

                if token_indices.numel() == 0:
                    continue

                # Gather tokens for this expert using index_select (TPU-compatible)
                expert_input = torch.index_select(x, 0, token_indices)
                expert_routing_weight = torch.index_select(
                    expert_weights, 0, token_indices)

                # Get expert weights
                w13 = layer.w13_weight[expert_id]  # [2*intermediate, hidden]
                w2 = layer.w2_weight[expert_id]  # [hidden, intermediate]
                w13_bias = layer.w13_bias[expert_id]  # [2*intermediate]
                w2_bias = layer.w2_bias[expert_id]  # [hidden]

                # Gate-Up projection: x @ w13.T + bias
                # Output: [num_expert_tokens, 2*intermediate]
                gate_up = torch.nn.functional.linear(expert_input, w13,
                                                     w13_bias)

                # Split gate and up projections
                gate, up = gate_up.chunk(2, dim=-1)

                # SwiGLU activation: silu(gate) * up
                hidden_states = torch.nn.functional.silu(gate) * up

                # Down projection: hidden @ w2.T + bias
                # Output: [num_expert_tokens, hidden]
                expert_output = torch.nn.functional.linear(
                    hidden_states, w2, w2_bias)

                # Apply routing weight
                expert_output = expert_output * expert_routing_weight.unsqueeze(
                    -1)

                # Scatter back to output using index_add (TPU-compatible)
                output.index_add_(0, token_indices, expert_output)

        return output
