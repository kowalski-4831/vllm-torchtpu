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

Dequantized from FP8 to BF16 at load time because the active TorchTPU-vLLM
linear runtime path currently uses BF16 linears. The vLLM
Fp8LinearMethod.create_weights() is reused for correct parameter allocation
(weight_scale_inv), then process_weights dequantizes using block scales.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch_tpu._internal import sync
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

from tpu_inference import envs
from tpu_inference.layers.common.quant_methods import FP8, get_tpu_quant_method
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 prebuild_fused_moe_kernel)
from tpu_inference.layers.vllm.moe_routing import (select_experts,
                                                   select_experts_ep)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_MOE_REQUANT_WEIGHT_DTYPES = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "int8": torch.int8,
}
if hasattr(torch, "float8_e5m2"):
    _MOE_REQUANT_WEIGHT_DTYPES["float8_e5m2"] = torch.float8_e5m2


def _get_activation_str(activation) -> str:
    """Convert MoEActivation enum or string to plain string."""
    return activation.value if hasattr(activation,
                                       'value') else str(activation)


def _dequantize_fp8_linear(
    weight: torch.Tensor,
    *,
    weight_scale: torch.Tensor | None,
    weight_scale_inv: torch.Tensor | None,
    block_quant: bool,
    weight_block_size: tuple[int, int] | None,
) -> torch.Tensor:
    """Dequantize FP8 linear weights to BF16."""
    weight_bf16 = weight.to(torch.bfloat16)

    if block_quant:
        assert weight_scale_inv is not None
        assert weight_block_size is not None
        block_h, block_w = weight_block_size

        out_dim, in_dim = weight_bf16.shape
        weight_bf16 = weight_bf16.reshape(out_dim // block_h, block_h,
                                          in_dim // block_w, block_w)
        weight_bf16 = weight_bf16 * weight_scale_inv.to(
            torch.bfloat16).unsqueeze(1).unsqueeze(3)
        weight_bf16 = weight_bf16.reshape(out_dim, in_dim)
    else:
        assert weight_scale is not None
        weight_bf16 = weight_bf16 * weight_scale.to(torch.bfloat16)

    return weight_bf16


def _process_fp8_moe_weights(
    layer: FusedMoE,
    *,
    weight_block_size: tuple[int, int],
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int
           | None]:
    """Dequantize and requantize FP8 MoE weights for the TPU GMM kernel.

    Returns (w13_weight, w13_scale, w2_weight, w2_scale,
             requant_dtype_name, requant_block_size).
    """
    if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
        desired_quant_dtype = desired_quant_dtype_from_env
    else:
        desired_quant_dtype = "float8_e4m3fn"

    requant_block_size = None
    if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
        requant_block_size = (int(requant_block_size_from_env)
                              if requant_block_size_from_env else None)

    requant_dtype = _MOE_REQUANT_WEIGHT_DTYPES.get(desired_quant_dtype)
    if requant_dtype is None:
        supported = ", ".join(sorted(_MOE_REQUANT_WEIGHT_DTYPES))
        raise ValueError(
            "Unsupported MOE_REQUANTIZE_WEIGHT_DTYPE="
            f"{desired_quant_dtype!r}. Supported values: {supported}.")

    moe_logging_str = (
        f"[MoE requantization]: re-quantizing MoE weights to {desired_quant_dtype}"
    )
    if requant_block_size is not None:
        moe_logging_str += f" with block size {requant_block_size}"
    logger.info_once(moe_logging_str)

    # Validate block alignment
    block_h, block_w = weight_block_size
    for name, weight in (("w13_weight", layer.w13_weight.data),
                         ("w2_weight", layer.w2_weight.data)):
        _, out_dim, in_dim = weight.shape
        if out_dim % block_h != 0 or in_dim % block_w != 0:
            raise ValueError(
                "FP8 block quantized MoE weights must be divisible by the checkpoint "
                f"block size, got {name}.shape={tuple(weight.shape)} and "
                f"block_size={weight_block_size}.")

    # Dequantize from checkpoint FP8 to float32
    w13 = dequantize_tensor(
        layer.w13_weight.data,
        layer.w13_weight_scale_inv.data,
        axis=(1, 2),
        out_dtype=torch.float32,
    )
    w2 = dequantize_tensor(
        layer.w2_weight.data,
        layer.w2_weight_scale_inv.data,
        axis=(1, 2),
        out_dtype=torch.float32,
    )

    # Requantize to runtime format
    if requant_block_size is not None:
        if w13.shape[-1] % requant_block_size != 0:
            raise ValueError(
                "Unsupported MoE requantization configuration: w13 contracting "
                f"dimension {w13.shape[-1]} is not divisible by "
                f"block_size={requant_block_size}. Padding is not implemented yet."
            )
        if w2.shape[-1] % requant_block_size != 0:
            raise ValueError(
                "Unsupported MoE requantization configuration: w2 contracting "
                f"dimension {w2.shape[-1]} is not divisible by "
                f"block_size={requant_block_size}. Padding is not implemented yet."
            )

    w13, w13_scale = quantize_tensor(w13,
                                     quant_dtype=requant_dtype,
                                     axis=-1,
                                     block_size=requant_block_size)
    w2, w2_scale = quantize_tensor(w2,
                                   quant_dtype=requant_dtype,
                                   axis=-1,
                                   block_size=requant_block_size)

    # Deinterleave w1/w3 if activation is swigluoai
    if activation == "swigluoai":
        w1 = w13[:, ::2, :]
        w3 = w13[:, 1::2, :]
        w13 = torch.cat([w1, w3], dim=1)

    # Transpose to [experts, in_dim, out_dim] for GMM kernel
    w13 = w13.transpose(1, 2).contiguous()
    w2 = w2.transpose(1, 2).contiguous()

    # Reshape scales for GMM kernel
    if w13_scale is not None:
        w13_scale = w13_scale.transpose(1, 2).contiguous()
        w13_scale = w13_scale.unsqueeze(2).to(torch.float32)
    if w2_scale is not None:
        w2_scale = w2_scale.transpose(1, 2).contiguous()
        w2_scale = w2_scale.unsqueeze(2).to(torch.float32)

    return w13, w13_scale, w2, w2_scale, desired_quant_dtype, requant_block_size


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


class VllmFp8MoEMethodTPU(Fp8MoEMethod):
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
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")
        self.fp8_backend = None

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer):
        return None

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
        assert not self.moe.has_bias, "TPU FP8 MoE path does not support bias."

        activation_str = _get_activation_str(layer.activation)
        layer._tpu_activation_str = activation_str

        (w13, w13_scale, w2, w2_scale, requant_dtype_name,
         requant_block_size) = _process_fp8_moe_weights(
             layer,
             weight_block_size=tuple(self.weight_block_size),
             activation=activation_str,
         )

        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale_inv = torch.nn.Parameter(w13_scale,
                                                        requires_grad=False)
        layer.w2_weight_scale_inv = torch.nn.Parameter(w2_scale,
                                                       requires_grad=False)

        # Eagerly materialize weights to avoid OOM during vLLM memory probing.
        # Without this, lazy tensors accumulate and the profiling run OOMs.
        if layer.w13_weight.device.type == "tpu":
            sync.synchronize(layer.w13_weight, wait=True)
            sync.synchronize(layer.w2_weight, wait=True)
            sync.synchronize(layer.w13_weight_scale_inv, wait=True)
            sync.synchronize(layer.w2_weight_scale_inv, wait=True)

        scale_desc = ("per-channel" if requant_block_size is None else
                      str(requant_block_size))
        logger.info_once(
            "FP8 weights transposed for GMM kernel: "
            f"w13={list(layer.w13_weight.shape)}, "
            f"w2={list(layer.w2_weight.shape)}, "
            f"w13_scale={list(layer.w13_weight_scale_inv.shape)}, "
            f"w2_scale={list(layer.w2_weight_scale_inv.shape)}, "
            f"requant_block_size={scale_desc}")
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
        )

    def apply_monolithic(
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
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
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
    to BF16 after loading because this runtime path currently uses BF16
    linears.

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
        weight_bf16 = _dequantize_fp8_linear(
            layer.weight.data,
            weight_scale=getattr(layer, "weight_scale", None).data if hasattr(
                layer, "weight_scale") else None,
            weight_scale_inv=getattr(layer, "weight_scale_inv", None).data
            if hasattr(layer, "weight_scale_inv") else None,
            block_quant=self.block_quant,
            weight_block_size=tuple(self.weight_block_size)
            if self.weight_block_size is not None else None,
        )

        replace_parameter(layer, "weight",
                          torch.nn.Parameter(weight_bf16, requires_grad=False))

        logger.info_once("FP8 linear weights dequantized to BF16: "
                         f"shape={list(weight_bf16.shape)}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
