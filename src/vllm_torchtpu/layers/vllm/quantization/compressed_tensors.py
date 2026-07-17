# Copyright 2026 Google LLC
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
TPU compressed-tensors quantization support.

compressed-tensors (https://github.com/neuralmagic/compressed-tensors) is a
general checkpoint quantization format: a single checkpoint's config_groups
can each use a different weight/activation scheme (FP8, INT8, NVFP4, W4A8,
...), matched to layers by name, regex, or module-class target.

This module wraps vLLM's native CompressedTensorsConfig for checkpoint
parsing (ignore list, per-layer target -> scheme matching via
get_scheme_dict()) and dispatches each matched layer to a TPU quant method.
Only the FP8 (float-quantized) scheme is implemented, by adapting the match
into a VllmFp8Config and delegating to VllmFp8LinearMethodTPU /
VllmFp8MoEMethodTPU. Every other scheme is a TODO below.

For a fuller per-scheme implementation (FP8, INT8, NVFP4, W4A8, each with
its own CompressedTensorsScheme class), see vllm-project/tpu-inference's
tpu_inference/layers/vllm/quantization/compressed_tensors/, which this
module is modeled on.
"""

from typing import Optional

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig

from vllm_torchtpu.layers.common.quant_methods import (COMPRESSED_TENSORS,
                                                       get_tpu_quant_method)
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.vllm.quantization.fp8 import (VllmFp8Config,
                                                        VllmFp8LinearMethodTPU,
                                                        VllmFp8MoEMethodTPU)
from vllm_torchtpu.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _is_fp8_scheme(weight_quant: Optional[QuantizationArgs]) -> bool:
    return (weight_quant is not None
            and weight_quant.type == QuantizationType.FLOAT
            and weight_quant.num_bits == 8)


def _build_fp8_config(
    weight_quant: QuantizationArgs,
    input_quant: Optional[QuantizationArgs],
) -> VllmFp8Config:
    """Adapt a matched compressed-tensors FP8 scheme into a VllmFp8Config.

    This reuses VllmFp8Config/VllmFp8LinearMethodTPU/VllmFp8MoEMethodTPU's
    existing dequant/requant runtime path instead of duplicating it.
    """
    weight_block_size = (weight_quant.block_structure if weight_quant.strategy
                         == QuantizationStrategy.BLOCK else None)
    activation_scheme = ("dynamic" if (input_quant is None
                                       or input_quant.dynamic) else "static")
    fp8_config = VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=activation_scheme,
        weight_block_size=weight_block_size,
    )
    fp8_config.is_channel_quant = (
        weight_quant.strategy == QuantizationStrategy.CHANNEL)
    return fp8_config


def _get_moe_scheme_dict(
    quant_config: "VllmCompressedTensorsConfig",
    layer: FusedMoE,
    layer_name: str,
) -> Optional[dict]:
    """Resolve the compressed-tensors scheme for a fused MoE experts layer.

    RoutedExperts/FusedMoE fuses each expert's separate gate_proj/up_proj/
    down_proj Linears into one module, so config_groups targets (often just
    "Linear") don't match the fused layer's own prefix directly. Mirrors
    vLLM's CompressedTensorsMoEMethod.get_moe_method: populate
    target_scheme_map["FusedMoE"] from target_scheme_map["Linear"], then
    probe one expert's unfused per-projection paths and require them to
    agree on a single scheme.
    """
    quant_config._add_fused_moe_to_target_scheme_map()
    unfused_names = [
        layer_name + proj_name
        for proj_name in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
    ]
    all_scheme_dicts = [
        quant_config.get_scheme_dict(layer, name) for name in unfused_names
    ]
    scheme_dict = all_scheme_dicts.pop()
    if not all(d == scheme_dict for d in all_scheme_dicts):
        raise ValueError(
            "All MoE projections need to have the same quantization scheme, "
            f"but found multiple for {layer_name!r}: {all_scheme_dicts}")
    return scheme_dict


@register_quantization_config(get_tpu_quant_method(COMPRESSED_TENSORS))
class VllmCompressedTensorsConfig(CompressedTensorsConfig, VllmQuantConfig):
    """The "compressed-tensors" quant_method entry point on TPU.

    Only the FP8 (float-quantized) scheme is implemented; every other
    compressed-tensors scheme (INT8, NVFP4, W4A8, ...) raises
    NotImplementedError below with a TODO.
    """

    @classmethod
    def get_name(cls) -> str:
        return COMPRESSED_TENSORS

    @classmethod
    def set_configs(cls, vllm_config) -> None:
        super().set_configs(vllm_config)
        # get_quant_method() below builds VllmFp8Config instances and calls
        # their (inherited from VllmQuantConfig) get_linear_config() /
        # get_moe_config(), which read VllmFp8Config's own class-level
        # vllm_config -- so it must be populated too, not just ours.
        VllmFp8Config.set_configs(vllm_config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, FusedMoE):
            scheme_dict = _get_moe_scheme_dict(self, layer, prefix)
        else:
            scheme_dict = self.get_scheme_dict(layer, prefix)
        weight_quant = scheme_dict.get("weights") if scheme_dict else None
        input_quant = (scheme_dict.get("input_activations")
                       if scheme_dict else None)

        if isinstance(layer, FusedMoE):
            if weight_quant is None:
                return VllmUnquantizedFusedMoEMethod(
                    self.get_moe_config(layer))
            if not _is_fp8_scheme(weight_quant):
                # TODO: INT8 / NVFP4 / W4A8 MoE schemes are not implemented
                # on TPU yet. See tpu-inference's compressed_tensors_moe/
                # for a reference per-scheme dispatch to port from.
                raise NotImplementedError(
                    f"compressed-tensors MoE scheme for {prefix!r} "
                    f"(weights={weight_quant}) is not supported on TPU yet; "
                    "only FP8 (float-quantized) is implemented.")
            fp8_config = _build_fp8_config(weight_quant, input_quant)
            return VllmFp8MoEMethodTPU(fp8_config,
                                       fp8_config.get_moe_config(layer))

        if isinstance(layer, LinearBase):
            if weight_quant is None:
                return UnquantizedLinearMethod()
            if not _is_fp8_scheme(weight_quant):
                # TODO: same as the MoE case above, for linear layers.
                raise NotImplementedError(
                    f"compressed-tensors scheme for {prefix!r} "
                    f"(weights={weight_quant}) is not supported on TPU yet; "
                    "only FP8 (float-quantized) is implemented.")
            fp8_config = _build_fp8_config(weight_quant, input_quant)
            return VllmFp8LinearMethodTPU(fp8_config,
                                          fp8_config.get_linear_config(layer))

        if isinstance(layer, Attention):
            # TODO: KV-cache quantization for compressed-tensors checkpoints
            # is not implemented on TPU yet.
            return None

        return None
