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
"""Helpers for preparing MoE weights for TPU runtime kernels."""
from dataclasses import dataclass

import torch
from torch_tpu._internal import sync
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from tpu_inference import envs
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.vllm.fused_moe import prebuild_fused_moe_kernel
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_MOE_REQUANT_WEIGHT_DTYPES = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "int8": torch.int8,
}
if hasattr(torch, "float8_e5m2"):
    _MOE_REQUANT_WEIGHT_DTYPES["float8_e5m2"] = torch.float8_e5m2


@dataclass
class FusedMoEWeights:
    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor | None
    w13_bias: torch.Tensor | None
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor | None
    w2_bias: torch.Tensor | None


def quantize_moe_weights(
    weights: FusedMoEWeights,
    dtype: torch.dtype,
    block_size: int | None,
) -> FusedMoEWeights:
    if block_size is not None:
        if weights.w13_weight.shape[-1] % block_size != 0:
            raise ValueError(
                "Unsupported MoE requantization configuration: w13 contracting "
                f"dimension {weights.w13_weight.shape[-1]} is not divisible by "
                f"block_size={block_size}. Padding is not implemented yet.")
        if weights.w2_weight.shape[-1] % block_size != 0:
            raise ValueError(
                "Unsupported MoE requantization configuration: w2 contracting "
                f"dimension {weights.w2_weight.shape[-1]} is not divisible by "
                f"block_size={block_size}. Padding is not implemented yet.")

    weights.w13_weight, weights.w13_weight_scale = quantize_tensor(
        weights.w13_weight,
        quant_dtype=dtype,
        axis=-1,
        block_size=block_size,
    )
    weights.w2_weight, weights.w2_weight_scale = quantize_tensor(
        weights.w2_weight,
        quant_dtype=dtype,
        axis=-1,
        block_size=block_size,
    )
    return weights


def process_moe_weights(
    weights: FusedMoEWeights,
    *,
    w13_interleave: bool,
) -> FusedMoEWeights:
    if w13_interleave:
        w1_weight = weights.w13_weight[:, ::2, :]
        w3_weight = weights.w13_weight[:, 1::2, :]
        weights.w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

        if weights.w13_bias is not None:
            w1_bias = weights.w13_bias[:, ::2]
            w3_bias = weights.w13_bias[:, 1::2]
            weights.w13_bias = torch.cat([w1_bias, w3_bias], dim=1)

    weights.w13_weight = weights.w13_weight.transpose(1, 2).contiguous()
    weights.w2_weight = weights.w2_weight.transpose(1, 2).contiguous()

    if weights.w13_weight_scale is not None:
        weights.w13_weight_scale = weights.w13_weight_scale.transpose(
            1, 2).contiguous()
        weights.w13_weight_scale = weights.w13_weight_scale.unsqueeze(2).to(
            torch.float32)
    if weights.w2_weight_scale is not None:
        weights.w2_weight_scale = weights.w2_weight_scale.transpose(
            1, 2).contiguous()
        weights.w2_weight_scale = weights.w2_weight_scale.unsqueeze(2).to(
            torch.float32)
    if weights.w13_bias is not None:
        weights.w13_bias = weights.w13_bias.unsqueeze(1).to(torch.float32)
    if weights.w2_bias is not None:
        weights.w2_bias = weights.w2_bias.unsqueeze(1).to(torch.float32)

    return weights


def process_fp8_moe_weights(
    layer: FusedMoE,
    *,
    weight_block_size: tuple[int, int],
    activation: str,
) -> tuple[FusedMoEWeights, str, int | None]:
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

    block_h, block_w = weight_block_size
    for name, weight in (("w13_weight", layer.w13_weight.data),
                         ("w2_weight", layer.w2_weight.data)):
        _, out_dim, in_dim = weight.shape
        if out_dim % block_h != 0 or in_dim % block_w != 0:
            raise ValueError(
                "FP8 block quantized MoE weights must be divisible by the checkpoint "
                f"block size, got {name}.shape={tuple(weight.shape)} and "
                f"block_size={weight_block_size}.")

    weights = FusedMoEWeights(
        w13_weight=dequantize_tensor(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            axis=(1, 2),
            out_dtype=torch.float32,
        ),
        w13_weight_scale=None,
        w13_bias=layer.w13_bias.data if layer.moe_config.has_bias else None,
        w2_weight=dequantize_tensor(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            axis=(1, 2),
            out_dtype=torch.float32,
        ),
        w2_weight_scale=None,
        w2_bias=layer.w2_bias.data if layer.moe_config.has_bias else None,
    )

    weights = quantize_moe_weights(
        weights,
        dtype=requant_dtype,
        block_size=requant_block_size,
    )
    weights = process_moe_weights(
        weights,
        w13_interleave=activation == "swigluoai",
    )
    return weights, desired_quant_dtype, requant_block_size


def materialize_moe_weights(layer: FusedMoE, ) -> None:
    if layer.w13_weight.device.type != "tpu":
        return

    # TODO(geyuhao): This eager materialization should not be needed.
    # But now, without this we will get OOM during memory probing.
    sync.synchronize(layer.w13_weight, wait=True)
    sync.synchronize(layer.w2_weight, wait=True)
    sync.synchronize(layer.w13_weight_scale_inv, wait=True)
    sync.synchronize(layer.w2_weight_scale_inv, wait=True)
    if layer.moe_config.has_bias:
        sync.synchronize(layer.w13_bias, wait=True)
        sync.synchronize(layer.w2_bias, wait=True)


def log_and_prebuild_fp8_moe(
    layer: FusedMoE,
    *,
    requant_dtype_name: str,
    requant_block_size: int | None,
    activation: str,
) -> None:
    scale_desc = ("per-channel"
                  if requant_block_size is None else str(requant_block_size))
    logger.info_once("FP8 weights transposed for GMM kernel: "
                     f"w13={list(layer.w13_weight.shape)}, "
                     f"w2={list(layer.w2_weight.shape)}, "
                     f"w13_scale={list(layer.w13_weight_scale_inv.shape)}, "
                     f"w2_scale={list(layer.w2_weight_scale_inv.shape)}, "
                     f"requant_block_size={scale_desc}")
    prebuild_fused_moe_kernel(
        topk=layer.moe_config.experts_per_token,
        activation=activation,
    )
