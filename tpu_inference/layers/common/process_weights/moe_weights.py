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

import os
from dataclasses import dataclass

import torch
from torch_tpu._internal import sync
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

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


def _dequantize_fp8_block_weight(
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    block_h, block_w = block_size
    num_experts, out_dim, in_dim = weight_q.shape
    if out_dim % block_h != 0 or in_dim % block_w != 0:
        raise ValueError(
            "FP8 block quantized MoE weights must be divisible by the checkpoint "
            f"block size, got weight.shape={tuple(weight_q.shape)} and "
            f"block_size={block_size}.")

    weight = weight_q.to(torch.float32).reshape(
        num_experts,
        out_dim // block_h,
        block_h,
        in_dim // block_w,
        block_w,
    )
    weight = weight * weight_scale.to(torch.float32).unsqueeze(2).unsqueeze(4)
    return weight.reshape(num_experts, out_dim, in_dim)


def _quantize_weight_along_last_dim(
    weight: torch.Tensor,
    quant_dtype: torch.dtype,
    block_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = block_size or weight.shape[-1]
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            "MoE requantization expects the contracting dimension to be divisible "
            f"by block_size, got weight.shape={tuple(weight.shape)} and "
            f"block_size={block_size}.")

    if torch.empty((), dtype=quant_dtype).is_floating_point():
        dtype_info = torch.finfo(quant_dtype)
    else:
        dtype_info = torch.iinfo(quant_dtype)

    num_blocks = weight.shape[-1] // block_size
    weight_blocks = weight.reshape(*weight.shape[:-1], num_blocks, block_size)
    abs_max = weight_blocks.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / float(dtype_info.max)
    scale_inv = torch.where(scale == 0, torch.full_like(scale, float("inf")),
                            1.0 / scale)
    weight_q = torch.clamp(weight_blocks * scale_inv,
                           min=float(dtype_info.min),
                           max=float(dtype_info.max))
    weight_q = weight_q.reshape_as(weight).to(quant_dtype)
    scale = scale.squeeze(-1).to(torch.float32)
    return weight_q, scale


def quantize_moe_weights(
    weights: FusedMoEWeights,
    dtype: torch.dtype,
    block_size: int | None,
) -> FusedMoEWeights:
    weights.w13_weight, weights.w13_weight_scale = _quantize_weight_along_last_dim(
        weights.w13_weight,
        quant_dtype=dtype,
        block_size=block_size,
    )
    weights.w2_weight, weights.w2_weight_scale = _quantize_weight_along_last_dim(
        weights.w2_weight,
        quant_dtype=dtype,
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
    weight_scale_name: str,
    checkpoint_block_size: tuple[int, int],
    activation: str,
) -> tuple[FusedMoEWeights, str, int | None]:
    requant_block_size_env = os.getenv("MOE_REQUANTIZE_BLOCK_SIZE")
    requant_block_size = (None if requant_block_size_env in (None, "") else
                          int(requant_block_size_env))
    requant_dtype_name = os.getenv("MOE_REQUANTIZE_WEIGHT_DTYPE",
                                   "float8_e4m3fn")
    requant_dtype = _MOE_REQUANT_WEIGHT_DTYPES.get(requant_dtype_name)
    if requant_dtype is None:
        supported = ", ".join(sorted(_MOE_REQUANT_WEIGHT_DTYPES))
        raise ValueError(
            "Unsupported MOE_REQUANTIZE_WEIGHT_DTYPE="
            f"{requant_dtype_name!r}. Supported values: {supported}.")

    weights = FusedMoEWeights(
        w13_weight=_dequantize_fp8_block_weight(
            layer.w13_weight.data,
            getattr(layer, f"w13_{weight_scale_name}").data,
            checkpoint_block_size,
        ),
        w13_weight_scale=None,
        w13_bias=layer.w13_bias.data if layer.moe_config.has_bias else None,
        w2_weight=_dequantize_fp8_block_weight(
            layer.w2_weight.data,
            getattr(layer, f"w2_{weight_scale_name}").data,
            checkpoint_block_size,
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
    return weights, requant_dtype_name, requant_block_size


def materialize_moe_weights(
    layer: FusedMoE,
    weight_scale_name: str,
) -> None:
    if layer.w13_weight.device.type != "tpu":
        return

    sync.synchronize(layer.w13_weight, wait=True)
    sync.synchronize(layer.w2_weight, wait=True)
    sync.synchronize(getattr(layer, f"w13_{weight_scale_name}"), wait=True)
    sync.synchronize(getattr(layer, f"w2_{weight_scale_name}"), wait=True)
    if layer.moe_config.has_bias:
        sync.synchronize(layer.w13_bias, wait=True)
        sync.synchronize(layer.w2_bias, wait=True)


def log_and_prebuild_fp8_moe(
    layer: FusedMoE,
    *,
    weight_scale_name: str,
    requant_dtype_name: str,
    requant_block_size: int | None,
    activation: str,
) -> None:
    scale_desc = ("per-channel"
                  if requant_block_size is None else str(requant_block_size))
    logger.info_once("[MoE requantization]: re-quantizing MoE weights to "
                     f"{requant_dtype_name}" +
                     ("" if requant_block_size is
                      None else f" with block size {requant_block_size}"))
    logger.info_once(
        "FP8 weights transposed for GMM kernel: "
        f"w13={list(layer.w13_weight.shape)}, "
        f"w2={list(layer.w2_weight.shape)}, "
        f"w13_scale={list(getattr(layer, f'w13_{weight_scale_name}').shape)}, "
        f"w2_scale={list(getattr(layer, f'w2_{weight_scale_name}').shape)}, "
        f"requant_block_size={scale_desc}")
    prebuild_fused_moe_kernel(
        topk=layer.moe_config.experts_per_token,
        activation=activation,
    )
