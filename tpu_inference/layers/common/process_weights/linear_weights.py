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
"""Helpers for preparing linear weights for TPU runtime."""

from dataclasses import dataclass

import torch


@dataclass
class LinearWeights:
    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    bias: torch.Tensor | None


def process_linear_weights(weights: LinearWeights) -> LinearWeights:
    return weights


def process_blockwise_fp8_linear_weights(
    weight: torch.Tensor,
    *,
    weight_scale: torch.Tensor | None,
    weight_scale_inv: torch.Tensor | None,
    block_quant: bool,
    weight_block_size: tuple[int, int] | None,
    bias: torch.Tensor | None = None,
) -> LinearWeights:
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

    return process_linear_weights(
        LinearWeights(
            weight=weight_bf16,
            weight_scale=None,
            bias=bias,
        ))


def process_fp8_linear_weights(
    weight: torch.Tensor,
    *,
    weight_scale: torch.Tensor | None,
    weight_scale_inv: torch.Tensor | None,
    block_quant: bool,
    weight_block_size: tuple[int, int] | None,
    bias: torch.Tensor | None = None,
) -> LinearWeights:
    return process_blockwise_fp8_linear_weights(
        weight,
        weight_scale=weight_scale,
        weight_scale_inv=weight_scale_inv,
        block_quant=block_quant,
        weight_block_size=weight_block_size,
        bias=bias,
    )
