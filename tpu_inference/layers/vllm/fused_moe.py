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
"""Torch bridge for fused MoE based on fused_moe_func in fused_moe_gmm."""

from typing import Callable, Optional

import torch
from torch_tpu._internal import pallas

from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func


def _pallas_fused_moe_kernel(
    topk: int,
    activation: str,
):
    """Build a pallas custom kernel wrapper around fused_moe_func."""

    def impl(hidden_states, w1, w2, w1_scale, w2_scale, w1_bias, w2_bias,
             topk_weights, topk_ids):
        return fused_moe_func(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk=topk,
            activation=activation,
        )

    return impl


_kernel_instance_counter = 0
_fused_moe_kernel_cache: dict[tuple[int, str], Callable] = {}


def _allocate_kernel_instance_id() -> int:
    global _kernel_instance_counter
    kernel_instance_id = _kernel_instance_counter
    _kernel_instance_counter += 1
    return kernel_instance_id


def _build_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
):
    kernel_instance_id = _allocate_kernel_instance_id()
    op_name = f"pallas::fused_moe_kernel_{kernel_instance_id}"

    wrapped_fn = _pallas_fused_moe_kernel(
        topk=topk,
        activation=activation,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=(),
        schema="(Tensor hidden_states, Tensor w1, Tensor w2, "
        "Tensor? w1_scale, Tensor? w2_scale, Tensor? w1_bias, "
        "Tensor? w2_bias, Tensor topk_weights, Tensor topk_ids) -> Tensor",
        device_types=["tpu"],
    )
    @pallas.custom_jax_kernel
    def fused_moe_kernel_impl(hidden_states, w1, w2, w1_scale, w2_scale,
                              w1_bias, w2_bias, topk_weights, topk_ids):
        return wrapped_fn(hidden_states, w1, w2, w1_scale, w2_scale, w1_bias,
                          w2_bias, topk_weights, topk_ids)

    def _fake_fused_moe(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        w1_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)
    cache_key = (topk, activation)
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
):
    cache_key = (topk, activation)
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        activation=activation,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    activation: str,
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing."""
    _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
    )


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    activation: str,
) -> torch.Tensor:
    """Fused MoE forward pass with precomputed routing."""
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
    )
    return fused_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    )
