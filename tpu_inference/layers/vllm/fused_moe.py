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

from typing import Optional

import torch
from torch_tpu._internal import pallas

from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def _pallas_fused_moe_kernel(
    topk: int,
    renormalize: bool,
    activation: str,
    use_ep: bool,
):
    """Build a pallas custom kernel wrapper around fused_moe_func."""
    mesh = get_vllm_model_wrapper_context().mesh

    def impl(hidden_states, w1, w2, w1_scale, w2_scale, w1_bias, w2_bias,
             gating_output):
        return fused_moe_func(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            mesh=mesh,
            use_ep=use_ep,
            activation=activation,
        )

    return impl


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str,
    use_ep: bool = False,
) -> torch.Tensor:
    """Fused MoE forward pass using fused_moe_func from fused_moe_gmm.py."""
    wrapper_impl = _pallas_fused_moe_kernel(
        topk=topk,
        renormalize=renormalize,
        activation=activation,
        use_ep=use_ep,
    )
    fused_moe = pallas.custom_jax_kernel(wrapper_impl)
    return fused_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        gating_output,
    )
