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


_kernel_instance_counter = 0
_fused_moe_kernel_cache: dict[tuple[int, int, bool, str, bool], Callable] = {}


def _allocate_kernel_instance_id() -> int:
    global _kernel_instance_counter
    kernel_instance_id = _kernel_instance_counter
    _kernel_instance_counter += 1
    return kernel_instance_id


def _build_fused_moe_custom_op(
    *,
    topk: int,
    renormalize: bool,
    activation: str,
    use_ep: bool,
):
    mesh = get_vllm_model_wrapper_context().mesh
    kernel_instance_id = _allocate_kernel_instance_id()
    op_name = f"pallas::fused_moe_kernel_{kernel_instance_id}"

    wrapped_fn = _pallas_fused_moe_kernel(
        topk=topk,
        renormalize=renormalize,
        activation=activation,
        use_ep=use_ep,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=(),
        schema="(Tensor hidden_states, Tensor w1, Tensor w2, "
        "Tensor? w1_scale, Tensor? w2_scale, Tensor? w1_bias, "
        "Tensor? w2_bias, Tensor gating_output) -> Tensor",
        device_types=["tpu"],
    )
    @pallas.custom_jax_kernel
    def fused_moe_kernel_impl(hidden_states, w1, w2, w1_scale, w2_scale,
                              w1_bias, w2_bias, gating_output):
        return wrapped_fn(hidden_states, w1, w2, w1_scale, w2_scale, w1_bias,
                          w2_bias, gating_output)

    def _fake_fused_moe(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        w1_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
        gating_output: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)
    cache_key = (id(mesh), topk, renormalize, activation, use_ep)
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    renormalize: bool,
    activation: str,
    use_ep: bool,
):
    mesh = get_vllm_model_wrapper_context().mesh
    cache_key = (id(mesh), topk, renormalize, activation, use_ep)
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        renormalize=renormalize,
        activation=activation,
        use_ep=use_ep,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    renormalize: bool,
    activation: str,
    use_ep: bool,
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing."""
    _get_fused_moe_custom_op(
        topk=topk,
        renormalize=renormalize,
        activation=activation,
        use_ep=use_ep,
    )


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
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        renormalize=renormalize,
        activation=activation,
        use_ep=use_ep,
    )
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
