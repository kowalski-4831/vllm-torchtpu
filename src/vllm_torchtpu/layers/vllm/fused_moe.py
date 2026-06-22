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

import functools
from typing import Callable, Optional

import torch
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func

_kernel_instance_counter = 0
_fused_moe_kernel_cache: dict[tuple[int, str, Optional[int]], Callable] = {}


def _allocate_kernel_instance_id() -> int:
    global _kernel_instance_counter
    kernel_instance_id = _kernel_instance_counter
    _kernel_instance_counter += 1
    return kernel_instance_id


def _build_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    experts_start: Optional[int],
):
    kernel_instance_id = _allocate_kernel_instance_id()
    op_name = f"pallas::fused_moe_kernel_{kernel_instance_id}"

    # TODO: once the pod's vLLM includes --moe-backend (added upstream in
    # vLLM PR #33807, 2026-03-01; absent in the 0.19.0 we run), register the
    # SparseCore vs plain variants through that selector + the MoE oracle
    # instead of this env flag -- the MoE analog of how #191 moved batched
    # RPA onto --attention-backend CUSTOM. Note MoEBackend has no CUSTOM slot
    # yet, so that also needs an upstream OOT-backend hook.
    use_sparse_core = envs.USE_MOE_SPARSE_CORE

    wrapped_fn = functools.partial(
        fused_moe_func,
        experts_start=experts_start,
        topk=topk,
        activation=activation,
        use_ep=experts_start is not None,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=envs.ONEHOT_MOE_PERMUTE_THRESHOLD)

    fused_moe_kernel_impl = pallas.jax_op(op_name, wrapped_fn)

    # We must overwrite the default fake implementation as vLLM uses dynamic
    # dimensions for the hidden_states.
    def _fake_fused_moe(hidden_states, *args, **kwargs):
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)

    cache_key = (topk, activation, experts_start)
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    experts_start: Optional[int],
):
    cache_key = (topk, activation, experts_start)
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        experts_start=experts_start,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    activation: str,
    experts_start: Optional[int],
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing."""
    _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        experts_start=experts_start,
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
    experts_start: Optional[int],
    topk: int,
    activation: str,
) -> torch.Tensor:
    """Fused MoE forward pass with precomputed routing.

    ``experts_start`` is the first global expert id owned by this shard under
    linear EP placement (or ``None`` for non-EP). It is a Python int derived at
    load time from ``ep_rank``, ``ep_size``, and ``global_num_experts``, baked
    into the JAX kernel closure as a compile-time constant -- so the global->
    local remap is a literal subtract fused into the routing loop. When
    present, the sparse-core parity path dispatches ragged gather/gather-reduce.
    """
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        experts_start=experts_start,
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
