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

import functools

import jax
from jax import numpy as jnp

from tpu_inference.kernels.megablox.gmm import gmm
from tpu_inference.kernels.megablox.gmm_v2 import (gmm_v2,
                                                   is_supported_by_gmm_v2)


def apply_act_fn(activation: str, x1: jax.Array, x2: jax.Array) -> jax.Array:
    match activation:
        case "silu":
            return jax.nn.silu(x1) * x2
        case "swigluoai":
            return _swigluoai(x1, x2)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {activation} activation function")


def _swigluoai(x1: jax.Array,
               x2: jax.Array,
               alpha=1.702,
               limit=7.0) -> jax.Array:
    x1 = jnp.clip(x1, a_max=limit)
    x2 = jnp.clip(x2, a_min=-limit, a_max=limit)

    gated_activation = x1 * jax.nn.sigmoid(alpha * x1)

    return gated_activation * (x2 + 1)


def gmm_wrapper(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset):
    if is_supported_by_gmm_v2(lhs, rhs, rhs_scale):
        gmm_res = gmm_v2(
            lhs=lhs,
            rhs=rhs,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_sizes=group_sizes,
            group_offset=group_offset[0],
        )
    else:
        gmm_res = gmm(
            lhs=lhs,
            rhs=rhs,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_sizes=group_sizes,
            preferred_element_type=lhs.dtype,
            tiling=None,
            group_offset=group_offset[0],
        )

    return gmm_res


def prepare_routed_gmm_inputs(
    hidden_states_local: jax.Array,
    topk_indices_local: jax.Array,
    topk_weights_local: jax.Array,
    *,
    local_num_experts: int,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare the local-only routing layout using already-local expert ids."""
    num_tokens_local = hidden_states_local.shape[0]
    topk_indices_flat = topk_indices_local.flatten()
    topk_weights_flat = topk_weights_local.flatten()
    token_indices_flat = jnp.arange(num_tokens_local,
                                    dtype=jnp.int32).repeat(topk)

    valid_mask = topk_indices_flat >= 0
    sort_keys = jnp.where(valid_mask, topk_indices_flat, local_num_experts)
    sorted_indices = jnp.argsort(sort_keys)

    token_indices_sorted = token_indices_flat[sorted_indices]
    topk_weights_sorted = topk_weights_flat[sorted_indices]
    valid_mask_sorted = valid_mask[sorted_indices]

    x = hidden_states_local[token_indices_sorted]
    counted_experts = jnp.where(valid_mask, topk_indices_flat, 0)
    group_sizes_local = (
        jax.nn.one_hot(counted_experts, local_num_experts, dtype=jnp.int32) *
        valid_mask[:, None].astype(jnp.int32))
    group_sizes_local = group_sizes_local.sum(axis=0)
    return (x, group_sizes_local, token_indices_sorted, topk_weights_sorted,
            valid_mask_sorted)


def moe_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    token_indices_sorted: jax.Array,
    topk_weights_sorted: jax.Array,
    valid_mask_sorted: jax.Array,
    *,
    activation: str,
    num_tokens: int,
) -> jax.Array:
    """Run grouped GEMM for routed tokens and scatter-add back to tokens."""
    group_offset = jnp.array([0], dtype=jnp.int32)

    gmm1_res_gate_up = gmm_wrapper(x, w1, w1_scale, w1_bias, group_sizes,
                                   group_offset)
    gmm1_res_gate, gmm1_res_up = jnp.split(gmm1_res_gate_up, 2, -1)
    gmm1_res = apply_act_fn(activation, gmm1_res_gate, gmm1_res_up)

    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset)

    routed_hidden = gmm2_res * jnp.expand_dims(topk_weights_sorted, axis=-1)
    routed_hidden = jnp.where(valid_mask_sorted[:, None], routed_hidden, 0)
    token_hidden = jnp.zeros((num_tokens, gmm2_res.shape[-1]),
                             dtype=gmm2_res.dtype)
    return token_hidden.at[token_indices_sorted].add(routed_hidden)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "activation",
    ),
)
def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    topk: int,
    activation: str,
) -> jax.Array:
    """Run MoE with precomputed expert ids and weights."""
    num_tokens, hidden_size = hidden_states.shape
    _, padded_hidden_size, _ = w1.shape

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    x, group_sizes, token_indices_sorted, topk_weights_sorted, valid_mask_sorted = (
        prepare_routed_gmm_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            local_num_experts=w1.shape[0],
            topk=topk,
        ))
    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))
    x = moe_gmm(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        token_indices_sorted,
        topk_weights_sorted,
        valid_mask_sorted,
        activation=activation,
        num_tokens=num_tokens,
    )
    return x[:num_tokens, :hidden_size]
