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
"""Tests for the local-routing fused MoE GMM wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.megablox.gmm_v2 import apply_act_fn
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _reference_fused_moe(hidden_states, w1, w2, w1_bias, w2_bias, topk_weights,
                         topk_ids, activation):
    num_tokens, hidden_size = hidden_states.shape
    padded_hidden_size = w1.shape[1]
    intermediate_size = w2.shape[1]

    hidden_states = jnp.pad(hidden_states,
                            ((0, 0), (0, padded_hidden_size - hidden_size)))
    out = jnp.zeros((num_tokens, padded_hidden_size), dtype=jnp.float32)

    for token_id in range(num_tokens):
        token_out = jnp.zeros((padded_hidden_size, ), dtype=jnp.float32)
        for expert_slot in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_id, expert_slot])
            if expert_id < 0:
                continue

            gate_up = jnp.matmul(hidden_states[token_id].astype(jnp.float32),
                                 w1[expert_id].astype(jnp.float32))
            if w1_bias is not None:
                gate_up = gate_up + w1_bias[expert_id, 0].astype(jnp.float32)

            activated = apply_act_fn(gate_up[None], intermediate_size,
                                     activation)[0]
            proj = jnp.matmul(activated.astype(jnp.float32),
                              w2[expert_id].astype(jnp.float32))
            if w2_bias is not None:
                proj = proj + w2_bias[expert_id, 0].astype(jnp.float32)

            token_out = token_out + topk_weights[token_id, expert_slot].astype(
                jnp.float32) * proj
        out = out.at[token_id].set(token_out)

    return out[:, :hidden_size].astype(hidden_states.dtype)


def test_fused_moe_local_routing_matches_reference():
    _require_tpu()

    # Match the local EP shard contract used by Qwen3-Coder-30B-A3B:
    # hidden_size=2048, moe_intermediate_size=768, hidden_act=silu,
    # num_experts_per_tok=8. This wrapper consumes already-local expert ids, so
    # use the per-shard expert count (128 global experts / 8 EP shards = 16).
    num_tokens = 4
    hidden_size = 2048
    padded_hidden_size = 2048
    intermediate_size = 768
    num_experts = 16
    topk = 8
    key = jax.random.key(0)
    hidden_key, w1_key, w2_key, w1_bias_key, w2_bias_key = jax.random.split(
        key, 5)

    hidden_states = (jax.random.normal(hidden_key, (num_tokens, hidden_size),
                                       dtype=jnp.float32) / 10).astype(
                                           jnp.bfloat16)
    w1 = (jax.random.normal(
        w1_key, (num_experts, padded_hidden_size, intermediate_size * 2),
        dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(w2_key,
                          (num_experts, intermediate_size, padded_hidden_size),
                          dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w1_bias = (jax.random.normal(w1_bias_key,
                                 (num_experts, 1, intermediate_size * 2),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2_bias = (jax.random.normal(w2_bias_key,
                                 (num_experts, 1, padded_hidden_size),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)

    topk_weights = jnp.array(
        [
            [0.30, 0.18, 0.14, 0.12, 0.10, 0.07, 0.05, 0.04],
            [0.22, 0.18, 0.16, 0.14, 0.11, 0.08, 0.06, 0.05],
            [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
            [0.28, 0.19, 0.15, 0.11, 0.09, 0.07, 0.06, 0.05],
        ],
        dtype=jnp.float32,
    )
    topk_ids = jnp.array(
        [
            [0, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 15],
            [15, 13, 11, 9, 7, 5, 3, 1],
            [14, 12, 10, 8, 6, 4, 2, 0],
        ],
        dtype=jnp.int32,
    )

    expected = _reference_fused_moe(hidden_states, w1, w2, w1_bias, w2_bias,
                                    topk_weights, topk_ids, "silu")
    actual = fused_moe_func(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=topk,
        activation="silu",
    )

    assert actual.dtype == hidden_states.dtype
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=1e-1,
                               rtol=1e-1)


def _make_ep_inputs(num_local_experts, seed=0):
    """Synthesize one local shard's weights + per-token local topk_ids.

    Matches the megablox-friendly shapes used by
    test_fused_moe_local_routing_matches_reference so the kernel does not
    return NaN on tile-misaligned inputs.
    """
    num_tokens = 4
    hidden_size = 2048
    intermediate_size = 768
    topk = 8
    key = jax.random.key(seed)
    hk, w1k, w2k, b1k, b2k = jax.random.split(key, 5)
    hidden_states = (
        jax.random.normal(hk, (num_tokens, hidden_size), dtype=jnp.float32) /
        10).astype(jnp.bfloat16)
    w1 = (jax.random.normal(
        w1k, (num_local_experts, hidden_size, intermediate_size * 2),
        dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(w2k,
                          (num_local_experts, intermediate_size, hidden_size),
                          dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w1_bias = (jax.random.normal(b1k,
                                 (num_local_experts, 1, intermediate_size * 2),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2_bias = (jax.random.normal(b2k, (num_local_experts, 1, hidden_size),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    topk_weights = jnp.array(
        [
            [0.30, 0.18, 0.14, 0.12, 0.10, 0.07, 0.05, 0.04],
            [0.22, 0.18, 0.16, 0.14, 0.11, 0.08, 0.06, 0.05],
            [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
            [0.28, 0.19, 0.15, 0.11, 0.09, 0.07, 0.06, 0.05],
        ],
        dtype=jnp.float32,
    )
    # Local ids stay in [0, num_local_experts); distinct per slot to exercise
    # the full local shard.
    base = jnp.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=jnp.int32)
    local_ids = jnp.stack([
        base,
        base + 1,
        (base + 1)[::-1],
        base[::-1],
    ]) % num_local_experts
    return (hidden_states, w1, w2, w1_bias, w2_bias, topk_weights, local_ids,
            topk)


@pytest.mark.parametrize(
    "num_local_experts,experts_start",
    [
        (16, 0),  # rank 0: experts_start == 0 must behave like the legacy path
        (16, 32),  # rank 2 of 8-way EP (Qwen3-30B shape: 128 / 8 = 16 local)
        (20,
         60),  # rank 3 of 8-way EP (Qwen3-480B-FP8 shape: 160 / 8 = 20 local)
    ])
def test_fused_moe_global_id_remap_matches_local(num_local_experts,
                                                 experts_start):
    """Global ids + experts_start must produce the same output as local ids."""
    _require_tpu()
    (hidden_states, w1, w2, w1_bias, w2_bias, topk_weights, local_ids,
     topk) = _make_ep_inputs(num_local_experts)
    global_ids = local_ids + experts_start
    expected = fused_moe_func(hidden_states=hidden_states,
                              w1=w1,
                              w2=w2,
                              w1_scale=None,
                              w2_scale=None,
                              w1_bias=w1_bias,
                              w2_bias=w2_bias,
                              topk_weights=topk_weights,
                              topk_ids=local_ids,
                              topk=topk,
                              activation="silu")
    actual = fused_moe_func(hidden_states=hidden_states,
                            w1=w1,
                            w2=w2,
                            w1_scale=None,
                            w2_scale=None,
                            w1_bias=w1_bias,
                            w2_bias=w2_bias,
                            topk_weights=topk_weights,
                            topk_ids=global_ids,
                            experts_start=experts_start,
                            topk=topk,
                            activation="silu")
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=1e-1,
                               rtol=1e-1)


def test_fused_moe_global_id_remap_masks_out_of_range():
    """Non-local global ids must contribute zero (range check + jnp.where)."""
    _require_tpu()
    num_local_experts, experts_start = 16, 32
    (hidden_states, w1, w2, w1_bias, w2_bias, topk_weights, local_ids,
     topk) = _make_ep_inputs(num_local_experts)
    global_ids = local_ids + experts_start
    # Token 0 slot 0: below shard (-> masked). Token 1 slot 1: above shard.
    global_ids = global_ids.at[0, 0].set(experts_start - 1)
    global_ids = global_ids.at[1, 1].set(experts_start + num_local_experts)
    # Reference: same call but with the offending weights pre-zeroed.
    masked_weights = topk_weights.at[0, 0].set(0.0).at[1, 1].set(0.0)
    expected = fused_moe_func(hidden_states=hidden_states,
                              w1=w1,
                              w2=w2,
                              w1_scale=None,
                              w2_scale=None,
                              w1_bias=w1_bias,
                              w2_bias=w2_bias,
                              topk_weights=masked_weights,
                              topk_ids=local_ids,
                              topk=topk,
                              activation="silu")
    actual = fused_moe_func(hidden_states=hidden_states,
                            w1=w1,
                            w2=w2,
                            w1_scale=None,
                            w2_scale=None,
                            w1_bias=w1_bias,
                            w2_bias=w2_bias,
                            topk_weights=topk_weights,
                            topk_ids=global_ids,
                            experts_start=experts_start,
                            topk=topk,
                            activation="silu")
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=1e-1,
                               rtol=1e-1)
