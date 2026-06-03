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

import tpu_inference.envs as envs
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.kernels.sparse_core import gather_reduce as gather_reduce_sc
from tpu_inference.kernels.sparse_core.ragged_gather import ragged_gather
from tpu_inference.kernels.sparse_core.ragged_scatter import ragged_scatter


def gmm_wrapper(lhs,
                rhs,
                rhs_scale,
                rhs_bias,
                group_sizes,
                group_offset,
                zero_initialize=False,
                fuse_act=None,
                preferred_element_type=None):
    return gmm_v2(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        group_offset=group_offset[0],
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        preferred_element_type=preferred_element_type,
    )


def prepare_routed_gmm_inputs(
    hidden_states_local: jax.Array,
    topk_indices_local: jax.Array,
    topk_weights_local: jax.Array,
    *,
    local_num_experts: int,
    topk: int,
    use_ep: bool,
    use_sparse_core: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare the local-only routing layout using already-local expert ids.

    For non-EP runs `topk_indices_local` is always non-negative, so the
    valid-mask path computed below is a no-op (jnp.where on an all-True
    mask). For EP, the caller applies the expert-map remap upstream
    (inside this JIT) and marks non-local entries with a negative id which
    the mask filters out. Always running the masked branch keeps a single
    code path.
    """
    num_tokens_local = hidden_states_local.shape[0]
    topk_indices_flat = topk_indices_local.flatten()
    topk_weights_flat = topk_weights_local.flatten()
    token_indices_flat = jnp.arange(num_tokens_local,
                                    dtype=jnp.int32).repeat(topk)

    valid_mask = topk_indices_flat >= 0
    sort_keys = jnp.where(valid_mask, topk_indices_flat, local_num_experts)
    sorted_indices = jnp.argsort(sort_keys)
    argsort_revert_indices = jnp.argsort(sorted_indices)

    token_indices_sorted = token_indices_flat[sorted_indices]
    topk_indices_for_count = jnp.where(valid_mask, topk_indices_flat, 0)
    group_sizes_local = (jax.nn.one_hot(
        topk_indices_for_count, local_num_experts, dtype=jnp.int32) *
                         valid_mask[:, None].astype(jnp.int32))
    group_sizes_local = group_sizes_local.sum(axis=0)

    if use_ep and use_sparse_core:
        # Match uLLM/reference EP routing: materialize the valid local
        # expert prefix through SparseCore ragged_gather. Invalid
        # non-local rows are sorted after the valid prefix.
        valid_count = group_sizes_local.sum(dtype=jnp.int32)
        x = ragged_gather(
            hidden_states_local,
            token_indices_sorted,
            jnp.array([0], dtype=jnp.int32),
            valid_count[None],
        )
    else:
        x = hidden_states_local[token_indices_sorted]
    return (x, group_sizes_local, argsort_revert_indices, topk_weights_flat,
            valid_mask)


def moe_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    argsort_revert_indices: jax.Array,
    topk_weights_flat: jax.Array,
    valid_mask_flat: jax.Array,
    *,
    activation: str,
    num_tokens: int,
    topk: int,
    use_ep: bool,
    use_sparse_core: bool,
) -> jax.Array:
    """Run grouped GEMM for routed tokens and reduce back to tokens.

    The valid-mask gating below is a no-op for non-EP runs (mask is all
    True) and keeps the single code path uniform for both EP and non-EP.
    EP uses uLLM/reference SparseCore ragged gather/scatter for token movement.
    """
    group_offset = jnp.array([0], dtype=jnp.int32)

    gmm1_res = gmm_wrapper(
        x,
        w1,
        w1_scale,
        w1_bias,
        group_sizes,
        group_offset,
        zero_initialize=False,
        fuse_act=activation,
    )
    gmm1_res = gmm1_res[:, :w2.shape[1]]

    topk_weights = topk_weights_flat.reshape((num_tokens, topk))
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))

    if (use_sparse_core and topk_weights_flat.size % 128 == 0
            and gather_reduce_sc.is_supported_by_sc_gather_reduce(
                gmm1_res.shape[0], envs.SC_KERNEL_THRESHOLD)):
        gmm2_res = gmm_wrapper(gmm1_res,
                               w2,
                               w2_scale,
                               w2_bias,
                               group_sizes,
                               group_offset,
                               zero_initialize=True,
                               preferred_element_type=jnp.float32)
        topk_weights_sc = jnp.where(valid_mask_flat, topk_weights_flat,
                                    0).astype(x.dtype).reshape(-1, 128)
        return gather_reduce_sc.sc_gather_reduce(
            op=gmm2_res,
            idx=argsort_revert_indices,
            reduce_group_size=topk,
            topk_weights=topk_weights_sc,
            col_chunk_size=envs.SC_KERNEL_COL_CHUNK_SIZE,
        ).astype(x.dtype)

    gmm2_res = gmm_wrapper(gmm1_res,
                           w2,
                           w2_scale,
                           w2_bias,
                           group_sizes,
                           group_offset,
                           zero_initialize=False,
                           preferred_element_type=x.dtype)

    if use_ep and use_sparse_core:
        valid_count = group_sizes.sum(dtype=jnp.int32)
        token_hidden = ragged_scatter(
            gmm2_res,
            argsort_revert_indices,
            jnp.array([0], dtype=jnp.int32),
            valid_count[None],
        )
    else:
        token_hidden = gmm2_res[argsort_revert_indices]

    token_topk_hidden = token_hidden.reshape(
        (num_tokens, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_topk_hidden = jnp.where(valid_mask[:, :, None], token_topk_hidden, 0)
    # FP32 top-k weights can promote BF16 outputs; keep the custom-op dtype.
    return token_topk_hidden.sum(axis=1).astype(x.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "activation",
        "use_ep",
        "use_sparse_core",
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
    experts_start: int | None = None,
    topk: int = 1,
    activation: str = "silu",
    use_ep: bool = False,
    use_sparse_core: bool = True,
) -> jax.Array:
    """Run MoE with precomputed expert ids and weights.

    For linear EP placement, ``experts_start`` is the first global expert id
    owned by this shard, bound as a Python int (compile-time constant) by the
    torch bridge. The kernel remaps global ids to local ids with an elementwise
    subtract and masks non-local experts. ``use_ep`` is a static flag enabling
    EP routing; ``use_sparse_core`` is a static flag selecting the #193
    SparseCore ragged gather/scatter (vs the pre-#193 plain-JAX path).
    """
    num_tokens, hidden_size = hidden_states.shape
    _, padded_hidden_size, _ = w1.shape

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    # Pin topk_weights to hidden_states.dtype so the EP mask, GMM
    # reduce, and `where` propagate in BF16 — some custom_routing_fn
    # callers return FP32 weights and would otherwise promote downstream.
    topk_weights = topk_weights.astype(hidden_states.dtype)

    if experts_start is not None:
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights,
                                 jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

    (x, group_sizes, argsort_revert_indices, topk_weights_flat,
     valid_mask) = prepare_routed_gmm_inputs(
         hidden_states,
         topk_ids,
         topk_weights,
         local_num_experts=w1.shape[0],
         topk=topk,
         use_ep=use_ep,
         use_sparse_core=use_sparse_core,
     )
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
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        activation=activation,
        num_tokens=num_tokens,
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
    )
    return x[:num_tokens, :hidden_size]
