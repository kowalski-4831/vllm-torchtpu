# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Torch custom-op bridges for Kimi KDA and its short convolution."""

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.kimi_k3 import chunk_kda, ragged_kda
from vllm_torchtpu.kernels.kimi_k3.decode_kda import decode_kda


def kimi_short_conv_scan(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    start_seq: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Correctness-first Kimi short convolution, one token at a time.

    The shared Pallas ragged-convolution kernel can halt the TPU when many
    donated recurrent buffers are embedded in Kimi's compiled hybrid graph.
    Keep this model-local fallback until that kernel interaction is fixed.

    ``start_seq`` restricts the walk to the tokens of segments at or after that
    index; earlier tokens keep a zero output row and their convolution state is
    left untouched. That is how a caller hands the decode part of a batch to
    ``decode_kda``, which does its own convolution.

    The loop bounds are *device* values, so the trip count follows the batch:
    with every segment restricted away the loop body never runs. This is a
    ``fori_loop`` rather than a ``scan`` for exactly that reason -- a ``scan``
    has a static length and would walk the whole token bucket, masking as it
    went, which costs the same whether or not there is anything to do. At
    roughly 20us per token that is not a rounding error.
    """
    num_tokens = x.shape[0]
    kernel_size = conv_weight.shape[-1]
    weights = jnp.swapaxes(conv_weight[:, 0, :], 0, 1)
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    # Dummy AOT runs may describe more one-token requests than fit in the
    # current token bucket; normal runtime metadata instead repeats the true
    # token total in its padded request tail.
    total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
    sequence_ids = jnp.searchsorted(query_start_loc[1:],
                                    token_ids,
                                    side="right")
    sequence_ids = jnp.minimum(sequence_ids, state_indices.shape[0] - 1)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial_state = seq_lens > query_lens

    if start_seq is None:
        first_token = jnp.zeros((), jnp.int32)
    else:
        first_token = jnp.minimum(
            query_start_loc[jnp.asarray(start_seq, jnp.int32)], total_tokens)

    def step(token_id, carry):
        state, out = carry
        sequence_id = sequence_ids[token_id]
        state_idx = state_indices[sequence_id]
        old_state = state[state_idx]
        is_first = token_id == query_start_loc[sequence_id]
        old_state = jnp.where(is_first & ~has_initial_state[sequence_id],
                              jnp.zeros_like(old_state), old_state)
        window = jnp.concatenate((old_state, x[token_id][None, :]), axis=0)
        output = jnp.sum(window.astype(jnp.float32) *
                         weights.astype(jnp.float32),
                         axis=0).astype(x.dtype)
        new_state = window[-(kernel_size - 1):].astype(conv_state.dtype)
        # The loop bounds already exclude every token that must not be written,
        # so no per-token validity mask is needed here.
        return (state.at[state_idx].set(new_state),
                out.at[token_id].set(output))

    new_state, output = jax.lax.fori_loop(
        first_token,
        total_tokens,
        step,
        (conv_state, jnp.zeros_like(x)),
    )
    return output, new_state


def build_kimi_sconv_op(
    prefix: str,
    *,
    kernel_size: int,
    state_dim_first: bool,
):
    """Build the layer-specific Pallas short-convolution custom op."""

    def sconv_core(
        mixed_qkv: jax.Array,
        conv_state: jax.Array,
        q_weight: jax.Array,
        k_weight: jax.Array,
        v_weight: jax.Array,
        query_start_loc: jax.Array,
        state_indices: jax.Array,
        seq_lens: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        num_tokens, mixed_dim = mixed_qkv.shape
        if mixed_dim % 3:
            raise ValueError(
                "Kimi short-convolution input must contain Q, K, V")
        projection_size = mixed_dim // 3
        expected_weight_shape = (projection_size, 1, kernel_size)
        for name, weight in (("q", q_weight), ("k", k_weight), ("v",
                                                                v_weight)):
            if weight.shape != expected_weight_shape:
                raise ValueError(
                    f"{name}_weight must have shape {expected_weight_shape}, "
                    f"got {weight.shape}")
        num_sequences = state_indices.shape[0]
        if query_start_loc.shape != (num_sequences + 1, ):
            raise ValueError("query_start_loc and state_indices disagree")
        if seq_lens.shape != (num_sequences, ):
            raise ValueError("seq_lens and state_indices disagree")
        expected_state_tail = ((mixed_dim,
                                kernel_size - 1) if state_dim_first else
                               (kernel_size - 1, mixed_dim))
        if conv_state.shape[1:] != expected_state_tail:
            raise ValueError(
                f"conv_state must end in {expected_state_tail}, got "
                f"{conv_state.shape}")
        if (query_start_loc.dtype != jnp.int32
                or state_indices.dtype != jnp.int32
                or seq_lens.dtype != jnp.int32):
            raise TypeError("Short-convolution metadata tensors must be int32")

        state = (jnp.swapaxes(conv_state, 1, 2)
                 if state_dim_first else conv_state)
        weight = jnp.concatenate((q_weight, k_weight, v_weight), axis=0)
        output, new_state = kimi_short_conv_scan(
            mixed_qkv,
            state,
            weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        if state_dim_first:
            new_state = jnp.swapaxes(new_state, 1, 2)
        if output.shape != (num_tokens, mixed_dim):
            raise ValueError("Short-convolution output shape changed")
        return new_state, jax.nn.silu(output)

    op_name = f"pallas::kimi_sconv_{prefix.replace('.', '_')}"
    sconv_op = pallas.jax_op(op_name, sconv_core, donate_argnums=(1, ))

    def _fake_sconv(mixed_qkv, conv_state, *args, **kwargs):
        return torch.empty_like(conv_state), torch.empty_like(mixed_qkv)

    sconv_op.register_fake(_fake_sconv)

    def sconv_impl(
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        new_state, output = sconv_op(
            mixed_qkv,
            conv_state,
            q_weight,
            k_weight,
            v_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        conv_state.copy_(new_state)
        return output

    return sconv_impl


def build_kimi_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
):
    """Build the layer-specific Pallas KDA recurrence custom op."""
    wrapped = functools.partial(ragged_kda, lower_bound=lower_bound, eps=eps)
    op_name = f"pallas::kimi_kda_{prefix.replace('.', '_')}"
    kda_op = pallas.jax_op(op_name, wrapped, donate_argnums=(4, ))

    def _fake_kda(mixed_qkv, _raw_gate, _beta, _output_gate, recurrent_state,
                  *args, **kwargs):
        num_tokens = mixed_qkv.size(0)
        num_heads, head_dim = recurrent_state.shape[1:3]
        output = torch.empty((num_tokens, num_heads, head_dim),
                             dtype=mixed_qkv.dtype,
                             device=mixed_qkv.device)
        return output, torch.empty_like(recurrent_state)

    kda_op.register_fake(_fake_kda)

    def kda_impl(
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        recurrent_state: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        output, new_state = kda_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            recurrent_state,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        recurrent_state.copy_(new_state)
        return output

    return kda_impl


# ===========================================================================
# Pallas chunked KDA
# ===========================================================================


def _activate_beta(beta: jax.Array) -> jax.Array:
    """Sigmoid beta; the Pallas kernel expects it pre-activated."""
    return jax.nn.sigmoid(beta.astype(jnp.float32)).astype(beta.dtype)


def _gated_output_norm(
    output: jax.Array,  # [num_rows, H, D]
    output_gate: jax.Array,  # [num_rows, H * D] raw gate
    norm_weight: jax.Array,  # [D]
    eps: float,
    activation_dtype,
) -> jax.Array:
    """Per-head ``o_norm`` followed by the sigmoid output gate."""
    output = output.astype(activation_dtype).astype(jnp.float32)
    output *= jax.lax.rsqrt(
        jnp.mean(output * output, axis=-1, keepdims=True) + eps)
    output = (output *
              norm_weight.astype(jnp.float32)).astype(activation_dtype)
    return output * jax.nn.sigmoid(output_gate.reshape(output.shape))


def _check_kda_abi(
    mixed_qkv: jax.Array,
    raw_gate: jax.Array,
    beta: jax.Array,
    output_gate: jax.Array,
    recurrent_state: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
) -> tuple[int, int, int, int]:
    """Validate the shared KDA operand contract; return the shape tuple.

    Deliberately the same contract `ragged_kda` enforces, so swapping ops cannot
    silently change what a caller is allowed to pass.
    """
    num_tokens, mixed_dim = mixed_qkv.shape
    num_heads, head_dim, state_v_dim = recurrent_state.shape[1:]
    if state_v_dim != head_dim or mixed_dim != 3 * num_heads * head_dim:
        raise ValueError("Incompatible KDA activation and state shapes")
    if raw_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA gate shape")
    if beta.shape != (num_tokens, num_heads):
        raise ValueError("Incompatible KDA beta shape")
    if output_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA output-gate shape")
    num_sequences = state_indices.shape[0]
    if a_log.shape != (num_heads, ):
        raise ValueError("KDA A_log shape does not match the head count")
    if dt_bias.shape != (num_heads * head_dim, ):
        raise ValueError("KDA dt_bias shape does not match the projection")
    if norm_weight.shape != (head_dim, ):
        raise ValueError("KDA norm weight shape does not match the head size")
    if query_start_loc.shape != (num_sequences + 1, ):
        raise ValueError("KDA query starts and state indices disagree")
    if seq_lens.shape != (num_sequences, ):
        raise ValueError("KDA sequence lengths and state indices disagree")
    if (raw_gate.dtype != mixed_qkv.dtype or beta.dtype != mixed_qkv.dtype
            or output_gate.dtype != mixed_qkv.dtype):
        raise TypeError("KDA activations must have one common dtype")
    if recurrent_state.dtype != jnp.float32:
        raise TypeError("KDA recurrent state must be float32")
    if a_log.dtype != jnp.float32 or dt_bias.dtype != jnp.float32:
        raise TypeError("KDA A_log and dt_bias must be float32")
    if (query_start_loc.dtype != jnp.int32 or state_indices.dtype != jnp.int32
            or seq_lens.dtype != jnp.int32):
        raise TypeError("KDA metadata tensors must be int32")
    return num_tokens, num_sequences, num_heads, head_dim


def build_kimi_chunk_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
):
    """Build the chunked KDA custom op.

    Torch-level signature and semantics match ``build_kimi_kda_op``: it takes
    the post-convolution activations and returns the normalised, gated output.
    Sequence lengths are arbitrary -- a 1-token row is just a length-1 segment
    -- so this op is correct for prefill, for pure decode, and for mixed
    batches alike, which is what lets the layer dispatch it unconditionally.

    It is not, however, the *fastest* decode path: the kernel aligns every
    segment up to a full 64-token chunk, so a batch of N one-token rows costs N
    chunks of work. A fused recurrent decode kernel is a follow-on.
    """

    def chunk_core(
        mixed_qkv: jax.Array,
        raw_gate: jax.Array,
        beta: jax.Array,
        output_gate: jax.Array,
        recurrent_state: jax.Array,
        a_log: jax.Array,
        dt_bias: jax.Array,
        norm_weight: jax.Array,
        query_start_loc: jax.Array,
        state_indices: jax.Array,
        seq_lens: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        num_tokens, num_seqs, num_heads, head_dim = _check_kda_abi(
            mixed_qkv, raw_gate, beta, output_gate, recurrent_state, a_log,
            dt_bias, norm_weight, query_start_loc, state_indices, seq_lens)
        activation_dtype = mixed_qkv.dtype

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        has_initial_state = seq_lens > query_lens
        # The kernel takes 1-indexed segment IDs where upstream takes
        # cu_seqlens; 0 marks a padding token. Derived on device so the whole
        # thing stays inside one graph.
        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
        total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
        sequence_ids = jnp.searchsorted(query_start_loc[1:],
                                        token_ids,
                                        side="right")
        sequence_ids = jnp.minimum(sequence_ids, num_seqs - 1)
        segment_ids = jnp.where(token_ids < total_tokens, sequence_ids + 1, 0)

        qkv = mixed_qkv.reshape(num_tokens, 3, num_heads, head_dim)

        def head_major(x):  # [T, H, D] -> [H, 1, T, D]
            return jnp.transpose(x, (1, 0, 2))[:, None]

        initial_state = recurrent_state[state_indices]
        initial_state = jnp.where(has_initial_state[:, None, None, None],
                                  initial_state, 0.0)

        output, final_state = chunk_kda(
            head_major(qkv[:, 0]),
            head_major(qkv[:, 1]),
            head_major(qkv[:, 2]),
            head_major(raw_gate.reshape(num_tokens, num_heads, head_dim)),
            jnp.transpose(_activate_beta(beta), (1, 0))[:, None],  # [H, 1, T]
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            segment_ids=segment_ids[None],
            N_max=num_seqs,
            initial_state=initial_state[None],
            output_final_state=True,
        )
        # [H, 1, T, D] -> [T, H, D]
        output = jnp.transpose(output[:, 0], (1, 0, 2))

        # A request with no scheduled token this step has no final state worth
        # keeping -- the kernel hands back its (zeroed) initial state, which
        # would wipe a live slot. Route those writes to the null block instead.
        write_indices = jnp.where(query_lens > 0, state_indices, 0)
        new_state = recurrent_state.at[write_indices].set(
            final_state[0].astype(recurrent_state.dtype))
        return _gated_output_norm(output, output_gate, norm_weight, eps,
                                  activation_dtype), new_state

    op_name = f"pallas::kimi_chunk_kda_{prefix.replace('.', '_')}"
    chunk_op = pallas.jax_op(op_name, chunk_core, donate_argnums=(4, ))

    def _fake_chunk(mixed_qkv, _raw_gate, _beta, _output_gate, recurrent_state,
                    *args, **kwargs):
        num_heads, head_dim = recurrent_state.shape[1:3]
        output = torch.empty((mixed_qkv.size(0), num_heads, head_dim),
                             dtype=mixed_qkv.dtype,
                             device=mixed_qkv.device)
        return output, torch.empty_like(recurrent_state)

    chunk_op.register_fake(_fake_chunk)

    def chunk_impl(
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        recurrent_state: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        output, new_state = chunk_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            recurrent_state,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        recurrent_state.copy_(new_state)
        return output

    return chunk_impl


# ===========================================================================
# Device-dispatched KDA: the fused decode kernel plus the chunked kernel
# ===========================================================================
# The batch is ordered [decode][prefill/mixed] and `request_distribution[0]` is
# the boundary. Both recurrences run every step, each restricted to its own side
# of it, and their outputs are selected per token.
#
# Nothing branches on the host. vLLM's `TorchCompileWithNoGuardsWrapper` drops
# every Dynamo guard and traces the model once, and the dummy run driving that
# trace is decode-only, so a host-side branch on batch composition freezes the
# decode path into the graph for every later step. That failure is silent -- it
# produced garbage output while every unit test passed -- so the boundary stays a
# device value and both kernels are emitted unconditionally.
#
# `decode_kda` fuses the short convolution into the recurrence, so it owns the
# convolution for its own rows; the scan covers the rest. The two touch disjoint
# cache slots, and are chained so each sees the other's writes.


def _null_out_non_decode(state_indices: jax.Array,
                         decode_end: jax.Array) -> jax.Array:
    """Point every non-decode request at the reserved null slot.

    Slot 0 is the null block, and ``decode_kda`` already reads it as "emit
    nothing, write nothing, leave the slot alone". Bounding the decode segment
    that way needs no new kernel argument and no kernel change.
    """
    sequence_ids = jnp.arange(state_indices.shape[0], dtype=jnp.int32)
    return jnp.where(sequence_ids < decode_end, state_indices, 0)


def _match_rows(x: jax.Array, num_rows: int) -> jax.Array:
    """Re-length a leading axis by truncation or zero padding.

    ``decode_kda`` is indexed by request while the activations are indexed by
    token. Over the decode segment the two coincide -- each decode request holds
    exactly one token, so request *s* is token row *s* -- and this only
    reconciles the two independently sized static buckets.
    """
    have = x.shape[0]
    if have == num_rows:
        return x
    if have > num_rows:
        return x[:num_rows]
    return jnp.pad(x, [(0, num_rows - have)] + [(0, 0)] * (x.ndim - 1))


def build_kimi_dispatched_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
    state_dim_first: bool,
):
    """Build the KDA op that sends decode rows to the fused decode kernel.

    Takes the *pre-convolution* projection output and owns both caches, because
    ``decode_kda`` fuses the convolution into the recurrence and so cannot be fed
    by a separate convolution op the way the chunked kernel is.
    """

    def dispatched_core(
        mixed_qkv: jax.Array,  # [T, 3 * H * D], pre-convolution
        raw_gate: jax.Array,  # [T, H * D]
        beta: jax.Array,  # [T, H]
        output_gate: jax.Array,  # [T, H * D]
        conv_state: jax.Array,  # paged short-convolution cache
        recurrent_state: jax.Array,  # [num_slots, H, K, V]
        q_weight: jax.Array,  # [H * D, 1, kernel_size]
        k_weight: jax.Array,
        v_weight: jax.Array,
        a_log: jax.Array,  # [H]
        dt_bias: jax.Array,  # [H * D]
        norm_weight: jax.Array,  # [D]
        query_start_loc: jax.Array,  # [N + 1]
        state_indices: jax.Array,  # [N]
        seq_lens: jax.Array,  # [N]
        distribution: jax.Array,  # [3] int32: [decode_end, _, mixed_end]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        num_tokens, num_seqs, num_heads, head_dim = _check_kda_abi(
            mixed_qkv, raw_gate, beta, output_gate, recurrent_state, a_log,
            dt_bias, norm_weight, query_start_loc, state_indices, seq_lens)
        activation_dtype = mixed_qkv.dtype
        decode_end = distribution[0].astype(jnp.int32)

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        has_initial_state = seq_lens > query_lens
        conv_in = (jnp.swapaxes(conv_state, 1, 2)
                   if state_dim_first else conv_state)
        # Both convolution paths want one weight over the concatenated q/k/v
        # projection, the same way `sconv_core` builds it.
        conv_weight = jnp.concatenate((q_weight, k_weight, v_weight), axis=0)

        # --- decode segment: fused convolution + recurrence ---------------
        decode_out, conv_after_decode, pool_after_decode = decode_kda(
            _match_rows(mixed_qkv, num_seqs),
            _match_rows(raw_gate, num_seqs).reshape(num_seqs, num_heads,
                                                    head_dim),
            _activate_beta(_match_rows(beta, num_seqs)),
            conv_in,
            conv_weight,
            recurrent_state,
            a_log,
            dt_bias.reshape(num_heads, head_dim),
            _null_out_non_decode(state_indices, decode_end),
            has_initial_state,
            lower_bound=lower_bound,
            # The pool is [slots, H, K, V], which is the layout the recurrence
            # indexes as state[k, v]; no in-kernel swap.
            state_transposed=False,
        )

        # --- prefill/mixed segment: convolution, then the chunked kernel ---
        conv_out, conv_after_prefill = kimi_short_conv_scan(
            mixed_qkv,
            conv_after_decode,
            conv_weight,
            query_start_loc,
            state_indices,
            seq_lens,
            start_seq=decode_end,
        )
        conv_out = jax.nn.silu(conv_out)

        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
        total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
        sequence_ids = jnp.minimum(
            jnp.searchsorted(query_start_loc[1:], token_ids, side="right"),
            num_seqs - 1)
        segment_ids = jnp.where(token_ids < total_tokens, sequence_ids + 1, 0)

        qkv = conv_out.reshape(num_tokens, 3, num_heads, head_dim)

        def head_major(x):  # [T, H, D] -> [H, 1, T, D]
            return jnp.transpose(x, (1, 0, 2))[:, None]

        initial_state = jnp.where(has_initial_state[:, None, None, None],
                                  pool_after_decode[state_indices], 0.0)

        chunk_out, final_state = chunk_kda(
            head_major(qkv[:, 0]),
            head_major(qkv[:, 1]),
            head_major(qkv[:, 2]),
            head_major(raw_gate.reshape(num_tokens, num_heads, head_dim)),
            jnp.transpose(_activate_beta(beta), (1, 0))[:, None],  # [H, 1, T]
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            segment_ids=segment_ids[None],
            N_max=num_seqs,
            initial_state=initial_state[None],
            output_final_state=True,
            start_seq=decode_end,
        )
        chunk_out = jnp.transpose(chunk_out[:, 0], (1, 0, 2))  # -> [T, H, D]

        # Rows with no scheduled token have no state worth keeping; their writes
        # go to the reserved null block, slot 0.
        #
        # Decode slots are excluded belt-and-braces. `start_seq` above already
        # makes the chunked kernel pass those segments through untouched, and the
        # state it passes through was read *after* `decode_kda` ran, so writing
        # it back would currently be a no-op. The exclusion is what makes that
        # independent of the read order rather than contingent on it -- and it is
        # the guard if a request ever reaches the decode segment without carried
        # state, where the passthrough would be zero rather than the live state.
        sequence_ids_all = jnp.arange(num_seqs, dtype=jnp.int32)
        keep = (query_lens > 0) & (sequence_ids_all >= decode_end)
        write_indices = jnp.where(keep, state_indices, 0)
        new_pool = pool_after_decode.at[write_indices].set(
            final_state[0].astype(recurrent_state.dtype))

        # --- select per token ---------------------------------------------
        # Each side left the other's rows zero, so this is a select rather than
        # a sum only to keep the intent legible.
        output = jnp.where(
            (token_ids < decode_end)[:, None, None],
            _match_rows(decode_out, num_tokens).astype(chunk_out.dtype),
            chunk_out,
        )
        output = _gated_output_norm(output, output_gate, norm_weight, eps,
                                    activation_dtype)

        conv_out_state = (jnp.swapaxes(conv_after_prefill, 1, 2)
                          if state_dim_first else conv_after_prefill)
        return output, conv_out_state, new_pool

    op_name = f"pallas::kimi_dispatched_kda_{prefix.replace('.', '_')}"
    dispatched_op = pallas.jax_op(op_name,
                                  dispatched_core,
                                  donate_argnums=(4, 5))

    def _fake_dispatched(mixed_qkv, _raw_gate, _beta, _output_gate, conv_state,
                         recurrent_state, *args, **kwargs):
        num_heads, head_dim = recurrent_state.shape[1:3]
        output = torch.empty((mixed_qkv.size(0), num_heads, head_dim),
                             dtype=mixed_qkv.dtype,
                             device=mixed_qkv.device)
        return (output, torch.empty_like(conv_state),
                torch.empty_like(recurrent_state))

    dispatched_op.register_fake(_fake_dispatched)

    def dispatched_impl(
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        distribution: torch.Tensor,
    ) -> torch.Tensor:
        output, new_conv_state, new_pool = dispatched_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            conv_state,
            recurrent_state,
            q_weight,
            k_weight,
            v_weight,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
            distribution,
        )
        conv_state.copy_(new_conv_state)
        recurrent_state.copy_(new_pool)
        return output

    return dispatched_impl
