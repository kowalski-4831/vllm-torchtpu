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
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.compressor import (CompressorStateCache,
                                                DeepseekCompressor)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_torchtpu.kernels.deepseek_v4.compressor import (
    compressor_forward, compressor_forward_indexer)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

_compressor_op_cache = {}

# The mesh has no batch axis today, so every batch-axis spec resolves to None.
# TODO(patemotter): name it here once the DP-attention PR adds the axis.
BATCH_AXIS = None


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _compressor_jax(
    kv_score: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    request_distribution: jax.Array,
    query_start_loc: jax.Array,
    seq_lens: jax.Array,
    state_input_positions: jax.Array,
    state_block_tables: jax.Array,
    k_input_positions: jax.Array,
    k_block_tables: jax.Array,
    cache: jax.Array,
    *,
    state_block_size: int,
    head_dim: int,
    state_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> jax.Array:

    forward_fn = (compressor_forward
                  if head_dim == 512 else compressor_forward_indexer)

    def _compressor_local(kv_score, ape, norm_weight, cos_sin_cache, positions,
                          request_distribution, query_start_loc, seq_lens,
                          state_input_positions, state_block_tables,
                          k_input_positions, k_block_tables, cache):
        num_valid_reqs = request_distribution[2]
        num_tokens = kv_score.shape[0]
        num_valid_tokens = query_start_loc[num_valid_reqs]

        q_per_req = query_start_loc[1:] - query_start_loc[:-1]
        max_num_seqs = seq_lens.shape[0]
        token_to_req_indices = jnp.repeat(jnp.arange(max_num_seqs),
                                          q_per_req,
                                          total_repeat_length=num_tokens)

        # One buffer, two independent block tables at different granularity;
        # vLLM's group allocator keeps their page-index ranges disjoint.
        assert state_block_tables.shape[0] % max_num_seqs == 0
        max_num_state_blocks_per_req = (state_block_tables.shape[0] //
                                        max_num_seqs)
        # Each cache uses its own metadata's `input_positions`: separate
        # groups can number positions differently. The modulo guards against
        # a stray absolute position aliasing into the next request.
        state_cache_block_num = (state_input_positions // state_block_size
                                 ) % max_num_state_blocks_per_req + (
                                     token_to_req_indices *
                                     max_num_state_blocks_per_req)
        state_cache_block_offset = state_input_positions % state_block_size
        slot_mapping = (
            state_block_tables[state_cache_block_num] * state_block_size +
            state_cache_block_offset)
        slot_mapping = jnp.where(
            jnp.arange(num_tokens) < num_valid_tokens, slot_mapping, -1)

        assert k_block_tables.shape[0] % max_num_seqs == 0
        max_num_k_blocks_per_req = k_block_tables.shape[0] // max_num_seqs
        k_block_size = cache.shape[1] * cache.shape[2]
        k_cache_block_num = (
            (k_input_positions // compress_ratio) //
            k_block_size) % max_num_k_blocks_per_req + (
                token_to_req_indices * max_num_k_blocks_per_req)
        k_cache_block_offset = ((k_input_positions // compress_ratio) %
                                k_block_size)
        kv_slot_mapping = (k_block_tables[k_cache_block_num] * k_block_size +
                           k_cache_block_offset)
        kv_slot_mapping = jnp.where(
            jnp.arange(num_tokens) < num_valid_tokens, kv_slot_mapping, -1)

        block_table = state_block_tables.reshape(max_num_seqs, -1)

        out = forward_fn(
            kv_score=kv_score,
            ape=ape,
            norm_weight=norm_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping,
            cache=cache,
            state_block_size=state_block_size,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            rms_eps=rms_eps,
            quant_block=quant_block,
        )

        return out

    if cache.shape[0] == 0:
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return cache

    return _compressor_local(kv_score, ape, norm_weight, cos_sin_cache,
                             positions, request_distribution, query_start_loc,
                             seq_lens, state_input_positions,
                             state_block_tables, k_input_positions,
                             k_block_tables, cache)


class VllmCompressorStateCache(CompressorStateCache):
    """Compressor state cache, paged to fit the host compressed-KV page."""

    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ) -> None:
        super().__init__(
            state_dim,
            dtype,
            compress_ratio,
            prefix,
        )
        coff = 1 + (compress_ratio == 4)
        self.head_dim = state_dim // 2 // coff
        # Block size follows the host page this state packs into, not the base
        # class's CUDA constants; `get_kv_cache_spec` sets the final value.
        self.compress_ratio = compress_ratio
        self._state_coff = coff

    def _state_block_size(self, cache_block_size: int) -> int:
        """Rows of compressor state packed into one host compressed-KV page."""
        compressed_kv_cache_bz = cache_block_size // self.compress_ratio
        # 4: bytes per f32. 2: kv-dim + score-dim.
        block_size = compressed_kv_cache_bz // 4 // 2 // self._state_coff
        if block_size <= 0:
            # Floors to 0 below cache_block_size 1024 (an HCA page is 4 rows
            # at 512), which would emit a zero-sized page.
            raise ValueError(
                f"{self.prefix!r}: a compressed-KV page of "
                f"{compressed_kv_cache_bz} rows is too small to pack this "
                f"layer's compressor state into (compress_ratio "
                f"{self.compress_ratio}, cache block size {cache_block_size})."
            )
        return block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # uint8 is the escape hatch: `kv_cache_spec_normalizer` retypes every
        # other spec to the model's KV dtype, which would truncate raw state.
        # The physical overlay onto the host tensor is done by the runner.
        self.block_size = self._state_block_size(
            vllm_config.cache_config.block_size)
        bytes_per_row = self.state_dim * 4
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(bytes_per_row, 128),
            dtype=torch.uint8,
            sliding_window=self.sliding_window,
            alignment=None,
        )


class VllmDeepseekCompressor(DeepseekCompressor):
    """TPU compressor: writes compressed KV rows and owns the state cache."""

    def __init__(self, *args, **kwargs) -> None:
        orig_state_cache = dsv4_compressor.CompressorStateCache
        dsv4_compressor.CompressorStateCache = VllmCompressorStateCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_compressor.CompressorStateCache = orig_state_cache
        # The MXFP4 record layout the base class selects for the indexer has no
        # TPU kernel: ours always emits the FP8/UE8M0 layout, so honoring the
        # flag would write records the indexer reads as MXFP4.
        if self.use_fp4_cache:
            raise NotImplementedError(
                "DeepSeek-V4 on TPU does not support "
                "`attention_config.use_fp4_indexer_cache`; the compressor "
                "kernel only emits the FP8/UE8M0 cache layout.")
        self.num_layers = get_current_vllm_config(
        ).model_config.hf_config.num_hidden_layers

    @property
    def compressor_op(self):
        """Built on first use, since `state_block_size` is baked in statically.

        Until `get_kv_cache_spec` runs during KV cache init, that attribute
        holds the base class's CUDA-derived value rather than what this host
        page can take, and the op would trace against the wrong geometry.
        """
        op = self.__dict__.get("_compressor_op_instance")
        if op is None:
            op = self._build_compressor_op()
            object.__setattr__(self, "_compressor_op_instance", op)
        return op

    def _build_compressor_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        state_dim = getattr(self.state_cache, "state_dim",
                            getattr(self, "state_dim", 512))

        wrapped_fn = functools.partial(
            _compressor_jax,
            state_block_size=self.state_cache.block_size,
            head_dim=self.head_dim,
            state_dim=state_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            rms_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
        )

        op_name = f"pallas::deepseek_v4_compressor_{self.head_dim}_{self.compress_ratio}_{int(self.overlap)}_{self.state_cache.block_size}_{state_dim}"

        global _compressor_op_cache
        if op_name in _compressor_op_cache:
            return _compressor_op_cache[op_name]

        attn_data_axis = None
        batch_axis = BATCH_AXIS
        # Donate the cache so XLA writes in place instead of allocating a
        # second full-size buffer; `forward` copies the result back. The index
        # counts torch_tpu's filtered tensor args, where `cache` is last.
        compressor_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            donate_argnums=(12, ),
            input_partition_specs=(
                P(attn_data_axis, None),  # kv_score
                P(),  # ape
                P(),  # norm_weight
                P(),  # cos_sin_cache
                P(attn_data_axis),  # positions
                P(),  # request_distribution
                P(),  # query_start_loc
                P(batch_axis),  # seq_lens
                P(attn_data_axis),  # state_input_positions
                P(batch_axis),  # state_block_tables
                P(attn_data_axis),  # k_input_positions
                P(batch_axis),  # k_block_tables
                P(),  # cache
            ),
        )

        def _fake_compressor(kv_score, ape, norm_weight, cos_sin_cache,
                             positions, request_distribution, query_start_loc,
                             seq_lens, state_input_positions,
                             state_block_tables, k_input_positions,
                             k_block_tables, cache, *args, **kwargs):
            return torch.empty_like(cache)

        compressor_jax_op.register_fake(_fake_compressor)

        _compressor_op_cache[op_name] = compressor_jax_op
        return compressor_jax_op

    def forward(
        self,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        kv_score = kv_score.to(torch.float32)

        attn_ctx = get_forward_context().attn_metadata
        if not isinstance(attn_ctx, dict):
            # As in the base class: without per-layer metadata there is
            # nothing to compress.
            return

        # Each cache must use its own layer's metadata: they are separate
        # groups with separate block tables, so borrowing another layer's
        # would scatter state rows into that layer's pages.
        prefix_key = getattr(getattr(self, "state_cache", None), "prefix",
                             None)
        _k_cache_obj = getattr(self, "k_cache", None)
        k_cache_prefix_key = getattr(
            _k_cache_obj, "prefix", getattr(_k_cache_obj, "custom_prefix",
                                            None))
        for _name, _key in (("state_cache", prefix_key), ("k_cache",
                                                          k_cache_prefix_key)):
            if _key not in attn_ctx:
                raise KeyError(
                    f"DeepSeek-V4 compressor {_name} prefix {_key!r} has no "
                    "attention metadata; using another layer's would corrupt "
                    f"its pages. Known: {sorted(attn_ctx)}")
        state_metadata = attn_ctx[prefix_key]
        k_cache_metadata = attn_ctx[k_cache_prefix_key]

        if state_metadata is not None:
            state_req_dist = state_metadata.request_distribution
            state_q_start_loc = state_metadata.query_start_loc
            state_seq_lens = state_metadata.seq_lens
            state_block_tables = state_metadata.block_tables
            if state_block_tables is not None:
                state_block_tables = state_block_tables.flatten()
            # Fall back to model-level positions only when this metadata has
            # none; the two caches can number positions differently.
            state_input_positions = getattr(state_metadata, "input_positions",
                                            None)
            if state_input_positions is None:
                state_input_positions = positions
        else:
            state_req_dist = state_q_start_loc = state_seq_lens = state_block_tables = None
            state_input_positions = positions

        # `k_block_tables` addresses the shared compressed-KV buffer, so it
        # must come from k_cache's metadata, not the state cache's.
        if k_cache_metadata is not None:
            k_block_tables = k_cache_metadata.block_tables
            if k_block_tables is not None:
                k_block_tables = k_block_tables.flatten()
            k_input_positions = getattr(k_cache_metadata, "input_positions",
                                        None)
            if k_input_positions is None:
                k_input_positions = positions
        else:
            k_block_tables = None
            k_input_positions = positions

        # vLLM binds `.kv_cache` to a numel==0 placeholder until
        # initialize_kv_cache runs, so test emptiness, not None.
        _k_cache_tensor = getattr(self.k_cache, "kv_cache", None)
        if _k_cache_tensor is None or _k_cache_tensor.numel() == 0:
            return

        # vLLM allocates this as `kv_cache_dtype`, but the kernel writes raw
        # bytes, and assigning uint8 into an fp8 array converts rather than
        # bitcasts. `.view()` shares storage, so the writes still land.
        orig_cache = self.k_cache.kv_cache
        if orig_cache is not None and orig_cache.dtype != torch.uint8:
            orig_cache = orig_cache.view(torch.uint8)

        # Writes use the per-cache positions, reads use the model-level
        # `positions`; the gather misses written rows if the two diverge.
        updated_cache = self.compressor_op(
            kv_score,
            self.ape.clone(),
            self.norm.weight.clone(),
            rotary_emb.cos_sin_cache,
            positions,
            state_req_dist,
            state_q_start_loc,
            state_seq_lens,
            state_input_positions,
            state_block_tables,
            k_input_positions,
            k_block_tables,
            orig_cache,
        )

        orig_cache.copy_(updated_cache)
