import functools
from dataclasses import dataclass, field
from typing import Any

import jax
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import \
    AttentionMetadataBuilder as BaseAttentionMetadataBuilder


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
    ],
    meta_fields=[],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None

    query_start_loc_cpu: Any = field(init=False)
    seq_lens_cpu: Any = field(init=False)


@dataclass
class AttentionMetadataBuilderContext:
    """Per-call inputs the TPU metadata builder reads from the runner.

    Stashed on TPUModelRunner before invoking _build_attention_metadata so
    builder.build() can pick them up. Only fields TPU populates differently from
    the parent runner (seq_lens / query_start_loc) live in TPU's own _cpu
    staging tensors, not parent's CpuGpuBuffers) or that have no
    `CommonAttentionMetadata` equivalent (num_reqs, use_max_model_len,
    start_index, request_distribution, position_ids_override) live here.
    """
    num_reqs: int
    start_index: int
    use_max_model_len: bool
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    request_distribution: torch.Tensor
    position_ids_override: torch.Tensor | None = None


class AttentionMetadataBuilder(BaseAttentionMetadataBuilder):
    """Single shared metadata builder for every TPU attention group.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, runner,
                 kv_cache_group_id):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.runner = runner
        self.kv_cache_group_id = kv_cache_group_id
        self.target_block_size = getattr(self.kv_cache_spec, "block_size",
                                         runner.block_size)

        block_table_obj = runner.input_batch.block_table[
            self.kv_cache_group_id]
        self.block_tables_cpu = torch.zeros(
            (runner.max_num_reqs, block_table_obj.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu")

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        target_num_reqs = common_attn_metadata.num_reqs
        block_table_obj = runner.input_batch.block_table[
            self.kv_cache_group_id]
        if ctx.use_max_model_len:
            target_num_blocks = block_table_obj.max_num_blocks_per_req
        else:
            assert runner.most_model_len is not None
            target_num_blocks = cdiv(runner.most_model_len,
                                     self.target_block_size)

        # `position_ids` is only used for the dummy run in dummy runs, where we
        # want to use fixed position IDs instead of copying from the CPU tensor
        # that gets updated every step.
        if ctx.position_ids_override is not None:
            block_tables_dev = torch.zeros(
                (ctx.num_reqs * target_num_blocks, ),
                dtype=torch.int32).to(runner.device)
            input_positions = ctx.position_ids_override
        else:
            block_tables = self.block_tables_cpu[:target_num_reqs, :
                                                 target_num_blocks]
            block_tables.zero_()
            block_tables[:ctx.num_reqs, :target_num_blocks] = (
                block_table_obj.get_cpu_tensor()[
                    ctx.start_index:ctx.start_index +
                    ctx.num_reqs, :target_num_blocks])
            # Flatten on CPU before H2D to avoid device-side as_strided/reshape
            # materialization on every decode step.
            block_tables_dev = block_tables.reshape(-1).to(runner.device,
                                                           non_blocking=True)
            input_positions = runner.position_ids

        return AttentionMetadata(
            input_positions=input_positions,
            block_tables=block_tables_dev,
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
        )
