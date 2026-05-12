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
import torch
from einops import rearrange
from torch_tpu._internal import pallas
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn_linear_attn import \
    GatedDeltaNetAttention

from tpu_inference import envs
from tpu_inference.layers.common.gdn_attention import (GdnAttentionConfig,
                                                       run_jax_gdn_attention)
from tpu_inference.layers.common.ragged_gated_delta_rule_wrapper import \
    RaggedGatedDeltaRuleImpl
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def gdn_attention_core_tpu(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    config: GdnAttentionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (new_conv_state, new_recurrent_state), output = run_jax_gdn_attention(
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        mesh=mesh,
        config=config,
    )

    return new_conv_state, new_recurrent_state, output


@GatedDeltaNetAttention.register_oot
class VllmGatedDeltaNetAttention(GatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gdn_op = self._build_gdn_op()

    def _build_gdn_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        config = GdnAttentionConfig(
            ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl(
                envs.RAGGED_GATED_DELTA_RULE_IMPL))
        local_num_v_heads = self.num_v_heads // self.tp_size
        wrapped_fn = functools.partial(
            gdn_attention_core_tpu,
            mesh=vllm_context.mesh,
            n_kq=self.num_k_heads // self.tp_size,
            n_v=local_num_v_heads,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
            kernel_size=self.conv_kernel_size,
            config=config,
        )

        op_name = f"pallas::gdn_attention_{self.prefix.replace('.', '_')}"
        gdn_jax_op = pallas.jax_op(op_name, wrapped_fn, donate_argnums=(3, 4))

        def _fake_gdn(mixed_qkv, _b, _a, conv_state, recurrent_state, *args,
                      **kwargs):
            num_tokens = mixed_qkv.size(0)
            out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
            return torch.empty_like(conv_state), torch.empty_like(
                recurrent_state), torch.empty(out_shape,
                                              dtype=mixed_qkv.dtype,
                                              device=mixed_qkv.device)

        gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                     conv_state: torch.Tensor, recurrent_state: torch.Tensor,
                     conv_weight: torch.Tensor, conv_bias: torch.Tensor | None,
                     A_log: torch.Tensor, dt_bias: torch.Tensor,
                     state_indices: torch.Tensor,
                     query_start_loc: torch.Tensor,
                     request_distribution: torch.Tensor,
                     seq_lens: torch.Tensor) -> torch.Tensor:
            new_conv, new_rec, outputs = gdn_jax_op(
                mixed_qkv, b, a, conv_state, recurrent_state, conv_weight,
                conv_bias, A_log, dt_bias, state_indices, query_start_loc,
                request_distribution, seq_lens)

            conv_state.copy_(new_conv)
            recurrent_state.copy_(new_rec)

            return outputs

        return gdn_impl

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba)
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value))
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        kv_cache = getattr(self, "kv_cache", None)
        local_num_v_heads = self.num_v_heads // self.tp_size

        # During warmup or memory profiling, the kv_cache might not be allocated yet
        if kv_cache is None or kv_cache[0].numel() == 0:
            core_attn_out = torch.zeros(
                (num_tokens, local_num_v_heads, self.head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            conv_state, recurrent_state = None, None
        else:
            fc = get_forward_context()
            attn_metadata = fc.attn_metadata[self.prefix]

            conv_state, recurrent_state = kv_cache
            # Extract block tables and convert them to state indices
            max_reqs = attn_metadata.seq_lens.shape[0]
            max_blocks_per_req = attn_metadata.block_tables.shape[0] // max_reqs
            block_tables_2d = torch.reshape(
                attn_metadata.block_tables,
                (max_reqs, max_blocks_per_req),
            )
            state_indices = block_tables_2d[:, 0].to(torch.int32)

            # Execute the TorchTPU custom op
            core_attn_out = self.gdn_op(
                mixed_qkv, b, a, conv_state, recurrent_state,
                self.conv1d.weight, self.conv1d.bias, self.A_log, self.dt_bias,
                state_indices, attn_metadata.query_start_loc,
                attn_metadata.request_distribution, attn_metadata.seq_lens)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        core_attn_out = core_attn_out.view(num_tokens, local_num_v_heads,
                                           self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)
