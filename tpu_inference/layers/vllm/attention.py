# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import torch
from torch_tpu._internal import pallas, sync
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, next_power_of_2

from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

# TPU requires the head size to be a multiple of 128.
TPU_HEAD_SIZE_ALIGNMENT = 128

# Note: TPU can fp8 as storage dtype but doesn't support converting from uint8
# from to fp32 directly. That's why it has a dtype mapping different from GPU
TPU_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


def _pallas_rpa_kernel(
    kv_cache,
    query,
    key,
    value,
    seq_lens,
    block_tables,
    query_start_loc,
    request_distribution,
    sinks,
    *,
    mesh,
    q_scale,
    k_scale,
    v_scale,
    sliding_window,
):
    metadata = AttentionMetadata(
        input_positions=
        None,  # FIXME: kernel wrapper will have error if pass this argument
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    new_kv_cache, outputs = attention(
        kv_cache,
        query,
        key,
        value,
        metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sinks=sinks,
        attention_chunk_size=sliding_window,
    )
    return new_kv_cache, outputs


# =========================================================================================
# VLLM Attention Backend
# =========================================================================================


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        padded_head_size = (cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) *
                            TPU_HEAD_SIZE_ALIGNMENT)
        # Two different RPA kernels have different KV cache layouts:
        # - hd64 (head_dim=64): K/V packed along head_dim
        # - v3 (head_dim!=64): K/V packed along heads
        # The Pallas kernels expect a 5D KV cache: [L, S, Kx2 / kv_packing, kv_packing, H]
        # where Kx2 = num_kv_heads for hd64 and Kx2 = num_kv_heads * 2 for v3.
        use_hd64 = (head_size == 64)
        if cache_dtype_str != "auto":
            raise NotImplementedError

        kv_packing = 2
        num_kv_heads_x2 = num_kv_heads if use_hd64 else num_kv_heads * 2
        num_kv_heads_x2 = cdiv(num_kv_heads_x2, kv_packing) * kv_packing
        return (
            num_blocks,
            block_size,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            padded_head_size,
        )

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # In recent TPU generations, up to v6e, the SMEM size is 1MB. The
    # block_tables within the PallasMetadata constitute almost the entire SMEM
    # requirement. Its size is max_num_seqs * num_page_per_seq * 4 (Int). Here
    # we simply make sure that the size is smaller than half of SMEM capacity.
    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        max_num_page_per_req = (1024 * 1024 // 2 //
                                vllm_config.scheduler_config.max_num_seqs // 4)
        min_page_size = cdiv(vllm_config.model_config.max_model_len,
                             max_num_page_per_req)
        min_page_size = 1 << (min_page_size - 1).bit_length()
        return min_page_size

    @staticmethod
    def get_max_num_seqs(model_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(model_len, page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4

    # TPU has limited SREGs (scalar registers), if page_size is too small, we
    # can spill SREGs easily which leads to bad performance. The strategy we
    # apply here is trying to split max-model-len to 16 pages which make the
    # spill less likely. Meanwhile we make sure the page size is in [16, 256].
    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        # TODO: This is a temporary fix for vmem OOM.
        # For long model length, we use 16 page-size to avoid too much
        # VMEM spill. A more robust solution should be implemented to
        # handle VREG spills.
        if vllm_config.model_config.max_model_len > 8192:
            return 16
        page_size = next_power_of_2(
            vllm_config.model_config.max_model_len) // 16
        if page_size <= 16:
            return 16
        if page_size >= 256:
            return 256
        return page_size


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(
                kv_cache_dtype.lower().strip())

        # Store sinks for attention sink optimization
        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")

        self.rpa_kernel = None

    def init_rpa_kernel(self, q_scale: float, k_scale: float, v_scale: float):
        ctx = get_vllm_model_wrapper_context()
        mesh = ctx.mesh

        # Prepare wrapper function with static arguments
        wrapped_fn = functools.partial(
            _pallas_rpa_kernel,
            mesh=mesh,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sliding_window=self.sliding_window,
        )

        # Create Torch operator
        # TODO: The current wrapped_fn still depends on jax shard_map,
        #       need to revise once we work on distributed.
        self.rpa_kernel = pallas.custom_jax_kernel(wrapped_fn)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """Process sinks after model loading - convert to float32 as required by RPA kernel."""
        if self.sinks is not None:
            # RPA v3 kernel requires sinks to be float32
            self.sinks = torch.nn.Parameter(self.sinks.to(torch.float32),
                                            requires_grad=False)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads_x2 // kv_packing,
                 kv_packing, padded_head_size] (preferred)
                or legacy 4D
                [num_blocks, block_size, num_kv_heads_x2, padded_head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for PallasAttentionBackendImpl")

        # For determine_available_memory case.
        if kv_cache.numel() == 0:
            logger.warning("Returning early due to kv_cache.numel() == 0")
            if output is None:
                output = torch.ones_like(query)
            return output

        q_len, q_compute_dim = query.shape
        k_len, k_compute_dim = key.shape
        assert key.shape == value.shape
        assert q_compute_dim == self.head_size * self.num_heads
        assert k_compute_dim == self.head_size * self.num_kv_heads

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            raise NotImplementedError(
                "Quantized KV cache is not supported for PallasAttentionBackendImpl"
            )

        sink = self.sinks
        query = query.view(q_len, self.num_heads, self.head_size)
        key = key.view(k_len, self.num_kv_heads, self.head_size)
        value = value.view(k_len, self.num_kv_heads, self.head_size)

        if self.rpa_kernel is None:
            self.init_rpa_kernel(q_scale, k_scale, v_scale)

        # TODO (geyuhao) the support of this API is pending discussion. 
        # This line will only influence performance, not functionality
        # # Mark kv_cache avaliable for donation
        # pallas.set_buffer_donor_(kv_cache, True)

        # Call the operator
        new_kv_cache, outputs = self.rpa_kernel(
            kv_cache,
            query,
            key,
            value,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
            sink,
        )

        # update kv cache
        kv_cache.copy_(new_kv_cache)

        # TODO (geyuhao) ideally we don't want this
        sync.synchronize(kv_cache)

        return outputs.reshape(q_len, self.num_heads * self.head_size)
