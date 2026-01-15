# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass

import torch
from torch_tpu._internal import pallas
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, next_power_of_2

# Import V3 kernels - these handle KV cache update internally
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention as jax_ragged_paged_attention_v3
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 import \
    ragged_paged_attention_hd64 as jax_ragged_paged_attention_v3_hd64

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

# =========================================================================================
# Pallas Kernel Wrappers (V3 - unified attention + KV cache update)
# =========================================================================================


def pallas_ragged_paged_attention_v3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_distribution: torch.Tensor,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TorchTPU-compatible wrapper for V3 ragged_paged_attention.

    This kernel handles both attention computation AND KV cache update internally.

    Args:
        query: [num_tokens, num_q_heads, head_dim]
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        kv_cache: [num_blocks, block_size, num_kv_heads*2 // packing, packing, padded_head_dim]
        seq_lens: [max_num_seqs] - sequence lengths (KV lengths)
        page_indices: [max_num_seqs * pages_per_seq] - flattened page indices
        query_start_loc: [max_num_seqs + 1] - cumulative query lengths
        request_distribution: [3] - (decode_end, prefill_end, mixed_end)
        sm_scale: softmax scale
        sliding_window: optional sliding window size
        soft_cap: optional logit soft cap
        k_scale: key scale for quantization
        v_scale: value scale for quantization

    Returns:
        (output, updated_kv_cache) tuple
    """
    wrapped_fn = functools.partial(
        jax_ragged_paged_attention_v3,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch_fn = pallas.custom_jax_kernel(wrapped_fn)
    output, updated_kv_cache = torch_fn(query, key, value, kv_cache, seq_lens,
                                        page_indices, query_start_loc,
                                        request_distribution)

    return output, updated_kv_cache


def pallas_ragged_paged_attention_v3_hd64(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_distribution: torch.Tensor,
    attention_sink: torch.Tensor | None = None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    strict_sliding_window: bool = True,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TorchTPU-compatible wrapper for V3 ragged_paged_attention_hd64.

    Specialized for head_dim=64 with optional attention sink support.
    """
    wrapped_fn = functools.partial(
        jax_ragged_paged_attention_v3_hd64,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        strict_sliding_window=strict_sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch_fn = pallas.custom_jax_kernel(wrapped_fn)

    if attention_sink is not None:
        output, updated_kv_cache = torch_fn(query, key, value, kv_cache,
                                            seq_lens, page_indices,
                                            query_start_loc,
                                            request_distribution,
                                            attention_sink)
    else:
        output, updated_kv_cache = torch_fn(query, key, value, kv_cache,
                                            seq_lens, page_indices,
                                            query_start_loc,
                                            request_distribution)

    return output, updated_kv_cache


# =========================================================================================
# VLLM Attention Backend
# =========================================================================================


@dataclass
class PallasMetadata:
    """Metadata for Pallas V3 attention.

    V3 kernel interface expects:
    - seq_lens: [max_num_seqs] - sequence lengths (KV lengths)
    - block_tables: [max_num_seqs, pages_per_seq] - page indices (will be flattened)
    - query_start_loc: [max_num_seqs + 1] - cumulative query lengths
    - request_distribution: [3] - (decode_end, prefill_end, mixed_end)
    """
    # block_tables: [max_num_seqs, pages_per_seq] - 2D page indices
    block_tables: torch.Tensor

    # seq_lens: [max_num_seqs] - sequence lengths (total tokens including new ones)
    seq_lens: torch.Tensor

    # query_start_loc: [max_num_seqs + 1] - cumulative query lengths
    query_start_loc: torch.Tensor

    # request_distribution: [3] - (decode_end, prefill_end, mixed_end)
    # For single-chip: [0, num_seqs, num_seqs] means all requests are "mixed" mode
    request_distribution: torch.Tensor


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
        return (num_blocks, block_size, num_kv_heads * 2, padded_head_size)

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
            if self.sinks.shape[0] != num_heads:
                raise ValueError(
                    f"Sinks must have the same number of heads as num_heads. "
                    f"Got sinks.shape[0]={self.sinks.shape[0]}, num_heads={num_heads}"
                )
            if head_size != 64:
                raise NotImplementedError(
                    "Attention sink support is only available when head_dim==64. "
                    f"Got head_size={head_size}")

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
        attn_metadata: PallasMetadata,
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
                [num_blocks, block_size, num_kv_heads * 2, padded_head_size]
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

        num_tokens, hidden_size = query.shape
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Prepare V3 kernel inputs
        page_indices = attn_metadata.block_tables.flatten().to(torch.int32)
        seq_lens = attn_metadata.seq_lens.to(torch.int32)
        query_start_loc = attn_metadata.query_start_loc.to(torch.int32)
        request_distribution = attn_metadata.request_distribution.to(
            torch.int32)

        # Quantization scaling
        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        if self.kv_cache_quantized_dtype is not None and k_scale != 0.0 and v_scale != 0.0:
            dtype_info = torch.finfo(self.kv_cache_quantized_dtype)
            key = torch.clamp(
                key.to(torch.float32) / k_scale, dtype_info.min,
                dtype_info.max).to(self.kv_cache_quantized_dtype)
            value = torch.clamp(
                value.to(torch.float32) / v_scale, dtype_info.min,
                dtype_info.max).to(self.kv_cache_quantized_dtype)

        # Reshape KV cache from 4D (vLLM format) to 5D (V3 kernel format)
        # 4D: [num_blocks, block_size, num_kv_heads * 2, padded_head_size]
        # 5D: [num_blocks, block_size, num_kv_heads * 2 // packing, packing, padded_head_size]
        num_blocks, block_size, num_kv_heads_x2, padded_head_size = kv_cache.shape
        packing = 2  # For bfloat16
        kv_cache_5d = kv_cache.view(num_blocks, block_size,
                                    num_kv_heads_x2 // packing, packing,
                                    padded_head_size)

        # Choose kernel based on head_size
        if self.head_size == 64:
            output, updated_kv_cache_5d = pallas_ragged_paged_attention_v3_hd64(
                query,
                key,
                value,
                kv_cache_5d,
                seq_lens,
                page_indices,
                query_start_loc,
                request_distribution,
                self.sinks,
                sm_scale=self.scale,
                sliding_window=self.sliding_window,
                k_scale=k_scale if k_scale != 1.0 else None,
                v_scale=v_scale if v_scale != 1.0 else None,
            )
        else:
            output, updated_kv_cache_5d = pallas_ragged_paged_attention_v3(
                query,
                key,
                value,
                kv_cache_5d,
                seq_lens,
                page_indices,
                query_start_loc,
                request_distribution,
                sm_scale=self.scale,
                sliding_window=self.sliding_window,
                soft_cap=self.logits_soft_cap,
                k_scale=k_scale if k_scale != 1.0 else None,
                v_scale=v_scale if v_scale != 1.0 else None,
            )

        # Reshape updated KV cache back to 4D and copy in-place
        updated_kv_cache_4d = updated_kv_cache_5d.view(num_blocks, block_size,
                                                       num_kv_heads_x2,
                                                       padded_head_size)
        kv_cache.copy_(updated_kv_cache_4d)

        # Handle padded head_dim from kernel
        if output.shape[-1] != self.head_size:
            output = output[:, :, :self.head_size]

        return output.reshape(num_tokens, hidden_size)
