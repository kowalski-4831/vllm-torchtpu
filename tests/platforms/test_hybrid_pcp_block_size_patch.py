from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec,
                                        SlidingWindowSpec)

from vllm_torchtpu import _patch_vllm_hybrid_pcp_block_sizes


def _config(*, pcp: int):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=512,
            enable_prefix_caching=False,
            hash_block_size=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=pcp,
        ),
        kv_transfer_config=None,
    )


def _full_spec(block_size: int = 512):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _mamba_spec(block_size: int = 4096):
    return MambaSpec(
        block_size=block_size,
        shapes=((3, 8192), ),
        dtypes=(torch.bfloat16, ),
    )


def _resolve_outcome(resolver, kv_cache_config, vllm_config):
    try:
        return "return", resolver(kv_cache_config, vllm_config)
    except ValueError as error:
        return "value_error", str(error)


def test_hybrid_full_attention_mamba_pcp_resolves_effective_block_size():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.engine import core as engine_core

    assert engine_core.resolve_kv_cache_block_sizes is resolve_kv_cache_block_sizes

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(["gdn"], _mamba_spec()),
        ],
    )

    assert resolve_kv_cache_block_sizes(kv_cache_config,
                                        _config(pcp=8)) == (4096, 4096)


def test_hybrid_pcp_patch_keeps_non_mamba_scope():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core import kv_cache_utils

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(
                ["sliding"],
                SlidingWindowSpec(
                    block_size=512,
                    num_kv_heads=4,
                    head_size=128,
                    sliding_window=1024,
                    dtype=torch.bfloat16,
                ),
            ),
        ],
    )

    patched_resolver = kv_cache_utils.resolve_kv_cache_block_sizes
    original_resolver = (
        kv_cache_utils._tpu_original_resolve_kv_cache_block_sizes)
    vllm_config = _config(pcp=8)

    assert _resolve_outcome(
        patched_resolver,
        kv_cache_config,
        vllm_config,
    ) == _resolve_outcome(
        original_resolver,
        kv_cache_config,
        vllm_config,
    )
