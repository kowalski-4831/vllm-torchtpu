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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_compressor import (
    VllmCompressorStateCache, VllmDeepseekCompressor)


def _state_cache(compress_ratio: int, state_dim: int = 1024):
    """A state cache with the attributes its geometry depends on.

    The base `__init__` registers into the vLLM forward context, which needs a
    live config; the geometry under test does not.
    """
    cache = VllmCompressorStateCache.__new__(VllmCompressorStateCache)
    coff = 1 + (compress_ratio == 4)
    cache.prefix = "model.layers.0.self_attn.compressor.state_cache"
    cache.compress_ratio = compress_ratio
    cache._state_coff = coff
    cache.state_dim = state_dim
    cache.sliding_window = coff * compress_ratio
    cache.block_size = 0
    return cache


@pytest.mark.parametrize(
    "compress_ratio, cache_block_size, expected",
    [
        # A CSA host page holds 256 compressed rows; each state row is
        # 2 (kv + score) x 2 (coff) f32 words wide, so 16 rows fit.
        (4, 1024, 16),
        (4, 512, 8),
        # HCA compresses 128:1, leaving an 8-row page and a single state row.
        (128, 1024, 1),
    ],
)
def test_state_block_size_follows_the_host_page(compress_ratio,
                                                cache_block_size, expected):
    cache = _state_cache(compress_ratio)
    assert cache._state_block_size(cache_block_size) == expected


def test_state_block_size_refuses_a_zero_sized_page():
    """Below 1024 an HCA page floors to zero rows; that must not be emitted."""
    cache = _state_cache(128)
    with pytest.raises(ValueError, match="too small to pack"):
        cache._state_block_size(512)


def test_state_cache_spec_is_raw_uint8_at_the_host_page_size():
    """The spec must survive `kv_cache_spec_normalizer` unretyped.

    Any dtype but uint8 is rewritten to the model's KV dtype, which truncates
    the raw f32 state; the block size must come from the final cache config,
    not the base class's CUDA constant.
    """
    cache = _state_cache(4, state_dim=1024)
    spec = cache.get_kv_cache_spec(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=1024)))

    assert spec.dtype == torch.uint8
    assert spec.block_size == 16
    assert cache.block_size == 16
    # 1024 f32 words, already a multiple of 128.
    assert spec.head_size == 4096
    assert spec.num_kv_heads == 1
    assert spec.sliding_window == 8
    # vLLM's DSv4 grouping is isinstance-based, so the base type is load-bearing.
    assert isinstance(spec, SlidingWindowMLASpec)


def test_compressor_op_is_built_lazily_and_cached():
    """Building in `__init__` bakes the pre-KV-init `state_block_size`.

    That constant is the base class's CUDA value until `get_kv_cache_spec`
    replaces it, so an eagerly built op traces against the wrong page geometry.
    """
    assert isinstance(
        VllmDeepseekCompressor.__dict__["compressor_op"],
        property), "compressor_op must stay a property, not an __init__ field"

    compressor = VllmDeepseekCompressor.__new__(VllmDeepseekCompressor)
    sentinel = object()
    with patch.object(VllmDeepseekCompressor,
                      "_build_compressor_op",
                      return_value=sentinel) as build:
        assert compressor.compressor_op is sentinel
        assert compressor.compressor_op is sentinel
    build.assert_called_once()
