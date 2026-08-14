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

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_indexer import (
    VllmDeepseekV4IndexerCache, _jax_quantize_tensor)


def test_quantize_all_zero_row_is_zero_not_nan():
    """A padded token quantizes to zeros, not NaN.

    Shape-padded batches carry all-zero `hidden_states`, and `wq_b` is
    bias-free, so `q_rope` has exact zero rows. Scaling those by `1 / 0` = inf
    gives NaN, which is not `-inf` and so slips past the `-1` sentinel guard in
    `streamindex_topk`, letting garbage indices reach the gather.
    """
    tensor = jnp.asarray([[0.0, 0.0, 0.0, 0.0], [0.5, -1.0, 0.25, 0.0]],
                         dtype=jnp.float32)
    q, scale = _jax_quantize_tensor(jnp.float8_e4m3fn, tensor)

    q32 = np.asarray(q.astype(jnp.float32))
    assert not np.isnan(q32).any(), "zero row quantized to NaN"
    assert (q32[0] == 0).all()
    assert np.asarray(scale)[0] == 0.0


def test_quantize_preserves_a_normal_row():
    """The zero-row guard must not perturb rows that do have signal."""
    tensor = jnp.asarray([[0.5, -1.0, 0.25, 0.0]], dtype=jnp.float32)
    q, scale = _jax_quantize_tensor(jnp.float8_e4m3fn, tensor)

    dequant = np.asarray(q.astype(jnp.float32)) * np.asarray(scale)[:, None]
    np.testing.assert_allclose(dequant,
                               np.asarray(tensor),
                               rtol=0.1,
                               atol=1e-3)


@pytest.mark.parametrize("block_size, expected_ratio", [(1024, 4), (2, 2)])
def test_indexer_cache_spec_is_raw_uint8(block_size, expected_ratio):
    """The indexer cache is a packed byte record, not KV-dtype features.

    Any dtype but uint8 is retyped by `kv_cache_spec_normalizer`; the compress
    ratio is clamped so it can never exceed the page it is packed into.
    """
    cache = VllmDeepseekV4IndexerCache.__new__(VllmDeepseekV4IndexerCache)
    cache.cache_config = SimpleNamespace(block_size=block_size)
    cache.compress_ratio = 4

    spec = cache.get_kv_cache_spec(SimpleNamespace())

    assert isinstance(spec, MLAAttentionSpec)
    assert spec.dtype == torch.uint8
    assert spec.block_size == block_size
    assert spec.num_kv_heads == 1
    # 128 record bytes + 1 scale byte, rounded to the 128-byte lane.
    assert spec.head_size == 256
    assert spec.compress_ratio == expected_ratio
