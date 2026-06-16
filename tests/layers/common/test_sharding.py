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
"""Tests for the sharding_config storage fix.

torchtpu-vllm used to attach the parsed sharding config as a dynamic attribute
(``vllm_config.sharding_config = manager``). That broke vLLM's
``replace()``/``deepcopy`` (which iterate ``__dict__`` and reject non-field
members) on the multimodal ``with_hf_config()`` path. The fix stores a
JSON-native dict under ``additional_config["_sharding_config"]`` and rebuilds the
manager on ``get()``. These tests lock in that contract:

  * the stored value is JSON-native (so ``VllmConfig.compute_hash``'s
    ``json.dumps(additional_config)`` succeeds),
  * nothing is written to the config's ``__dict__`` (so ``replace()`` is safe),
  * the data round-trips through ``initialize() -> get()`` and through a
    ``deepcopy`` of the config,
  * ``initialize()`` keeps its one-time DP side effect (read then normalize).
"""

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.config.utils import replace as vllm_replace

from vllm_torchtpu.layers.common.sharding import (ShardingConfigManager,
                                                  ShardingStrategy)


def _make_vllm_config(*,
                      tensor_parallel_size=1,
                      data_parallel_size=1,
                      additional_config=None,
                      speculative_config=None,
                      lora_config=None):
    """Minimal stand-in for VllmConfig.

    ``ShardingConfigManager.from_vllm_config`` only touches the attributes set
    here (with ``enable_dp_attention`` off, ``model_config``/``cache_config``
    are not read). Using a plain namespace -- rather than a MagicMock -- lets us
    assert that the manager never sets a ``sharding_config`` attribute, which a
    MagicMock would silently fabricate.
    """
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=0,
        data_parallel_size_local=1,
    )
    return SimpleNamespace(
        parallel_config=parallel_config,
        additional_config=({} if additional_config is None else
                           additional_config),
        speculative_config=speculative_config,
        lora_config=lora_config,
        model_config=SimpleNamespace(use_mla=False),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def _make_real_vllm_config(*, tensor_parallel_size=1):
    """A real (pydantic) VllmConfig for exercising vLLM's replace()/compute_hash.

    Heavier than the namespace fake (it constructs ModelConfig, etc.), but
    required to test the actual regression: the stored sharding config must
    survive vLLM's dataclass ``replace()`` and its JSON-based ``compute_hash()``.
    ``device='cpu'`` keeps it constructible off-TPU. Mirrors the VllmConfig built
    in tests/runner/test_tpu_runner.py.
    """
    return VllmConfig(
        model_config=ModelConfig(tokenizer_mode="auto",
                                 trust_remote_code=False,
                                 seed=0,
                                 dtype=torch.bfloat16),
        cache_config=CacheConfig(block_size=16,
                                 gpu_memory_utilization=0.9,
                                 cache_dtype="auto"),
        scheduler_config=SchedulerConfig(max_num_seqs=16,
                                         max_model_len=1024,
                                         is_encoder_decoder=False),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=tensor_parallel_size),
        device_config=DeviceConfig(device="cpu"),
        additional_config={},
    )


class TestShardingStrategy:

    def test_round_trips_through_asdict(self):
        strategy = ShardingStrategy(tensor_parallelism=4,
                                    expert_parallelism=2,
                                    sequence_parallelism=1,
                                    data_parallelism=2,
                                    attention_data_parallelism=1)
        assert ShardingStrategy(**asdict(strategy)) == strategy

    def test_asdict_is_all_ints(self):
        # compute_hash json.dumps relies on the values being JSON-native ints.
        assert all(
            isinstance(v, int) for v in asdict(ShardingStrategy()).values())


class TestInitialize:

    def test_returns_manager_with_expected_strategy(self):
        cfg = _make_vllm_config(tensor_parallel_size=4)
        manager = ShardingConfigManager.initialize(cfg)
        assert manager.tp_size == 4
        assert manager.total_dp_size == 1
        assert manager.total_devices == 4

    def test_stores_json_native_dict_under_key(self):
        cfg = _make_vllm_config(tensor_parallel_size=2)
        ShardingConfigManager.initialize(cfg)

        stored = cfg.additional_config[
            ShardingConfigManager._ADDITIONAL_CONFIG_KEY]
        assert isinstance(stored, dict)
        assert isinstance(stored["sharding_strategy"], dict)
        assert stored["sharding_strategy"]["tensor_parallelism"] == 2
        assert stored["device_indexes"] is None

    def test_stored_config_is_json_serializable(self):
        # Regression: storing the manager OBJECT broke compute_hash, which does
        # json.dumps(additional_config) with the stdlib encoder.
        cfg = _make_vllm_config(tensor_parallel_size=2)
        ShardingConfigManager.initialize(cfg)

        dumped = json.dumps(cfg.additional_config)
        assert json.loads(dumped) == cfg.additional_config

    def test_does_not_set_dynamic_attribute(self):
        # Regression: the whole point of the fix. A `sharding_config` attribute
        # on __dict__ is what made vLLM's replace() raise
        # "Field 'sharding_config' not found in VllmConfig".
        cfg = _make_vllm_config()
        ShardingConfigManager.initialize(cfg)
        assert not hasattr(cfg, "sharding_config")

    def test_does_not_mutate_parallel_config(self):
        # Native vLLM multi-engine DP is not represented in the JAX mesh, so
        # from_vllm_config hardcodes data_parallelism=1 and has no side effect
        # on parallel_config (earlier versions normalized DP to 1 in place).
        cfg = _make_vllm_config(data_parallel_size=4)
        manager = ShardingConfigManager.initialize(cfg)

        assert manager.model_dp_size == 1
        assert cfg.parallel_config.data_parallel_size == 4  # left untouched

    def test_is_idempotent(self):
        # initialize() re-fires on every config reconstruction (it runs inside
        # VllmConfig.__post_init__, which replace()/with_hf_config re-run). The
        # guard reuses the stored parse instead of re-parsing -- proven here by
        # tampering with the stored dict: a re-parse would reset tp back to 4.
        cfg = _make_vllm_config(tensor_parallel_size=4)
        first = ShardingConfigManager.initialize(cfg)
        stored = cfg.additional_config[
            ShardingConfigManager._ADDITIONAL_CONFIG_KEY]
        stored["sharding_strategy"]["tensor_parallelism"] = 999
        second = ShardingConfigManager.initialize(cfg)
        assert first.tp_size == 4
        assert second.tp_size == 999  # reused stored parse, did not re-parse


class TestGet:

    def test_round_trips_strategy_and_derived_sizes(self):
        cfg = _make_vllm_config(
            tensor_parallel_size=2,
            additional_config={
                "sharding": {
                    "sharding_strategy": {
                        "expert_parallelism": 2,
                        "sequence_parallelism": 1,
                    }
                }
            },
        )
        original = ShardingConfigManager.initialize(cfg)
        restored = ShardingConfigManager.get(cfg)

        assert restored.sharding_strategy == original.sharding_strategy
        assert restored.tp_size == original.tp_size == 2
        assert restored.expert_size == original.expert_size == 2
        assert restored.total_devices == original.total_devices
        assert restored.total_dp_size == original.total_dp_size
        assert restored.device_indexes == original.device_indexes

    def test_before_initialize_raises(self):
        cfg = _make_vllm_config()
        with pytest.raises(AssertionError, match="not initialized"):
            ShardingConfigManager.get(cfg)

    def test_survives_deepcopy_of_config(self):
        # Storing in additional_config (a real field) means the data travels
        # with a copied config -- unlike an id()-keyed side table.
        cfg = _make_vllm_config(tensor_parallel_size=4)
        ShardingConfigManager.initialize(cfg)

        cfg_copy = copy.deepcopy(cfg)
        restored = ShardingConfigManager.get(cfg_copy)
        assert restored.tp_size == 4
        assert restored.total_devices == 4

    def test_returns_independent_managers(self):
        cfg = _make_vllm_config(tensor_parallel_size=2)
        ShardingConfigManager.initialize(cfg)
        assert ShardingConfigManager.get(cfg) is not ShardingConfigManager.get(
            cfg)


class TestDeviceIndexes:

    def test_round_trip_as_json_native_list(self):
        cfg = _make_vllm_config(
            tensor_parallel_size=2,
            additional_config={
                "sharding": {
                    "sharding_strategy": {
                        "device_indexes": [0, 1],
                    }
                }
            },
        )
        ShardingConfigManager.initialize(cfg)

        stored = cfg.additional_config[
            ShardingConfigManager._ADDITIONAL_CONFIG_KEY]["device_indexes"]
        assert stored == [0, 1]
        assert isinstance(stored, list)
        # Still serializable with device indexes present.
        json.dumps(cfg.additional_config)

        restored = ShardingConfigManager.get(cfg)
        assert restored.device_indexes == [0, 1]
        assert restored.total_devices == 2

    def test_mismatched_length_raises(self):
        # __init__ asserts total_devices == len(device_indexes).
        cfg = _make_vllm_config(
            tensor_parallel_size=4,
            additional_config={
                "sharding": {
                    "sharding_strategy": {
                        "device_indexes": [0, 1],  # 2 != 4 devices
                    }
                }
            },
        )
        with pytest.raises(AssertionError):
            ShardingConfigManager.initialize(cfg)


class TestValidate:

    # data_parallelism is hardcoded to 1 on this branch, so total_dp_size > 1
    # can only come from attention_data_parallelism. Exercise validate()
    # directly with such a strategy.
    def test_rejects_speculative_decoding_with_data_parallelism(self):
        cfg = _make_vllm_config(speculative_config=object())
        strategy = ShardingStrategy(attention_data_parallelism=2)
        with pytest.raises(ValueError, match="Speculative decoding"):
            ShardingConfigManager.validate(cfg, strategy)

    def test_rejects_lora_with_data_parallelism(self):
        cfg = _make_vllm_config(lora_config=object())
        strategy = ShardingStrategy(attention_data_parallelism=2)
        with pytest.raises(ValueError, match="LoRA"):
            ShardingConfigManager.validate(cfg, strategy)


class TestReplaceVllmConfig:
    """Integration tests against a real VllmConfig -- the actual regression.

    The original bug surfaced when a multimodal model's ``with_hf_config()``
    called vLLM's ``replace()`` while a dynamic ``sharding_config`` attribute was
    attached. ``replace()`` reconstructs the dataclass from its init fields only,
    so a non-field attribute is dropped (and ``compute_hash``'s
    ``json.dumps(additional_config)`` chokes on a non-serializable manager
    object). Storing a JSON-native dict in ``additional_config`` (a real field)
    survives both.
    """

    def test_replace_preserves_sharding_config(self):
        cfg = _make_real_vllm_config(tensor_parallel_size=4)
        ShardingConfigManager.initialize(cfg)

        replaced = vllm_replace(cfg)  # must not raise

        # additional_config is an init field, so replace() carries it forward.
        assert (ShardingConfigManager._ADDITIONAL_CONFIG_KEY
                in replaced.additional_config)
        manager = ShardingConfigManager.get(replaced)
        assert manager.tp_size == 4
        assert manager.total_devices == 4

    def test_replace_does_not_introduce_dynamic_attribute(self):
        cfg = _make_real_vllm_config()
        ShardingConfigManager.initialize(cfg)
        assert not hasattr(cfg, "sharding_config")

        replaced = vllm_replace(cfg)
        assert not hasattr(replaced, "sharding_config")

    def test_compute_hash_succeeds_after_initialize(self):
        # compute_hash does json.dumps(additional_config); storing the manager
        # object here used to raise "not JSON serializable".
        cfg = _make_real_vllm_config()
        ShardingConfigManager.initialize(cfg)

        digest = cfg.compute_hash()
        assert isinstance(digest, str) and digest
