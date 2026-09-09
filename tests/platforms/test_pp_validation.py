# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_torchtpu.platforms.pp_validation import \
    validate_pipeline_parallel_config


def _config(pp=2,
            async_scheduling=False,
            speculative=None,
            kv_transfer=None,
            tp=1,
            layer_types=None):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp,
                                        tensor_parallel_size=tp),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative,
        kv_transfer_config=kv_transfer,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(
            layer_types=layer_types)),
    )


def test_single_stage_accepts_everything():
    validate_pipeline_parallel_config(
        _config(pp=1,
                async_scheduling=True,
                speculative=object(),
                kv_transfer=object()))


def test_pipeline_parallel_accepts_sync_scheduling():
    validate_pipeline_parallel_config(_config(pp=2))


def test_rejects_async_scheduling_and_names_the_flag():
    with pytest.raises(NotImplementedError,
                       match="async scheduling.*--no-async-scheduling"):
        validate_pipeline_parallel_config(_config(pp=2, async_scheduling=True))


def test_rejects_speculative_decoding():
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        validate_pipeline_parallel_config(_config(pp=2, speculative=object()))


def test_rejects_kv_transfer_connectors():
    raiden = SimpleNamespace(kv_connector="TPURaidenConnector")
    with pytest.raises(NotImplementedError, match="KV transfer connectors"):
        validate_pipeline_parallel_config(_config(pp=2, kv_transfer=raiden))


def test_rejects_tensor_parallel_stages():
    with pytest.raises(NotImplementedError, match="tensor-parallel-size 1"):
        validate_pipeline_parallel_config(_config(pp=2, tp=4))


def test_rejects_the_legacy_ray_executor(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0")
    with pytest.raises(NotImplementedError,
                       match="legacy Ray.*VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1"):
        validate_pipeline_parallel_config(_config(pp=2))


def test_accepts_the_ray_v2_executor(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1")
    validate_pipeline_parallel_config(_config(pp=2))
