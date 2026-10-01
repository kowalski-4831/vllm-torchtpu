# SPDX-License-Identifier: Apache-2.0
"""Unit tests for src/vllm_torchtpu/omni/tpu_ar_worker.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from .mock_omni import _MockOmniWorkerMixin
from vllm_torchtpu.omni.omni_tpu_ar_model_runner import OmniTPUARModelRunner
from vllm_torchtpu.omni.platform import OmniTpuPlatform
from vllm_torchtpu.omni.tpu_ar_worker import TPUARWorker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_worker import TPUWorker

pytestmark = pytest.mark.cpu_test


def test_platform_ar_worker_cls():
    assert (
        OmniTpuPlatform.get_omni_ar_worker_cls()
        == "vllm_torchtpu.omni.tpu_ar_worker.TPUARWorker"
    )


def test_tpu_ar_worker_hierarchy_and_runner_cls():
    assert issubclass(TPUARWorker, TPUWorker)
    assert issubclass(TPUARWorker, _MockOmniWorkerMixin)
    assert TPUWorker.model_runner_cls is TPUModelRunner
    assert TPUARWorker.model_runner_cls is OmniTPUARModelRunner


def test_tpu_worker_uses_model_runner_cls():
    worker = TPUARWorker.__new__(TPUARWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype="bfloat16", seed=0),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        additional_config={},
    )
    worker.devices = ["tpu:0"]
    worker.profile_rank = 0
    worker.local_rank = 0
    worker.rank = 0

    dummy_runner = MagicMock()
    with patch.object(TPUARWorker, "model_runner_cls", return_value=dummy_runner) as mock_cls:
        assert mock_cls is TPUARWorker.model_runner_cls
        runner = worker.model_runner_cls(
            worker.vllm_config,
            worker.devices[0],
            profiler_rank=worker.profile_rank,
            is_driver_worker=True,
            is_first_rank=True,
            is_last_rank=True,
        )
        assert runner is dummy_runner
        mock_cls.assert_called_once()
