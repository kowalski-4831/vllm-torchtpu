# SPDX-License-Identifier: Apache-2.0
"""Unit tests for src/vllm_torchtpu/omni/omni_tpu_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.cpu_test


def _make_bare_runner(cls=OmniTPUModelRunner):
    """Instantiate a runner class without invoking TPUModelRunner.__init__."""
    runner = cls.__new__(cls)
    runner._omni_connector_initialized = False
    return runner


def test_omni_tpu_model_runner_inheritance():
    assert issubclass(OmniTPUModelRunner, TPUModelRunner)


def test_omni_tpu_model_runner_load_model():
    runner = _make_bare_runner()
    runner.model_config = MagicMock()
    runner.init_omni_connectors = MagicMock()

    with (
        patch(
            "vllm_torchtpu.omni.omni_tpu_model_runner.apply_omni_ar_patches"
        ) as mock_patch,
        patch.object(TPUModelRunner, "load_model") as mock_super_load,
    ):
        runner.load_model()
        mock_patch.assert_called_once_with()
        mock_super_load.assert_called_once_with()

    runner.init_omni_connectors.assert_called_once_with(runner.model_config)


def test_omni_tpu_model_runner_update_states():
    runner = _make_bare_runner()
    runner._omni_connector_initialized = True
    runner.cleanup_finished_request = MagicMock()
    sched_out = SimpleNamespace(finished_req_ids=["req-1"])

    with patch.object(TPUModelRunner, "_update_states", return_value="updated"):
        res = runner._update_states(sched_out)

    assert res == "updated"
    runner.cleanup_finished_request.assert_called_once_with("req-1")
