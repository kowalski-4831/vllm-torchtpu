# SPDX-License-Identifier: Apache-2.0
"""Unit tests for src/vllm_torchtpu/omni/omni_tpu_ar_model_runner.py."""

from unittest.mock import patch

import pytest

from .mock_omni import _MockOmniConnectorModelRunnerMixin  # noqa: F401
from vllm_torchtpu.omni.omni_tpu_ar_model_runner import OmniTPUARModelRunner
from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner

pytestmark = pytest.mark.cpu_test


def _make_bare_ar_runner():
    runner = OmniTPUARModelRunner.__new__(OmniTPUARModelRunner)
    runner._omni_connector_initialized = False
    return runner


def test_omni_tpu_ar_model_runner_inheritance():
    assert issubclass(OmniTPUARModelRunner, OmniTPUModelRunner)
    assert issubclass(OmniTPUARModelRunner, _MockOmniConnectorModelRunnerMixin)


def test_omni_tpu_ar_model_runner_sample_tokens():
    runner = _make_bare_ar_runner()

    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value="raw_tokens"):
        runner._omni_connector_initialized = False
        assert runner.sample_tokens() == "raw_tokens"

        runner._omni_connector_initialized = True
        assert runner.sample_tokens() == ("connector_attached", "raw_tokens")
