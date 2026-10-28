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
"""Unit tests for src/vllm_torchtpu/omni/omni_tpu_ar_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.omni_tpu_ar_model_runner import OmniTPUARModelRunner
from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

pytestmark = pytest.mark.omni


def _make_bare_ar_runner(connector_initialized=True):
    runner = OmniTPUARModelRunner.__new__(OmniTPUARModelRunner)
    runner._omni_connector_initialized = connector_initialized
    # The real attach reads connector state this bare runner does not have;
    # stand in with a tag so the tests can see where the output was routed.
    runner.attach_omni_connector_output = lambda output: ("connector_attached", output)
    return runner


def _async_output(inner):
    return AsyncTPUModelRunnerOutput(
        model_runner_output=inner,
        copy_state=None,
        discard_sampled_tokens_req_indices=[],
    )


def test_omni_tpu_ar_model_runner_inheritance():
    # The AR runner only adds the AR hooks; the mixin comes from the base.
    assert OmniTPUARModelRunner.__bases__ == (OmniTPUModelRunner,)
    assert issubclass(OmniTPUARModelRunner, OmniConnectorModelRunnerMixin)


def test_omni_tpu_ar_model_runner_forward_model():
    runner = _make_bare_ar_runner()

    # 1. Extracts text_hidden_states from multimodal output
    hidden = torch.randn(4, 16)
    multimodal_out = SimpleNamespace(text_hidden_states=hidden, extra="mm")
    with patch.object(
        OmniTPUModelRunner, "forward_model", return_value=(multimodal_out, "aux")
    ):
        out, aux = runner.forward_model()
    assert out is hidden
    assert aux == "aux"

    # 2. Passes plain tensor output through directly
    plain = torch.randn(4, 16)
    with patch.object(OmniTPUModelRunner, "forward_model", return_value=(plain, None)):
        out, aux = runner.forward_model()
    assert out is plain
    assert aux is None


def test_sample_tokens_sync_output():
    runner = _make_bare_ar_runner()
    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value="raw_tokens"):
        assert runner.sample_tokens() == ("connector_attached", "raw_tokens")


def test_sample_tokens_without_connectors_is_passthrough():
    runner = _make_bare_ar_runner(connector_initialized=False)
    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value="raw_tokens"):
        assert runner.sample_tokens() == "raw_tokens"


def test_sample_tokens_async_output_attaches_to_inner_output():
    """Under async scheduling the engine reads get_output(), not the wrapper."""
    runner = _make_bare_ar_runner()
    inner = SimpleNamespace(sampled_token_ids=None)
    async_output = _async_output(inner)

    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value=async_output):
        result = runner.sample_tokens()

    assert result is async_output
    assert result._model_runner_output == ("connector_attached", inner)


def test_sample_tokens_none_is_passthrough():
    runner = _make_bare_ar_runner()
    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value=None):
        assert runner.sample_tokens() is None


def test_execute_model_two_phase_path_defers_to_sample_tokens():
    runner = _make_bare_ar_runner()
    sched_out = SimpleNamespace(total_num_scheduled_tokens=4)
    with patch.object(OmniTPUModelRunner, "execute_model", return_value=None) as m:
        assert runner.execute_model(sched_out) is None
    m.assert_called_once_with(sched_out, None)


def test_execute_model_zero_token_early_return_attaches_connector_output():
    """A zero-token step returns from execute_model and skips sample_tokens()."""
    runner = _make_bare_ar_runner()
    sched_out = SimpleNamespace(total_num_scheduled_tokens=0)
    with patch.object(OmniTPUModelRunner, "execute_model", return_value="empty_output"):
        assert runner.execute_model(sched_out) == ("connector_attached", "empty_output")


def test_omni_tpu_ar_model_runner_mixed_dim_mm_slicing():
    runner = _make_bare_ar_runner()
    runner.supports_mm_inputs = True

    vid_emb, aud_emb = torch.ones(6, 32), torch.ones(4, 8)
    vid_emb.modality, aud_emb.modality = "video", "audio"
    mask = torch.ones(10, dtype=torch.bool)

    with patch.object(
        OmniTPUModelRunner,
        "_gather_mm_embeddings",
        return_value=([vid_emb, aud_emb], mask),
    ):
        idx_list, out_mask = runner._gather_mm_embeddings("sched_out")
    assert len(idx_list) == 1 and torch.equal(idx_list[0], torch.arange(10))
    assert out_mask is mask

    captured_mm = []

    def _dummy_embed_input_ids(ids, **kwargs):
        captured_mm.append(kwargs.get("multimodal_embeddings"))
        return torch.zeros(ids.shape[0], 8)

    runner.model = SimpleNamespace(embed_input_ids=_dummy_embed_input_ids)
    _, embeds_out = runner._get_model_inputs(
        torch.zeros(5, dtype=torch.long),
        ([torch.arange(4, 9, dtype=torch.int64)], torch.ones(5, dtype=torch.bool)),
    )
    assert embeds_out.shape == (5, 8) and len(captured_mm[0]) == 2
    assert captured_mm[0][0].shape == (2, 32)
    assert captured_mm[0][0].modality == "video"
    assert captured_mm[0][1].shape == (3, 8)
    assert captured_mm[0][1].modality == "audio"
