# SPDX-License-Identifier: Apache-2.0
"""Unit tests for src/vllm_torchtpu/omni/omni_tpu_ar_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

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
    with patch.object(
        OmniTPUModelRunner, "forward_model", return_value=(plain, None)
    ):
        out, aux = runner.forward_model()
    assert out is plain
    assert aux is None


def test_omni_tpu_ar_model_runner_sample_tokens():
    runner = _make_bare_ar_runner()

    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value="raw_tokens"):
        runner._omni_connector_initialized = False
        assert runner.sample_tokens() == "raw_tokens"

        runner._omni_connector_initialized = True
        assert runner.sample_tokens() == ("connector_attached", "raw_tokens")


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
