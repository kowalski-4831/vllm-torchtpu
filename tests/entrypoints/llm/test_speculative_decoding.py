# Copyright 2025 Google LLC
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

from __future__ import annotations

import random
import string
import time

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.metrics.reader import Counter

from tpu_inference import tpu_info


def _is_v7x():
    return (tpu_info.get_tpu_type() or "").startswith("tpu7x")


def _get_tensor_parallel_size():
    if _is_v7x():
        return 2
    return 1


def get_ngram_test_prompts():
    num_prompts = 10
    prompts = []
    for _ in range(num_prompts):
        w = random.choice(list(string.ascii_lowercase))
        prompts.append(
            f"Keep repeating: {w} {w} {w} {w} {w} {w} {w} {w} {w} {w}")
    return prompts


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0,
                          max_tokens=16,
                          ignore_eos=True,
                          repetition_penalty=1,
                          frequency_penalty=0,
                          presence_penalty=0,
                          min_p=0,
                          logprobs=None)


@pytest.fixture
def model_name():
    return "Qwen/Qwen3-0.6B"


def _test_correctness_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
    speculative_config: dict,
    max_num_seqs: int = 4,
    extra_kwargs: dict | None = None,
):
    with monkeypatch.context():
        test_prompts = get_ngram_test_prompts()

        kwargs = {
            "max_model_len": 256,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": _get_tensor_parallel_size(),
            "async_scheduling": False,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        # 1. Reference LLM Run
        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        del ref_llm
        cleanup_dist_env_and_memory()

        # Waiting for TPUs to be fully released.
        time.sleep(15)

        # 2. Speculative LLM Run
        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       **kwargs)
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)

        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        assert misses == 0
        del spec_llm
        cleanup_dist_env_and_memory()


@pytest.mark.timeout(500)
def test_ngram_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    _test_correctness_helper(
        monkeypatch, sampling_config, model_name, {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        })


def _test_performance_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    speculative_config: dict,
    min_acceptance_rate: float,
    max_num_seqs: int = 4,
    model_name: str = "Qwen/Qwen3-0.6B",
    extra_kwargs: dict | None = None,
):
    with monkeypatch.context():
        test_prompts = get_ngram_test_prompts()

        kwargs = {
            "max_model_len": 256,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": _get_tensor_parallel_size(),
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "disable_log_stats": False,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       **kwargs)

        spec_llm.generate(test_prompts, sampling_config)

        metrics = spec_llm.get_metrics()
        num_draft_tokens = num_accepted_tokens = 0
        acceptance_rate = 0.0
        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_draft_tokens":
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        if num_draft_tokens > 0:
            acceptance_rate = num_accepted_tokens / num_draft_tokens
            print(f"Acceptance rate: {acceptance_rate:.2%}")
            print("num_accepted_tokens:" + str(num_accepted_tokens))
            print("num_draft_tokens:" + str(num_draft_tokens))

        del spec_llm
        cleanup_dist_env_and_memory()

        assert num_draft_tokens > 0, "Draft tokens should be greater than 0."
        assert acceptance_rate >= min_acceptance_rate, f"Expected at least {min_acceptance_rate:.2%} acceptance rate for {speculative_config['method']}, got {acceptance_rate:.2%}"


@pytest.mark.timeout(500)
def test_ngram_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    _test_performance_helper(monkeypatch,
                             sampling_config, {
                                 "method": "ngram",
                                 "prompt_lookup_max": 2,
                                 "prompt_lookup_min": 2,
                                 "num_speculative_tokens": 4,
                             },
                             min_acceptance_rate=0.85)


def test_sd_rejects_non_greedy(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str = "Qwen/Qwen3-0.6B",
):
    # Currently only greedy is supported.
    with monkeypatch.context():
        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
                "num_speculative_tokens": 3,
            },
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=False,
        )
        try:
            non_greedy = SamplingParams(temperature=0.7)
            with pytest.raises(NotImplementedError, match="greedy"):
                spec_llm.generate(get_ngram_test_prompts(), non_greedy)
        finally:
            del spec_llm
            cleanup_dist_env_and_memory()


def _force_multi_chunk_cap(self, cap: int = 2):
    runner = self.model_runner
    runner.num_reqs_max_model_len = cap
    if runner.num_reqs_most_model_len is not None:
        runner.num_reqs_most_model_len = cap


@pytest.mark.timeout(500)
def test_sd_correctness_greedy_multi_chunk(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    with monkeypatch.context() as mp:
        # collective_rpc below sends a Python function to TPU workers,
        # which requires pickle fallback in vLLM's IPC encoder.
        mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        test_prompts = get_ngram_test_prompts()
        kwargs = dict(
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=False,
        )

        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        del ref_llm
        cleanup_dist_env_and_memory()
        time.sleep(15)

        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
                "num_speculative_tokens": 3,
            },
            **kwargs,
        )
        # Force the runner's while-loop in execute_model to iterate at
        # least twice for any batch larger than 2.
        spec_llm.llm_engine.collective_rpc(_force_multi_chunk_cap, args=(2, ))

        spec_outputs = spec_llm.generate(test_prompts, sampling_config)

        misses = 0
        for ref, spec in zip(ref_outputs, spec_outputs):
            if ref.outputs[0].text != spec.outputs[0].text:
                misses += 1
                print(f"ref:  {ref.outputs[0].text}")
                print(f"spec: {spec.outputs[0].text}")
        assert misses == 0

        del spec_llm
        cleanup_dist_env_and_memory()
