"""
End-to-end tests for vLLM LLM generation API on TPU.

This module tests the LLM.generate() API with various configurations using
the Qwen3-0.6B model, under both eager and compiled execution modes.

Tests include:
- Multiple sampling parameters
- Priority queue handling
- Max model length enforcement
- Statistics logging

Run with: pytest tests/entrypoints/llm/test_generate.py -v
"""

import weakref

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]


def _get_local_tpu_chip_count() -> int:
    try:
        from tpu_info import device as tpu_device

        _, chips = tpu_device.get_local_chips()
        return len(chips)
    except Exception as e:
        pytest.skip(f"Unable to detect local TPU chip count: {e}")


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=64,
        max_model_len=64,
        tensor_parallel_size=1,
        # TODO: Reduce GPU memory utilization after KV cache optimization.
        # For reference, this value is set to 0.10 in the equivalent test in vllm.
        gpu_memory_utilization=0.6,
        enforce_eager=False,
        disable_log_stats=False,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def llm_tp_local_size():
    tp_size = _get_local_tpu_chip_count()

    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=64,
        max_model_len=64,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        disable_log_stats=False,
    )

    yield weakref.proxy(llm), tp_size

    del llm

    cleanup_dist_env_and_memory()


def test_multiple_sampling_params(llm: LLM):
    """Test multiple sampling parameters."""
    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(PROMPTS, sampling_params=None)
    assert len(PROMPTS) == len(outputs)


def test_multiple_priority(llm: LLM):
    # Generate works when priority is None
    outputs = llm.generate(PROMPTS, sampling_params=None, priority=None)
    assert len(PROMPTS) == len(outputs)

    # Generate works when length of priority is same as the len(PROMPTS)
    outputs = llm.generate(PROMPTS,
                           sampling_params=None,
                           priority=[0] * len(PROMPTS))
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the length of priority does not match the length of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS,
                               sampling_params=None,
                               priority=[0] * (len(PROMPTS) - 1))

    # Exception raised, if the priority list is empty
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, sampling_params=None, priority=[])


def test_max_model_len(llm: LLM):
    max_model_len = llm.model_config.max_model_len
    sampling_params = SamplingParams(max_tokens=max_model_len + 10)
    outputs = llm.generate(PROMPTS, sampling_params)
    for output in outputs:
        num_total_tokens = len(output.prompt_token_ids) + len(
            output.outputs[0].token_ids)
        # Total tokens must not exceed max_model_len.
        # It can be less if generation finishes due to other reasons (e.g., EOS)
        # before reaching the absolute model length limit.
        assert num_total_tokens <= max_model_len


def test_log_stats(llm: LLM):
    outputs = llm.generate(PROMPTS, sampling_params=None)
    assert all(output.metrics is not None for output in outputs)


def test_generate_with_tp_equal_local_tpu_count(llm_tp_local_size):
    llm, tp_size = llm_tp_local_size

    outputs = llm.generate(
        ["Tensor parallel test prompt."],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )

    assert tp_size >= 1
    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
