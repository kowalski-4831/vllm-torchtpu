#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly sweep for Qwen3.5-35B-A3B-FP8 with TP=4.
MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
TENSOR_PARALLELISM=4
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.9

# Route through the fused Conv1D+GDN v3 kernel: the non-fused
# chunked_kernel_pd path hits a whole-graph libtpu scheduling regression on
# this model that the single-custom_call fused kernel avoids. Scoped to this
# config because the same override significantly regresses the 397B DP=8 EP
# configuration.
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd

# Extra arguments for vllm serve
EXTRA_SERVE_ARGS="--block-size 256 --limit-mm-per-prompt {\"image\":0,\"video\":0} --default-chat-template-kwargs {\"enable_thinking\":false}"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EVAL_TOLERANCE="0.02"
