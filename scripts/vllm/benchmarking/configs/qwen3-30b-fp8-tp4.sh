#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: Qwen3-30B-A3B-FP8 with TP=4
MODEL="Qwen/Qwen3-30B-A3B-FP8"
TENSOR_PARALLELISM=4
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="512:512 1024:1"
CONCURRENCY_OPTIONS="1 16 64"
MAX_NUM_BATCHED_TOKENS=8192
