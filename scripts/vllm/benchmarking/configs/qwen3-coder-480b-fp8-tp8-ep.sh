#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: Qwen3-Coder-480B-A35B-Instruct-FP8 with TP=8, EP
MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024 8192:1024 1024:8192 1024:1 8192:1"
CONCURRENCY_OPTIONS="1 16 32 64 128 256"
RANDOM_RANGE_RATIO="0.8"
# Uncomment to cap compile bucket sizes (avoids host RAM OOM during compilation)
# MAX_NUM_BATCHED_TOKENS=1024
