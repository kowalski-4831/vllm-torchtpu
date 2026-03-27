#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: Qwen3-0.6B smoke test (no quantization, TP=1)
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="256:256"
CONCURRENCY_OPTIONS="1 4"
MAX_NUM_BATCHED_TOKENS=1024
