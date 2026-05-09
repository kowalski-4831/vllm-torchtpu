#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Full benchmark sweep for Qwen3-Coder-480B-A35B-Instruct-FP8 with TP=8, EP.
MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024 8192:1024 1024:8192 1024:1 8192:1"
CONCURRENCY_OPTIONS="1 16 32 64 128 256"
RANDOM_RANGE_RATIO="0.8"

# Switch attention from default RPA v3 to the experimental batched RPA kernel
# under tpu_inference/kernels/experimental/batched_rpa/. Same flag as the
# nightly short config. Set to 0 to ablate.
export USE_BATCHED_RPA_KERNEL=1
