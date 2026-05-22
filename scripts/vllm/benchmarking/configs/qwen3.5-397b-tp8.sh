#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly sweep for Qwen3.5-397B-A17B with TP=8.
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024 1024:8192 8192:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"

# Use batched RPA kernel if applicable
export USE_BATCHED_RPA_KERNEL=1

# Set headroom for TPU KV cache
export TPU_KV_CACHE_HEADROOM_MIB=32768

# Control GPU memory utilization
GPU_MEMORY_UTILIZATION=0.9

# Extra arguments for vllm serve
EXTRA_SERVE_ARGS="--block-size 256 --limit-mm-per-prompt {\"image\":0,\"video\":0}"
