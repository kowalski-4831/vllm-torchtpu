#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: Qwen3-Coder-30B-A3B-Instruct (MoE, TP=8 + EP on v7x-8).
# Model is not pre-quantized (no FP8 weights); KV stays fp8 via runner default.
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
