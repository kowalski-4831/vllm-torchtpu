#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly perf benchmark for Kimi-K3 on the multihost TPU v7x-32 pod
# (4 hosts x 8 chips): TP=32 with EP. Single-request latency probe at
# 8k in / 1k out. Baseline calibrated on the pod from two dev-pipeline
# runs with run-to-run spread <= 0.3% on every gated metric — the
# default 5% perf tolerance applies.
MODEL="moonshotai/Kimi-K3"
# v7x-32 hosts have no data disk that fits the 1561 GB checkpoint, so
# weights stream directly from GCS via --load-format runai_streamer.
# --enable-ep-weight-filter (model_loader_patches.py, #373) keeps that
# tractable: each rank streams only the byte ranges of its own experts
# (28/896 at EP32) plus dense weights, instead of the full checkpoint —
# full-checkpoint streaming accumulated >1 TB of host anonymous memory
# and drew Ray OOM kills (dev builds 82/83/85).
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# No --quantization: the FP8 checkpoint self-describes (compressed-tensors).
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# 64 sequential bs=1 requests (~42 s each, ~45 min of bench time) give a
# stabler median for the gated metrics; checkpoint stream + compile still
# dominate the step's wall clock. check_regression.py fails a cell when
# completed < 0.95 x baseline completed, so the floor is 60.8 and up to 3
# transient request failures do not fail the nightly on the single
# tpu_v7x_32 agent.
NUM_PROMPTS=64

# Disable random length jitter: benchmark requests use exact nominal
# 8192:1024 lengths, so TTFT/TPOT medians are comparable run to run.
RANDOM_RANGE_RATIO="0.0"

# Server configuration (mirrors the proven multihost bring-up recipe):
# - max-model-len=9216: exact nominal length for 8192 in + 1024 out.
# - max-num-batched-tokens=512 / max-num-seqs=1: bs=1 latency regime.
MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=512
MAX_NUM_SEQS=1

# Startup budget covers the GCS stream plus a cold XLA compile
# (VLLM_DISABLE_COMPILE_CACHE=1 in the multihost containers).
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
# The proven recipe serves with vllm's default KV cache dtype and
# synchronous scheduling; both are recorded in the run's config.json.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

# Kimi-K3 is an MLA model; ATTENTION_BACKEND default CUSTOM applies.

# Serve flags:
# --trust-remote-code: the HF repo ships custom code (TikTokenTokenizer).
# --language-model-only + --limit-mm-per-prompt: K3 is image-text-to-text;
#   serve just the LM (the vision tower has no TPU path here).
# --model-loader-extra-config memory_limit=16GiB: caps the runai streamer's
#   per-rank host buffer so 32 ranks fit host RAM.
# --compilation-config compile_sizes [16,512]: the only token-count
#   buckets this bench reaches — prefill chunks are exactly 512 (8192 is
#   a multiple of max-num-batched-tokens, no remainder) and bs=1 decode
#   pads to 16. 64/256 were compiled cold every run (~5.6 min on the pod)
#   and never executed.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,512]}"

# The bench client tokenizes from the dir the server pulled from GCS
# (run_benchmarks.sh derives --tokenizer from MODEL_URI); the harness
# passes --trust-remote-code to the client unconditionally, which covers
# the custom tokenizer code in that dir.

# Greedy decoding for bench requests.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
