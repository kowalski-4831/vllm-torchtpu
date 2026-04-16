# Qwen3 480B Benchmark Recipe

Use one terminal for the server and another for the benchmark client.

## Server

TorchTPU:

```bash
MODEL_IMPL_TYPE=vllm vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --max-model-len=16589 --max-num-batched-tokens=8192 --max-num-seqs=512 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization=0.9 --tensor-parallel-size=8 \
  --async-scheduling --enable-expert-parallel
```

TorchAX:

```bash
MODEL_IMPL_TYPE=vllm vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --max-model-len=16589 --max-num-batched-tokens=8192 --max-num-seqs=512 \
  --kv-cache-dtype=fp8 --no-enable-prefix-caching \
  --gpu-memory-utilization=0.9 --tensor-parallel-size=8 \
  --async-scheduling --enable-expert-parallel
```

## Client

Run `vllm bench serve` against the running server. Replace `64` with the concurrency you want.

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-filename isl1024_osl1024_c64.json
```

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-filename isl8192_osl1024_c64.json
```

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 8192 \
  --random-range-ratio 0.8 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-filename isl1024_osl8192_c64.json
```
