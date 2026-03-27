# Benchmarking torchtpu-vllm

## Quick Start

```bash
# Smoke test (Qwen3-0.6B, TP=1, no quantization)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-0.6b-smoke

# Full benchmark (Qwen3-Coder-480B-FP8, TP=8, EP)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep

# Dry run (print commands without executing)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep --dry-run
```

## Available Configs

| Config | Model | TP | Quantization | Use Case |
|--------|-------|----|-------------|----------|
| `qwen3-coder-480b-fp8-tp8-ep` | Qwen3-Coder-480B-A35B-Instruct-FP8 | 8 | FP8 + EP | Primary target |
| `qwen3-30b-fp8-tp4` | Qwen3-30B-A3B-FP8 | 4 | FP8 | Smaller MoE validation |
| `qwen3-0.6b-smoke` | Qwen3-0.6B | 1 | None | Quick smoke test |

Configs are shell files in `configs/`. Add your own by copying an existing one.

## What It Does

1. Loads a config (model, TP, ISL/OSL sweep, concurrency levels)
2. Starts a `vllm serve` instance with the right flags
3. Runs `benchmark_serving.py` (from your vLLM install) for each ISL/OSL x concurrency combo
4. Saves JSON results to `benchmark_runs/<model>_tp<N>_<timestamp>/`

## Prerequisites

- vLLM installed (provides `benchmark_serving.py`)
- `tpu_inference` installed (this repo)
- TPU device available
- Model weights downloaded (or use `--load-format dummy` via config)

## Results

Results are saved to `benchmark_runs/` (gitignored). Each run creates a directory containing:
- `config.json` — benchmark parameters
- `isl<N>_osl<N>_c<N>.json` — per-combo results from benchmark_serving.py
- `server.log` — vLLM server output
- `benchmark.log` — run metadata

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` | `uniform_random` | MoE routing for consistent results |
