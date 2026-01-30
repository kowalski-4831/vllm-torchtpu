# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Implements profiling for vLLM on TPU VMs using the JAX profiler.
# NOTE: you will need the tensorboard-plugin-profile python package to
# visualize the results in TensorBoard.
# Please see docs/profiler.md for more details.
# Usage example for prefilling 1 request of 1024 tokens:
# python3 examples/tpu_profiling.py --input-len 1024 --output-len 1   --batch-size 1
# Usage example for decoding 256 requests of 1 token each:
# python3 examples/tpu_profiling.py --input-len 1 --output-len 1 --batch-size=256

import argparse
import dataclasses
import os
import time

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils.argparse_utils import FlexibleArgumentParser

DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))


def main(args: argparse.Namespace):
    print(args)

    # Profile
    profile_dir = args.profile_result_dir
    print(f"Profiling (results will be saved to '{profile_dir}')...")
    os.environ["VLLM_TORCH_PROFILER_DIR"] = profile_dir

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)

    sp_ttft = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1)
    print(sp_ttft)

    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    # We will simply perform the measurement on the "Profile Iterations" by
    # 1. Running a prompt with max_tokens=1 to measure TTFT (approx).
    # 2. Running a prompt with max_tokens=N to measure Total Time.
    # TPOT = (Total - TTFT) / (N - 1)

    # Warmup
    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
    for _ in tqdm(range(args.num_iters_warmup), desc="TTFT Warmup iterations"):
        llm.generate(dummy_prompts, sampling_params=sp_ttft, use_tqdm=False)

    print("\nStarting Benchmark...")

    # 1. Measure TTFT (Prefill + 1 decode step)
    ttft_latencies = []
    for _ in range(args.num_iters):
        start = time.perf_counter()
        llm.generate(dummy_prompts, sampling_params=sp_ttft, use_tqdm=False)
        end = time.perf_counter()
        ttft_latencies.append(end - start)

    avg_ttft_sec = np.mean(ttft_latencies)

    # 2. Measure Total Time (Prefill + N decode steps)
    # Enable tracing on server
    llm.start_profile()
    if DELAY_MS == 0:
        time.sleep(1.0)

    total_latencies = []
    for _ in range(args.num_iters):
        start = time.perf_counter()
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end = time.perf_counter()
        total_latencies.append(end - start)

    avg_total_sec = np.mean(total_latencies)
    llm.stop_profile()

    num_decode_tokens = args.output_len - 1
    if num_decode_tokens > 0:
        decode_time = avg_total_sec - avg_ttft_sec
        avg_tpot_sec = decode_time / num_decode_tokens
        decode_throughput = (args.batch_size * num_decode_tokens) / decode_time
    else:
        avg_tpot_sec = 0.0
        decode_throughput = 0.0

    print(f"\n{'='*40}")
    print(f"Benchmark Results: {args.model}")
    print(
        f"Config: BS={args.batch_size}, InLen={args.input_len}, OutLen={args.output_len}"
    )
    print(f"{'-'*40}")
    print(f"TTFT (ms):          {avg_ttft_sec * 1000:.2f}")
    print(f"TPOT (ms):          {avg_tpot_sec * 1000:.2f}")
    print(f"Throughput (tok/s): {decode_throughput:.2f} (decode only)")
    print(f"Total Latency (s):  {avg_total_sec:.4f}")
    print(f"{'='*40}\n")

    return


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=5,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="Number of iterations to run for profiling.",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default="profiles",
        help=("path to save the JAX profiler output. Can be visualized "
              "with ui.perfetto.dev, Tensorboard, or XProf"),
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
