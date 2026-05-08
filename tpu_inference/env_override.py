# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

# Ask XLA to use conservative collective-matmul fusion after SPMD
# partitioning. These modes let the TPU compiler fuse all-gather and
# reduce-scatter collectives with nearby matmuls when the pattern is
# safe, reducing exposed tensor-parallel communication overhead.
#
# Use setdefault so an explicit LIBTPU_INIT_ARGS from the caller remains
# authoritative for ablations or stack-specific flags.
os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    " ".join((
        "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative",
        "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative",
    )),
)
