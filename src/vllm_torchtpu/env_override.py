# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

# Map VLLM_XLA_CACHE_PATH to TorchTPU compilation cache environment variables
vllm_xla_cache_path = os.getenv("VLLM_XLA_CACHE_PATH")
if vllm_xla_cache_path:
    # Route Tier-3 native cache to a subdirectory under VLLM_XLA_CACHE_PATH
    os.environ.setdefault("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
                          os.path.join(vllm_xla_cache_path, "torch_tpu_tier3"))
    # Enable Tier-2 cache in memory (required by Tier-3)
    os.environ.setdefault("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE",
                          "tpu_tier2_cache")
