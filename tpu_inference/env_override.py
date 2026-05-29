# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

# Disable forced graph breaks for collectives by default.
# Commit 477f611f in torch_tpu made the collective-ops graph-break guardrail
# active by default on module import in non-absl environments, causing a
# fatal Dynamo Unsupported crash during startup in vLLM (due to fullgraph=True).
# This sets TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false by default
# to allow functional collectives to compile under SPMD, which is safe and
# required for Tensor Parallelism.
os.environ["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"] = "false"
