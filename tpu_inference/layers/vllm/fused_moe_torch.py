# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused MoE implementation using GMM kernel for single-device TPU.

This module provides MoE forward pass using PyTorch with GMM kernel via
pallas.custom_jax_kernel. For multi-device with sharding, see
tpu_inference/layers/vllm/fused_moe.py (JAX-based).

Key functions:
    - fused_moe_gmm: Core MoE algorithm for single device
    - activation_fn: SiLU vs SwiGLUOAI activation
"""

from typing import Optional

import jax.numpy as jnp
import torch

from tpu_inference.kernels.megablox.gmm import gmm
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _create_gmm_with_bias_fn(tiling: tuple[int, int, int], has_bias: bool):
    """Create a JAX GMM function that takes bias as input argument.

    When using pallas.custom_jax_kernel, tensor arguments must be passed
    as function arguments, not as functools.partial frozen args. This creates
    a wrapper function with the correct signature.

    Args:
        tiling: (tm, tk, tn) tile sizes
        has_bias: Whether bias will be passed

    Returns:
        JAX function compatible with custom_jax_kernel
    """
    if has_bias:

        def gmm_with_bias(lhs, rhs, group_sizes, rhs_bias):
            return gmm(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                preferred_element_type=jnp.bfloat16,
                rhs_scale=None,
                rhs_bias=rhs_bias,
                tiling=tiling,
                group_offset=jnp.array(0),
            )

        return gmm_with_bias
    else:

        def gmm_no_bias(lhs, rhs, group_sizes):
            return gmm(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                preferred_element_type=jnp.bfloat16,
                rhs_scale=None,
                rhs_bias=None,
                tiling=tiling,
                group_offset=jnp.array(0),
            )

        return gmm_no_bias


def activation_fn_torch(activation: str, x1: torch.Tensor,
                        x2: torch.Tensor) -> torch.Tensor:
    """Apply activation function to gate and up projections.

    Args:
        activation: "silu" or "swigluoai"
        x1: Gate projection [num_tokens, intermediate_size]
        x2: Up projection [num_tokens, intermediate_size]

    Returns:
        Activated hidden states [num_tokens, intermediate_size]
    """
    if activation == "silu":
        return torch.nn.functional.silu(x1) * x2
    elif activation == "swigluoai":
        return _swigluoai_torch(x1, x2)
    else:
        raise NotImplementedError(
            f"FusedMoE does not support {activation} activation")


def _swigluoai_torch(x1: torch.Tensor,
                     x2: torch.Tensor,
                     alpha: float = 1.702,
                     limit: float = 7.0) -> torch.Tensor:
    """SwiGLU-OAI activation function (GPT-OSS uses this)."""
    x1 = torch.clamp(x1, max=limit)
    x2 = torch.clamp(x2, min=-limit, max=limit)
    gated_activation = x1 * torch.sigmoid(alpha * x1)
    return gated_activation * (x2 + 1)


def _get_tiling_size(m: int, k: int, n: int, g: int) -> tuple[int, int, int]:
    """Calculate tiling sizes for GMM kernel.

    Args:
        m: Total tokens (num_tokens * topk)
        k: Input dimension (contracting)
        n: Output dimension
        g: Number of experts

    Returns:
        (tm, tk, tn) tile sizes

    Note: GMM kernel requires m % tm == 0
    """
    # tm must divide m evenly
    # Start with a reasonable default and find a divisor
    tm_candidates = [128, 64, 32, 16, 8]
    tm = 128
    for candidate in tm_candidates:
        if m % candidate == 0 and m >= candidate:
            tm = candidate
            break
    else:
        # Fallback: use m itself if small enough
        if m <= 128:
            tm = m
        else:
            tm = 128  # Will cause error if m not divisible

    tk = min(128, k)
    tn = min(128, n)
    return tm, tk, tn


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str,
) -> torch.Tensor:
    """Fused MoE forward pass using GMM kernel for single-device TPU.

    Algorithm:
    1. Route: softmax → top_k → argsort by expert
    2. Sort tokens by expert assignment
    3. Compute group_sizes via bincount
    4. GMM: x @ w1 → [tokens*topk, 2*intermediate]
    5. Activation: SiLU or SwiGLUOAI
    6. GMM: hidden @ w2 → [tokens*topk, hidden_size]
    7. Unsort, apply routing weights, sum over top_k

    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: [num_experts, hidden_size, intermediate_size * 2] (transposed for GMM)
        w2: [num_experts, intermediate_size, hidden_size] (transposed for GMM)
        w1_bias: [num_experts, 1, intermediate_size * 2] or None
        w2_bias: [num_experts, 1, hidden_size] or None
        gating_output: [num_tokens, num_experts]
        topk: Number of experts per token
        renormalize: Whether to renormalize top-k weights
        activation: "silu" or "swigluoai"

    Returns:
        Output tensor [num_tokens, hidden_size]
    """
    from torch_tpu._internal import pallas

    num_tokens, hidden_size = hidden_states.shape
    num_experts = gating_output.shape[1]
    dtype = hidden_states.dtype

    # Weight shapes: [E, in_dim, out_dim] after transpose
    _, padded_hidden_size, intermediate_size_2 = w1.shape
    _, intermediate_size, out_hidden_size = w2.shape

    # 1. Route tokens: softmax → top_k
    topk_weights = torch.softmax(gating_output.to(torch.float32), dim=-1)
    topk_weights, topk_indices = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # 2. Sort tokens by expert assignment
    topk_indices_flat = topk_indices.flatten()
    topk_argsort = torch.argsort(topk_indices_flat)
    topk_argsort_revert = torch.argsort(topk_argsort)

    token_indices = torch.arange(num_tokens,
                                 device=hidden_states.device,
                                 dtype=torch.int32).repeat_interleave(topk)
    token_indices_sorted = token_indices[topk_argsort]
    x_sorted = hidden_states[token_indices_sorted]

    # 3. Compute group_sizes
    group_sizes = torch.bincount(topk_indices_flat,
                                 minlength=num_experts).to(torch.int32)

    # 4. Pad input if needed
    if padded_hidden_size > hidden_size:
        x_sorted = torch.nn.functional.pad(
            x_sorted, (0, padded_hidden_size - hidden_size))

    # 5. GMM for w1 (gate_up projection)
    m = num_tokens * topk
    _, k1, n1 = w1.shape
    tm1, tk1, tn1 = _get_tiling_size(m, k1, n1, num_experts)

    gmm_w1_fn = _create_gmm_with_bias_fn((tm1, tk1, tn1),
                                         has_bias=(w1_bias is not None))
    gmm_w1_torch = pallas.custom_jax_kernel(gmm_w1_fn)

    if w1_bias is not None:
        hidden = gmm_w1_torch(x_sorted, w1, group_sizes, w1_bias)
    else:
        hidden = gmm_w1_torch(x_sorted, w1, group_sizes)

    # 7. Activation: split and apply
    x1, x2 = hidden.chunk(2, dim=-1)
    hidden = activation_fn_torch(activation, x1, x2)

    # 8. GMM for w2 (down projection)
    _, k2, n2 = w2.shape
    tm2, tk2, tn2 = _get_tiling_size(m, k2, n2, num_experts)

    gmm_w2_fn = _create_gmm_with_bias_fn((tm2, tk2, tn2),
                                         has_bias=(w2_bias is not None))
    gmm_w2_torch = pallas.custom_jax_kernel(gmm_w2_fn)

    if w2_bias is not None:
        output = gmm_w2_torch(hidden, w2, group_sizes, w2_bias)
    else:
        output = gmm_w2_torch(hidden, w2, group_sizes)

    # 10. Finalize: unsort, apply weights, sum
    output = output[topk_argsort_revert].reshape(num_tokens, topk, -1)
    output = output * topk_weights.unsqueeze(-1)
    output = output.sum(dim=1)

    logger.debug(
        f"[MoE DEBUG] x_sorted: shape={x_sorted.shape}, mean={x_sorted.float().mean():.6f}, absmax={x_sorted.float().abs().max():.6f}"
    )
    logger.debug(
        f"[MoE DEBUG] hidden (after w1+act): shape={hidden.shape}, mean={hidden.float().mean():.6f}, absmax={hidden.float().abs().max():.6f}"
    )
    logger.debug(
        f"[MoE DEBUG] output (before slice): shape={output.shape}, mean={output.float().mean():.6f}, absmax={output.float().abs().max():.6f}"
    )

    # 11. Remove padding
    return output[:, :hidden_size]
