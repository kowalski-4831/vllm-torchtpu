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
    - fused_moe_gmm: Core MoE algorithm for single device (uses GMM kernel)
    - activation_fn: SiLU vs SwiGLUOAI activation
"""

from typing import Optional

import torch
from torch_tpu._internal import pallas

from tpu_inference.kernels.megablox.gmm import gmm
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _pallas_gmm_kernel(tiling):

    def impl(lhs, rhs, gs, s=None, b=None):
        return gmm(lhs,
                   rhs,
                   gs,
                   preferred_element_type=lhs.dtype,
                   tiling=tiling,
                   rhs_scale=s,
                   rhs_bias=b,
                   group_offset=0)

    return impl


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


def _round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
    """
    Rounds the given integer `x` up to the nearest multiple of 128, without
    exceeding the specified `limit`.

    If `x` is less than or equal to 128, returns 128.
    If `x` is less than `limit`, returns the smallest multiple of 128 greater
    than or equal to `x`.
    If `x` is greater than or equal to `limit`, searches for the largest
    multiple of 128 less than or equal to `limit` (down to 512) that divides `x`
    evenly, and returns it.
    If no such candidate is found, returns `limit`.

    Args:
        x (int): The integer to round up.
        limit (int): The upper bound (must be a multiple of 128).

    Returns:
        int: The rounded value according to the rules above.

    Raises:
        AssertionError: If `limit` is less than 128 or not a multiple of 128.
    """
    assert limit >= 128 and limit % 128 == 0
    if x <= 128:
        return 128
    if x < limit:
        return (x + 127) // 128 * 128
    for candidate in range(limit, 511, -128):
        if x % candidate == 0:
            return candidate
    return limit


def _get_tiling_size_for_gmm_kernel(m: int, k: int, n: int,
                                    g: int) -> tuple[int, int, int]:
    """
    Calculate optimal tiling sizes for a GMM kernel in a Mixture of Experts
    (MoE) setting.

    Args:
        m (int): The total number of tokens.
        n (int): The output feature dimension.
        k (int): The input feature dimension.
        g (int): The number of experts.

    Returns:
        tuple[int, int, int]: A tuple (tm, tk, tn)
    """

    # TODO(Chengji): increase the upper limit tiling size of m when we can set
    # the vmem size to be used for gmm kernel.
    # NOTE: In average each expert has m // g tokens, but as it might be
    # unbalanced, here we doubled the token size when choosing tiling size of m.
    # 2m//g can be either greater or less than 512. If there are 32 tokens and
    # topk=2, m=topk * num_tokens=64, in this case, 2*m//g will be less than
    # 512.
    tm = _round_up_to_multiple_of_128_within_limit(2 * m // g, 512)
    tm = min(tm, m)  # there's a requirement that m % tm == 0
    # k/n correspond to n_input_features/n_output_features in the matmul so they
    # are normally greater than 2048, unless the num shards is large.
    tk = _round_up_to_multiple_of_128_within_limit(k, 2048)
    tn = _round_up_to_multiple_of_128_within_limit(n, 2048)
    return tm, tk, tn


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
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

    # 3. Compute group_sizes (how many tokens go to each expert)
    # Using one-hot + sum instead of bincount - more TPU-friendly
    # TODO: Debug with bincount to file a bug for the TorchTPU or XLA team
    one_hot = torch.nn.functional.one_hot(topk_indices_flat.to(torch.int64),
                                          num_classes=num_experts)
    group_sizes = one_hot.sum(dim=0).to(torch.int32)

    # 4. Pad input if needed
    if padded_hidden_size > hidden_size:
        x_sorted = torch.nn.functional.pad(
            x_sorted, (0, padded_hidden_size - hidden_size))

    # 5. GMM for w1 (gate_up projection)
    m = num_tokens * topk
    _, k1, n1 = w1.shape
    tm1, tk1, tn1 = _get_tiling_size_for_gmm_kernel(m, k1, n1, num_experts)

    w1_wrapper_impl = _pallas_gmm_kernel((tm1, tk1, tn1))
    gmm_w1_torch = pallas.custom_jax_kernel(w1_wrapper_impl)

    hidden = gmm_w1_torch(x_sorted, w1, group_sizes, w1_scale, w1_bias)

    # 7. Activation: split and apply
    x1, x2 = hidden.chunk(2, dim=-1)
    hidden = activation_fn_torch(activation, x1, x2)

    # 8. GMM for w2 (down projection)
    _, k2, n2 = w2.shape
    tm2, tk2, tn2 = _get_tiling_size_for_gmm_kernel(m, k2, n2, num_experts)

    w2_wrapper_impl = _pallas_gmm_kernel((tm2, tk2, tn2))
    gmm_w2_torch = pallas.custom_jax_kernel(w2_wrapper_impl)

    output = gmm_w2_torch(hidden, w2, group_sizes, w2_scale, w2_bias)

    # 10. Finalize: unsort, apply weights, sum
    output = output[topk_argsort_revert].reshape(num_tokens, topk, -1)
    output = output * topk_weights.unsqueeze(-1)
    output = output.sum(dim=1)

    # 11. Remove padding
    return output[:, :hidden_size]
