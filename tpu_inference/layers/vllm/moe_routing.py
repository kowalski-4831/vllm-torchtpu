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
"""Routing helpers for TPU local MoE kernels."""

import torch


def _apply_scoring_fn(scoring_fn: str,
                      router_logits: torch.Tensor) -> torch.Tensor:
    scores = router_logits.float()
    if scoring_fn == "softmax":
        return scores.softmax(dim=-1)
    if scoring_fn == "sigmoid":
        return scores.sigmoid()
    raise NotImplementedError(
        f"FusedMoE does not support {scoring_fn} scoring function for TPU.")


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local routed ids/weights for the non-EP path."""
    topk_weights, topk_ids = torch.topk(_apply_scoring_fn(
        scoring_fn, router_logits),
                                        k=topk,
                                        dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)


def select_experts_ep(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global routing, then keep only local experts for this EP rank."""
    topk_weights, topk_ids = torch.topk(_apply_scoring_fn(
        scoring_fn, router_logits),
                                        k=topk,
                                        dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    local_topk_ids = expert_map[topk_ids.to(torch.long)].to(torch.int32)
    valid_mask = local_topk_ids >= 0
    local_topk_weights = topk_weights.to(hidden_states.dtype)
    local_topk_weights = torch.where(
        valid_mask,
        local_topk_weights,
        torch.zeros_like(local_topk_weights),
    )
    local_topk_ids = torch.where(
        valid_mask,
        local_topk_ids,
        torch.full_like(local_topk_ids, -1),
    )
    return local_topk_weights, local_topk_ids
