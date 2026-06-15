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
"""Torch-based rejection sampler for speculative decoding on TPU."""

import torch

# Placeholder token ID for rejected tokens
PLACEHOLDER_TOKEN_ID = -1


class RejectionSampler:
    """Torch-based rejection sampler for speculative decoding on TPU."""

    def __init__(self):
        pass

    def __call__(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
    ) -> torch.Tensor:
        return self.forward(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            segment_ids=segment_ids,
            group_indices=group_indices,
            max_draft_tokens=max_draft_tokens,
        )

    def forward(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
    ) -> torch.Tensor:
        """Performs rejection sampling for speculative decoding.

    Currently, this implementation only supports greedy rejection sampling.
    This function is intended to be torch.compiled for efficiency on TPU.

    Args:
      draft_token_ids: The token IDs from the draft model.
      num_draft_tokens: The number of draft tokens per sequence in the batch.
      target_logits: The logits from the target model.
      bonus_token_ids: The bonus token IDs to append if all draft tokens are
        accepted.
      segment_ids: Segment IDs to group tokens by batch item.
      group_indices: Indices within each segment.
      max_draft_tokens: The maximum number of draft tokens.

    Returns:
      A tensor of shape [batch_size, max_draft_tokens + 1] containing the
      accepted tokens and bonus tokens, padded with PLACEHOLDER_TOKEN_ID.
    """
        # Currently only supports greedy rejection sampling.
        # This function should be torch.compiled for efficiency.
        return _greedy_rejection_sample_with_segment(
            draft_token_ids,
            target_logits,
            num_draft_tokens,
            bonus_token_ids,
            segment_ids,
            group_indices,
            max_draft_tokens,
        )


def _get_segment_info(num_draft_tokens: torch.Tensor, total_tokens: int):
    """Helper to create segment IDs and per-segment indices."""
    batch_size = num_draft_tokens.shape[0]
    device = num_draft_tokens.device

    # segment_ids assigns a unique ID to each token.
    # In Torch, we can use repeat_interleave.
    segment_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        num_draft_tokens,
    )

    # group_indices creates a within-segment index for each token.
    # E.g., [0, 1, 2, 0, 1, 0, 1, 2, 3] for sequences [3, 2, 4].
    # We can use cumsum and subtraction.
    cum_draft_tokens = torch.cumsum(num_draft_tokens, dim=0)
    segment_starts = torch.cat([
        torch.zeros(1, dtype=num_draft_tokens.dtype, device=device),
        cum_draft_tokens[:-1],
    ])
    broadcast_starts = torch.repeat_interleave(segment_starts,
                                               num_draft_tokens)
    group_indices = (torch.arange(total_tokens, device=device) -
                     broadcast_starts).to(torch.int32)
    return segment_ids, group_indices


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _greedy_rejection_sample_with_segment(
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
) -> torch.Tensor:
    """Vectorized greedy speculative decoding validation in Torch."""
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device

    # Get target argmax
    target_logits_argmax = torch.argmax(target_logits, dim=-1).to(torch.int32)
    draft_token_ids = draft_token_ids.to(torch.int32)

    # Find the first mismatch index per segment using matrix masking.
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices

    # Large value for positions with no mismatches
    # Create an array where mismatched positions hold their `group_index`
    # and matched positions hold a large value.
    large_value = total_tokens

    if total_tokens > 0:
        mismatches = draft_token_ids != target_logits_argmax
        mismatch_indices = torch.where(
            mismatches,
            group_indices,
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )

        masked_mismatch_indices = torch.where(
            match_mask,
            mismatch_indices.unsqueeze(0),
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )
        first_mismatch_idx_per_segment = torch.min(masked_mismatch_indices,
                                                   dim=1).values
    else:
        first_mismatch_idx_per_segment = torch.full((batch_size, ),
                                                    large_value,
                                                    dtype=torch.int32,
                                                    device=device)

    # Step 3: Broadcast Mismatch Info and Generate Main Token Output
    if total_tokens > 0:
        first_mismatch_idx_broadcast = first_mismatch_idx_per_segment[
            segment_ids]

        # Valid if group_index <= first_mismatch_idx
        main_tokens = torch.where(
            group_indices <= first_mismatch_idx_broadcast,
            target_logits_argmax,
            torch.tensor(PLACEHOLDER_TOKEN_ID,
                         dtype=torch.int32,
                         device=device),
        )
    else:
        main_tokens = torch.tensor([], dtype=torch.int32, device=device)

    # Step 4: Handle Bonus Tokens
    all_accepted = first_mismatch_idx_per_segment == large_value
    no_draft_tokens = num_draft_tokens == 0
    should_get_bonus = all_accepted | no_draft_tokens

    # Step 5: Arrange into a [batch_size, max_draft_tokens + 1] matrix
    selected_tokens = torch.full(
        (batch_size, max_draft_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    # Ensure all indices are uniformly int64 to prevent stablehlo.concatenate
    # type errors
    segment_ids = segment_ids.to(torch.int64)
    group_indices = group_indices.to(torch.int64)

    # Place main tokens at their exact per-request positions
    selected_tokens[segment_ids, group_indices] = main_tokens

    # Place bonus tokens immediately after the accepted draft sequence
    # ONLY if all tokens were accepted or there were no draft tokens.
    # If a mismatch occurred at k, selected_tokens[segment_ids, k] already
    # contains target_logits_argmax[k].
    accepted_count = torch.where(
        first_mismatch_idx_per_segment == large_value,
        num_draft_tokens,
        first_mismatch_idx_per_segment,
    ).to(torch.int64)

    batch_range = torch.arange(batch_size, dtype=torch.int64, device=device)
    bonus_token_at_accepted_count = torch.where(
        should_get_bonus,
        bonus_token_ids.to(torch.int32),
        selected_tokens[batch_range, accepted_count],
    )
    selected_tokens[batch_range,
                    accepted_count] = bonus_token_at_accepted_count

    return selected_tokens
