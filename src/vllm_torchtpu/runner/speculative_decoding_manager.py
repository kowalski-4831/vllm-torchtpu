# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding on Torch/TPU, containing all necessary indices."""

    draft_token_ids: torch.Tensor
    draft_lengths: torch.Tensor
    draft_lengths_cpu: np.ndarray
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    final_logits_indices: torch.Tensor
    segment_ids: torch.Tensor
    group_indices: torch.Tensor


class SpeculativeDecodingManager:
    """Manages speculative decoding operations, including draft proposal and metadata generation."""

    def __init__(self, runner: TPUModelRunner):
        self.runner = runner
        # Cached draft tokens.
        self._draft_token_ids: Optional[list[list[int]]] = None
        self.spec_token_ids: dict[str, list[int]] = {}

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        num_reqs = self.runner.input_batch.num_reqs
        req_ids = self.runner.input_batch.req_ids[:num_reqs]
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def propose_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        discard_sampled_tokens_req_indices: list[int] | None = None,
        num_rejected_tokens_np: np.ndarray | None = None,
        scheduler_output=None,
    ) -> None:
        """Proposes draft token IDs based on the speculative decoding method.

    Args:
      sampled_token_ids: The token IDs sampled in the previous step.
      discard_sampled_tokens_req_indices: Request indices whose sampled
        tokens should be discarded (partial-prefill case).
      num_rejected_tokens_np: Per-request count of draft tokens rejected
        this step.
      scheduler_output: vLLM SchedulerOutput; needed by eagle3 for partial
        prefill next-token lookup.
    """
        num_reqs = self.runner.input_batch.num_reqs
        if self.runner.speculative_config.method == "ngram":
            assert isinstance(self.runner.drafter, NgramProposer)
            self._draft_token_ids = self.runner.drafter.propose(
                sampled_token_ids[:num_reqs],
                self.runner.input_batch.num_tokens_no_spec,
                self.runner.input_batch.token_ids_cpu,
            )
        elif self.runner.speculative_config.method == "eagle3":
            assert isinstance(self.runner.drafter, Eagle3Proposer)
            self._draft_token_ids = self.runner.drafter.propose(
                sampled_token_ids[:num_reqs],
                discard_sampled_tokens_req_indices or [],
                num_rejected_tokens_np,
                scheduler_output,
            )
        else:
            raise NotImplementedError(
                "Speculative decoding method "
                f"'{self.runner.speculative_config.method}' is not supported.")

    def get_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
    ) -> SpecDecodeMetadata:
        """Calculates indices for speculative decoding forward pass and rejection sampling.

    Args:
      num_draft_tokens: The number of draft tokens for each request.
      cu_num_scheduled_tokens: The cumulative number of scheduled tokens.
      padded_num_reqs: The padded number of requests.

    Returns:
      A SpecDecodeMetadata object containing the necessary indices for
      speculative decoding.
    """
        # [num_reqs]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)

        arange_cpu_np = self.runner.arange_np

        arange = np.concatenate(
            [arange_cpu_np[:n] for n in num_sampled_tokens])

        # Step 2. logits_indices for all tokens (draft + target)
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)

        # Step 3. actual indices in the input batch
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the target logits indices
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)

        arange_draft = np.concatenate(
            [arange_cpu_np[:n] for n in num_draft_tokens])
        target_logits_indices += arange_draft

        # Compute the draft token ids for validation.
        all_token_ids = self.runner.input_ids_cpu.numpy()
        draft_token_ids = all_token_ids[logits_indices[target_logits_indices +
                                                       1]]

        # Padding
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

        padded_logits_length = _get_padded_token_len(
            self.runner.num_tokens_paddings, logits_indices.shape[0])

        padded_logits_indices = np.zeros(padded_logits_length, dtype=np.int32)
        padded_logits_indices[:logits_indices.shape[0]] = logits_indices

        padded_bonus_logits_indices = np.zeros(padded_num_reqs, dtype=np.int32)
        padded_bonus_logits_indices[:bonus_logits_indices.shape[0]] = (
            bonus_logits_indices)

        padded_num_draft_tokens = np.zeros(padded_num_reqs, dtype=np.int32)
        padded_num_draft_tokens[:num_draft_tokens.shape[0]] = num_draft_tokens

        padded_draft_token_ids = np.zeros(padded_logits_length, dtype=np.int32)
        padded_draft_token_ids[:draft_token_ids.shape[0]] = draft_token_ids

        padded_target_logits_indices = np.zeros(padded_logits_length,
                                                dtype=np.int32)
        padded_target_logits_indices[:target_logits_indices.shape[0]] = (
            target_logits_indices)

        segment_ids = np.repeat(
            np.arange(num_draft_tokens.shape[0], dtype=np.int64),
            num_draft_tokens)
        group_indices = (np.concatenate([
            np.arange(n, dtype=np.int32) for n in num_draft_tokens
        ]) if num_draft_tokens.any() else np.array([], dtype=np.int32))

        padded_segment_ids = np.full(padded_logits_length,
                                     num_draft_tokens.shape[0],
                                     dtype=np.int64)
        padded_segment_ids[:segment_ids.shape[0]] = segment_ids

        padded_group_indices = np.zeros(padded_logits_length, dtype=np.int32)
        padded_group_indices[:group_indices.shape[0]] = group_indices

        # CPU -> TPU copy.
        device = self.runner.device

        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.from_numpy(padded_draft_token_ids).to(
                device),
            draft_lengths=torch.from_numpy(padded_num_draft_tokens).to(device),
            draft_lengths_cpu=num_draft_tokens,
            target_logits_indices=torch.from_numpy(
                padded_target_logits_indices).to(device),
            bonus_logits_indices=torch.from_numpy(
                padded_bonus_logits_indices).to(device),
            final_logits_indices=torch.from_numpy(padded_logits_indices).to(
                device),
            segment_ids=torch.from_numpy(padded_segment_ids).to(device),
            group_indices=torch.from_numpy(padded_group_indices).to(device),
        )
        return metadata
