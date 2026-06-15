# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

INVALID_TOKEN_ID = -1


@dataclass
class AsyncTPUCopyState:
    """Async D2H copy state for sampled tokens.

    Per-chunk device tensors are kept at their bucketed (padded) shapes so
    each D2H copy hits a precompiled program. Host-side trim and concat
    happen lazily on `wait()` and produce a single contiguous CPU tensor
    matching the real (unpadded) request count.
    """
    chunks_cpu: list[torch.Tensor]
    chunk_real_lens: list[int]
    copy_ready_event: Any
    chunks_tpu: list[torch.Tensor] | None = None
    completed: bool = False
    _sampled_token_ids_cpu: torch.Tensor | None = None

    @classmethod
    def from_device_chunks(cls, chunks_tpu: list[torch.Tensor],
                           chunk_real_lens: list[int]) -> "AsyncTPUCopyState":
        assert len(chunks_tpu) == len(chunk_real_lens) and len(chunks_tpu) > 0
        chunks_cpu = [t.to("cpu", non_blocking=True) for t in chunks_tpu]
        copy_ready_event = torch.tpu.Event()
        copy_ready_event.record()
        return cls(
            chunks_cpu=chunks_cpu,
            chunk_real_lens=chunk_real_lens,
            copy_ready_event=copy_ready_event,
            chunks_tpu=chunks_tpu,
        )

    def wait(self) -> None:
        if self.completed:
            return
        self.copy_ready_event.synchronize()
        self.chunks_tpu = None
        self.completed = True

    @property
    def sampled_token_ids_cpu(self) -> torch.Tensor:
        """Return the unpadded concatenated CPU tensor.

        Performed lazily after `wait()`; trims each bucketed chunk to its real
        length on the host (no device recompile).
        """
        if self._sampled_token_ids_cpu is None:
            trimmed = [
                t[:n] for t, n in zip(self.chunks_cpu, self.chunk_real_lens)
            ]
            if len(trimmed) == 1:
                self._sampled_token_ids_cpu = trimmed[0]
            else:
                self._sampled_token_ids_cpu = torch.cat(trimmed, dim=0)
        return self._sampled_token_ids_cpu


@dataclass
class AsyncPreResults:
    req_ids: list[str]
    next_tokens_tpu: torch.Tensor
    request_seq_lens: list[tuple[int, Any, int, str]]
    discard_sampled_tokens_req_indices: list[int]
    req_id_to_index_copy: dict[str, int]
    copy_state: AsyncTPUCopyState | None = None

    def wait_for_copy(self) -> torch.Tensor | None:
        if self.copy_state is None:
            return None
        self.copy_state.wait()
        return self.copy_state.sampled_token_ids_cpu


class AsyncTPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Holds asynchronous model output specifically from a TPU runner."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        copy_state: AsyncTPUCopyState | None,
        discard_sampled_tokens_req_indices: list[int],
    ):
        self._model_runner_output: ModelRunnerOutput = model_runner_output
        self.copy_state = copy_state
        self.discard_sampled_tokens_req_indices = discard_sampled_tokens_req_indices

    def get_output(self) -> ModelRunnerOutput:
        if self.copy_state is None:
            self._model_runner_output.sampled_token_ids = []
            return self._model_runner_output

        self.copy_state.wait()
        selected_token_ids = self.copy_state.sampled_token_ids_cpu
        if selected_token_ids.numel() == 0:
            self._model_runner_output.sampled_token_ids = []
            return self._model_runner_output
        if selected_token_ids.dim() == 1:
            selected_token_ids = selected_token_ids.unsqueeze(-1)

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

        self._model_runner_output.sampled_token_ids = valid_sampled_token_ids
        return self._model_runner_output
