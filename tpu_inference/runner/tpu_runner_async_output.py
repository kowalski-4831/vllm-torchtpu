# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

INVALID_TOKEN_ID = -1


@dataclass
class AsyncTPUCopyState:
    sampled_token_ids_cpu: torch.Tensor
    copy_ready_event: Any
    sampled_token_ids_tpu: torch.Tensor | None = None
    completed: bool = False

    @classmethod
    def from_device(cls,
                    sampled_token_ids: torch.Tensor) -> "AsyncTPUCopyState":
        sampled_token_ids_cpu = sampled_token_ids.to("cpu", non_blocking=True)
        copy_ready_event = torch.tpu.Event()
        copy_ready_event.record()
        return cls(
            sampled_token_ids_cpu=sampled_token_ids_cpu,
            copy_ready_event=copy_ready_event,
            sampled_token_ids_tpu=sampled_token_ids,
        )

    def wait(self) -> None:
        if self.completed:
            return
        self.copy_ready_event.synchronize()
        self.sampled_token_ids_tpu = None
        self.completed = True


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
