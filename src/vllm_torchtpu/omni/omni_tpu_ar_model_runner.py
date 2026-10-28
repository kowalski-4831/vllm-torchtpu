# Copyright 2026 Google LLC
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
"""TPU Autoregressive (AR) Model Runner for Omni models."""

from typing import Any

import torch

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput


def _slice_inter_stage_payload(
    raw_mm: dict[str, Any],
    req_idx: int,
    tok_start: int | None,
    tok_end: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, val in raw_mm.items():
        if key == "hidden_states" and isinstance(val, dict) and "layers" in val:
            layers = {}
            for lk, lt in val["layers"].items():
                if isinstance(lt, torch.Tensor):
                    sliced = (
                        lt[tok_start:tok_end]
                        if tok_start is not None and tok_end is not None
                        else lt
                    )
                    layers[lk] = sliced.detach().cpu()
                else:
                    layers[lk] = lt
            payload["hidden_states"] = {"layers": layers}
        elif key == "codes" and isinstance(val, dict):
            codes = {}
            for ck, cv in val.items():
                if isinstance(cv, torch.Tensor):
                    sliced = (
                        cv[tok_start:tok_end]
                        if tok_start is not None
                        and tok_end is not None
                        and cv.shape[0] >= tok_end
                        else cv
                    )
                    codes[ck] = sliced.detach().cpu()
                else:
                    codes[ck] = cv
            payload["codes"] = codes
        elif key == "embed" and isinstance(val, dict):
            embed = {}
            for ek, ev in val.items():
                item = ev[req_idx] if isinstance(ev, list) and req_idx < len(ev) else ev
                embed[ek] = (
                    item.detach().cpu() if isinstance(item, torch.Tensor) else item
                )
            payload["embed"] = embed
        elif isinstance(val, torch.Tensor):
            payload[key] = val.detach().cpu()
        else:
            payload[key] = val
    return payload


class OmniTPUARModelRunner(OmniTPUModelRunner):
    """TPU model runner for autoregressive stages (thinker, talker) in Omni models.

    The connector lifecycle lives in ``OmniTPUModelRunner``; this class adds
    multimodal chunk slicing, Talker preprocess / MTP / postprocess hooks, and
    connector output attachment.
    """

    def _prepare_inputs(
        self, scheduler_output: Any, start_index: int, *args: Any, **kwargs: Any
    ) -> Any:
        res = super()._prepare_inputs(scheduler_output, start_index, *args, **kwargs)
        self._omni_chunk_req_slice = (start_index, res[4])
        return res

    def _chunk_req_spans(self) -> list[tuple[str, int, int, int]]:
        input_batch = getattr(self, "input_batch", None)
        query_start_loc = getattr(self, "query_start_loc_np", None)
        if input_batch is None or query_start_loc is None:
            return []
        num_reqs = getattr(input_batch, "num_reqs", len(input_batch.req_ids))
        start_idx, end_idx = getattr(self, "_omni_chunk_req_slice", (0, num_reqs))
        spans: list[tuple[str, int, int, int]] = []
        for r_i, req_id in enumerate(input_batch.req_ids[start_idx:end_idx]):
            if req_id is None or r_i + 1 >= len(query_start_loc):
                continue
            spans.append(
                (
                    req_id,
                    start_idx + r_i,
                    int(query_start_loc[r_i]),
                    int(query_start_loc[r_i + 1]),
                )
            )
        return spans

    def _gather_mm_embeddings(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_multimodal = super()._gather_mm_embeddings(*args, **kwargs)
        self._omni_mm_embeds_list = mm_embeds
        if mm_embeds:
            total = sum(embed.shape[0] for embed in mm_embeds)
            return [torch.arange(total, dtype=torch.int64)], is_multimodal
        return mm_embeds, is_multimodal

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> Any:
        if mm_embed_inputs is not None:
            chunk_embeds, is_multimodal = mm_embed_inputs
            mm_embeds = getattr(self, "_omni_mm_embeds_list", None)
            if chunk_embeds and mm_embeds and chunk_embeds[0].ndim == 1:
                chunk_start = int(chunk_embeds[0][0])
                chunk_end = chunk_start + chunk_embeds[0].shape[0]
                sliced_embeds, offset = [], 0
                for embed in mm_embeds:
                    rows = embed.shape[0]
                    start = max(0, chunk_start - offset)
                    end = min(rows, chunk_end - offset)
                    if start < end:
                        sliced = embed[start:end]
                        if hasattr(embed, "modality"):
                            sliced.modality = embed.modality
                        sliced_embeds.append(sliced)
                    offset += rows
                mm_embed_inputs = (sliced_embeds, is_multimodal)
        model_input_ids, inputs_embeds = super()._get_model_inputs(
            input_ids, mm_embed_inputs
        )
        orig_model = getattr(
            getattr(self, "model", None), "_orig_mod", getattr(self, "model", None)
        )
        if getattr(orig_model, "has_preprocess", False):
            preprocess_ids = (
                model_input_ids if model_input_ids is not None else input_ids
            )
            device = getattr(
                self,
                "device",
                preprocess_ids.device
                if preprocess_ids is not None
                else torch.device("cpu"),
            )
            num_computed_arr = getattr(
                getattr(self, "input_batch", None), "num_computed_tokens_cpu", None
            )
            mtp_fn = getattr(orig_model, "talker_mtp", None)
            for req_id, req_idx, s, e in self._chunk_req_spans():
                req_info = self._intermediate_buffer.setdefault(req_id, {})
                num_computed = (
                    int(num_computed_arr[req_idx])
                    if num_computed_arr is not None
                    else 0
                )
                req_info["_omni_num_computed_tokens"] = num_computed
                req_info["_omni_is_prefill"] = num_computed == 0
                r_ids = preprocess_ids[s:e] if preprocess_ids is not None else None
                r_embeds = inputs_embeds[s:e] if inputs_embeds is not None else None
                p_ids, p_embeds, update_dict = orig_model.preprocess(
                    input_ids=r_ids, input_embeds=r_embeds, **req_info
                )
                self._update_intermediate_buffer(req_id, update_dict)
                if inputs_embeds is None and p_embeds is not None:
                    total_tokens = (
                        preprocess_ids.shape[0] if preprocess_ids is not None else e
                    )
                    inputs_embeds = torch.zeros(
                        (total_tokens, p_embeds.shape[-1]),
                        device=device,
                        dtype=p_embeds.dtype,
                    )
                if p_embeds is not None and inputs_embeds is not None:
                    p_len = min(p_embeds.shape[0], inputs_embeds.shape[0] - s)
                    inputs_embeds[s : s + p_len] = p_embeds[:p_len].to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                mtp_inputs = req_info.pop("mtp_inputs", None)
                if (
                    mtp_inputs is not None
                    and callable(mtp_fn)
                    and inputs_embeds is not None
                ):
                    last_talker_hidden, text_step = mtp_inputs
                    mtp_embeds, code_predictor_codes = mtp_fn(
                        p_ids, inputs_embeds[s:e], last_talker_hidden, text_step
                    )
                    p_len = min(mtp_embeds.shape[0], inputs_embeds.shape[0] - s)
                    inputs_embeds[s : s + p_len] = mtp_embeds[:p_len].to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                    self._update_intermediate_buffer(
                        req_id, {"codes": {"audio": code_predictor_codes}}
                    )
            if inputs_embeds is not None:
                return None, inputs_embeds
        return model_input_ids, inputs_embeds

    def forward_model(self, *args: Any, **kwargs: Any) -> Any:
        out, aux = super().forward_model(*args, **kwargs)
        hidden_states = getattr(out, "text_hidden_states", out)
        orig_model = getattr(
            getattr(self, "model", None), "_orig_mod", getattr(self, "model", None)
        )
        if getattr(orig_model, "has_postprocess", False) and isinstance(
            hidden_states, torch.Tensor
        ):
            for req_id, _req_idx, s, e in self._chunk_req_spans():
                req_info = self._intermediate_buffer.get(req_id, {})
                post_kwargs = {
                    k: v for k, v in req_info.items() if k != "hidden_states"
                }
                update_dict = orig_model.postprocess(hidden_states[s:e], **post_kwargs)
                self._update_intermediate_buffer(req_id, update_dict)
        self._omni_last_hidden_states = hidden_states
        self._omni_last_aux_hidden_states = aux
        return hidden_states, aux

    def _populate_inter_stage_outputs(self, real_output: Any) -> None:
        orig_model = getattr(
            getattr(self, "model", None), "_orig_mod", getattr(self, "model", None)
        )
        make_fn = getattr(orig_model, "make_omni_output", None)
        raw_out = getattr(self, "_omni_last_hidden_states", None)
        aux_out = getattr(self, "_omni_last_aux_hidden_states", None)
        self._omni_last_hidden_states = None
        self._omni_last_aux_hidden_states = None
        req_ids = getattr(real_output, "req_ids", None)
        if (
            not callable(make_fn)
            or not req_ids
            or (raw_out is None and aux_out is None)
        ):
            return
        model_out = (raw_out, aux_out) if aux_out is not None else raw_out
        req_infos = [self._intermediate_buffer.get(rid, {}) for rid in req_ids]
        omni_out = make_fn(model_out, model_intermediate_buffer=req_infos)
        raw_mm = getattr(
            omni_out,
            "multimodal_outputs",
            omni_out if isinstance(omni_out, dict) else None,
        )
        if not isinstance(raw_mm, dict):
            return
        query_start_loc = getattr(self, "query_start_loc_np", None)
        per_req_payloads = []
        for idx in range(len(req_ids)):
            tok_start = (
                int(query_start_loc[idx])
                if query_start_loc is not None and idx + 1 < len(query_start_loc)
                else None
            )
            tok_end = (
                int(query_start_loc[idx + 1])
                if query_start_loc is not None and idx + 1 < len(query_start_loc)
                else None
            )
            per_req_payloads.append(
                _slice_inter_stage_payload(raw_mm, idx, tok_start, tok_end)
            )
        real_output.inter_stage_outputs = per_req_payloads

    def _with_omni_connector_output(self, output: Any) -> Any:
        """Attach connector signals to the ``ModelRunnerOutput`` the engine reads.

        Under async scheduling ``sample_tokens`` returns an
        ``AsyncTPUModelRunnerOutput`` and the engine consumes ``get_output()``,
        so the signals have to live on the wrapped ``ModelRunnerOutput``
        (upstream GPUARModelRunner sets them on the output before wrapping it),
        not on the wrapper, where they would be dropped.
        """
        if output is None or not self._omni_connector_initialized:
            return output
        if isinstance(output, AsyncTPUModelRunnerOutput):
            self._populate_inter_stage_outputs(output._model_runner_output)
            output._model_runner_output = self.attach_omni_connector_output(
                output._model_runner_output
            )
            return output
        self._populate_inter_stage_outputs(output)
        return self.attach_omni_connector_output(output)

    def execute_model(
        self, scheduler_output: Any, intermediate_tensors: Any = None
    ) -> Any:
        output = super().execute_model(scheduler_output, intermediate_tensors)
        # ``None`` is the normal two-phase path: the engine calls sample_tokens()
        # next, which attaches the signals. A zero-token step (and the
        # KV-connector-only variant) returns an output right here and the engine
        # skips sample_tokens(), so attach on this path as well.
        return self._with_omni_connector_output(output)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        return self._with_omni_connector_output(super().sample_tokens(grammar_output))
