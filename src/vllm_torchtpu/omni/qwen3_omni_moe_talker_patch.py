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
"""TPU runtime patches for Qwen3-Omni MoE Talker and Code Predictor."""

from contextlib import nullcontext
from functools import cache as run_once
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def apply_omni_talker_patches() -> None:
    try:
        from vllm_omni.model_executor.models.common import qwen3_code_predictor as cp
        from vllm_omni.model_executor.models.qwen3_omni import (
            qwen3_omni as qwen3_omni_mod,
        )

        if hasattr(cp, "_RMSNorm"):
            cp._RMSNorm.forward_tpu = cp._RMSNorm.forward_native

        if hasattr(cp, "_RotaryEmbedding"):

            def _rotary_forward_tpu(
                self: Any, x: torch.Tensor, position_ids: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                cos = self.cos_cached.to(position_ids.device)[position_ids].to(
                    dtype=x.dtype
                )
                sin = self.sin_cached.to(position_ids.device)[position_ids].to(
                    dtype=x.dtype
                )
                return cos, sin

            cp._RotaryEmbedding.forward_tpu = _rotary_forward_tpu

        if hasattr(cp, "CodePredictorBaseModel"):

            def _base_forward(
                self: Any, inputs_embeds: torch.Tensor, position_ids: torch.Tensor
            ) -> torch.Tensor:
                input_dtype = inputs_embeds.dtype
                use_fp32 = (
                    input_dtype == torch.float16
                    and inputs_embeds.device.type not in ("cpu", "tpu")
                )
                if use_fp32:
                    inputs_embeds = inputs_embeds.float()
                    ctx = torch.amp.autocast(
                        inputs_embeds.device.type, dtype=torch.float32
                    )
                else:
                    ctx = nullcontext()
                with ctx:
                    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
                    hidden_states = inputs_embeds
                    for layer in self.layers:
                        hidden_states = layer(hidden_states, position_embeddings)
                    hidden_states = self.norm(hidden_states)
                return hidden_states.to(input_dtype)

            cp.CodePredictorBaseModel.forward = _base_forward

        if hasattr(cp, "CodePredictorWrapper"):
            orig_cp_forward = cp.CodePredictorWrapper.forward

            def _wrapper_forward(
                self: Any,
                layer0_code: torch.Tensor,
                layer0_embed: torch.Tensor,
                last_talker_hidden: torch.Tensor,
                do_sample: bool = True,
                temperature: float = 0.9,
                top_k: int = 50,
                top_p: float = 1.0,
                generator: torch.Generator | None = None,
                generators: list[torch.Generator | None] | None = None,
            ) -> Any:
                device = layer0_code.device
                if device.type in ("tpu", "xla"):
                    if next(self.parameters()).device.type != "cpu":
                        self.to("cpu")
                    if getattr(self, "_compiled_model_fwd", None) is None:
                        self._model_dtype = next(self.model.parameters()).dtype
                        self._lm_heads_list = list(self.lm_head)
                        self._codec_embeds_list = list(self.model.codec_embedding)
                        self._compiled_model_fwd = self.model.forward
                    res = orig_cp_forward(
                        self,
                        layer0_code=layer0_code.cpu(),
                        layer0_embed=layer0_embed.cpu(),
                        last_talker_hidden=last_talker_hidden.cpu(),
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                        generators=generators,
                    )
                    if isinstance(res, tuple):
                        return tuple(
                            r.to(device=device) if isinstance(r, torch.Tensor) else r
                            for r in res
                        )
                    return res.to(device=device)
                return orig_cp_forward(
                    self,
                    layer0_code,
                    layer0_embed,
                    last_talker_hidden,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                    generators=generators,
                )

            cp.CodePredictorWrapper.forward = _wrapper_forward

        omni_cls = getattr(qwen3_omni_mod, "Qwen3OmniMoeForConditionalGeneration", None)
        if omni_cls is not None and hasattr(omni_cls, "talker_preprocess"):
            orig_talker_preprocess = omni_cls.talker_preprocess

            def _talker_preprocess(
                self: Any,
                input_ids: torch.Tensor | None,
                input_embeds: torch.Tensor | None = None,
                *args: Any,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
                if input_ids is None and input_embeds is not None:
                    input_ids = torch.zeros(
                        (input_embeds.shape[0],),
                        dtype=torch.long,
                        device=input_embeds.device,
                    )
                ids, embeds, update_dict = orig_talker_preprocess(
                    self, input_ids, input_embeds, *args, **kwargs
                )
                codes = (
                    update_dict.get("codes") if isinstance(update_dict, dict) else None
                )
                if isinstance(codes, dict) and isinstance(
                    codes.get("audio"), torch.Tensor
                ):
                    codes["audio"] = codes["audio"].detach().cpu()
                return ids, embeds, update_dict

            omni_cls.talker_preprocess = _talker_preprocess

        logger.info("Applied TPU patch: Qwen3-Omni Talker and Code Predictor.")
    except Exception as e:
        logger.debug("Skipped Qwen3-Omni Talker patch: %s", e)
