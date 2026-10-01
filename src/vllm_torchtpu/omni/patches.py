# SPDX-License-Identifier: Apache-2.0
"""TPU runtime patches for vLLM-Omni diffusion models."""

from functools import cache as run_once, wraps
import os
import sys
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _is_mock_vllm_omni() -> bool:
    mod = sys.modules.get("vllm_omni")
    return mod is not None and getattr(mod, "__file__", None) is None


def patch_omni_platform_interface() -> None:
    """Register TPU enum and is_tpu check on OmniPlatform."""
    try:
        from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

        if not hasattr(OmniPlatformEnum, "TPU"):
            OmniPlatformEnum.TPU = "tpu"
        if not hasattr(OmniPlatform, "is_tpu"):
            OmniPlatform.is_tpu = lambda self: getattr(
                self, "_omni_enum", None
            ) in (OmniPlatformEnum.OOT, getattr(OmniPlatformEnum, "TPU", None))
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_omni_platform_interface skipped: %s", e)


def patch_orchestrator_timeouts() -> None:
    """Extend vLLM-Omni orchestrator stage initialization timeout to 1800s."""
    if _is_mock_vllm_omni():
        return
    try:
        import vllm_omni.entrypoints.omni as ob

        if hasattr(ob, "OmniBase") and not getattr(
            ob.OmniBase, "_tpu_timeout_patched", False
        ):
            orig_init = ob.OmniBase.__init__

            def patched_omni_base_init(self: Any, *args: Any, **kwargs: Any) -> None:
                kwargs.setdefault("init_timeout", 1800)
                kwargs.setdefault("stage_init_timeout", 1800)
                orig_init(self, *args, **kwargs)

            ob.OmniBase.__init__ = patched_omni_base_init
            ob.OmniBase._tpu_timeout_patched = True
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_orchestrator_timeouts skipped: %s", e)


def patch_deploy_config_loader() -> None:
    """Inject TPU stage overrides into vLLM-Omni deploy configurations."""
    if _is_mock_vllm_omni():
        return
    try:
        import vllm_omni.config.stage_config as sc

        if getattr(sc, "_tpu_patched", False):
            return

        def _inject_tpu_stages(deploy: Any) -> None:
            if not hasattr(deploy, "platforms") or deploy.platforms is None:
                deploy.platforms = {}
            if not isinstance(deploy.platforms, dict) or "tpu" in deploy.platforms:
                return
            tpu_stages = [
                {
                    "stage_id": (sid := getattr(s, "stage_id", 0)),
                    "devices": (
                        "0,1,2,3"
                        if sid == 0 and getattr(s, "tensor_parallel_size", 1) == 2
                        else getattr(s, "devices", None)
                    ),
                    "tensor_parallel_size": (
                        4
                        if sid == 0 and getattr(s, "tensor_parallel_size", 1) == 2
                        else (getattr(s, "tensor_parallel_size", 1) or 1)
                    ),
                    "worker_cls": (
                        "vllm_torchtpu.omni.worker.TPUGenerationWorker"
                        if sid == 2
                        else "vllm_torchtpu.omni.worker.TPUARWorker"
                    ),
                }
                for s in getattr(deploy, "stages", []) or []
            ]
            if tpu_stages:
                deploy.platforms["tpu"] = {"stages": tpu_stages}

        orig_load = sc.load_deploy_config

        def patched_load_deploy_config(
            path_or_dict: Any, *args: Any, **kwargs: Any
        ) -> Any:
            deploy = orig_load(path_or_dict, *args, **kwargs)
            _inject_tpu_stages(deploy)
            return deploy

        import vllm_omni.config.config_factory as cf
        import vllm_omni.config.omni_config as oc

        orig_build = sc.build_stage_runtime_overrides

        def patched_build(
            stage_id: int, cli_overrides: dict[str, Any], *a: Any, **kw: Any
        ) -> Any:
            filtered = dict(cli_overrides)
            if filtered.get("dtype") == "auto":
                filtered.pop("dtype", None)
            return orig_build(stage_id, filtered, *a, **kw)

        for mod in (sc, cf, oc, sys.modules.get("vllm_omni.config")):
            if mod is not None:
                setattr(mod, "load_deploy_config", patched_load_deploy_config)
                if hasattr(mod, "build_stage_runtime_overrides"):
                    setattr(mod, "build_stage_runtime_overrides", patched_build)

        from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
            QWEN3_OMNI_THINKER_ONLY_PIPELINE,
        )

        if QWEN3_OMNI_THINKER_ONLY_PIPELINE.default_deploy_config_name is None:
            object.__setattr__(
                QWEN3_OMNI_THINKER_ONLY_PIPELINE,
                "default_deploy_config_name",
                "qwen3_omni_moe_thinking.yaml",
            )

        sc._tpu_patched = True
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_deploy_config_loader skipped: %s", e)


def patch_qwen3_omni_models() -> None:
    """Patch Qwen3-Omni model classes for Thinker execution and stub Stage 1/2 methods."""
    if _is_mock_vllm_omni():
        return
    try:
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
            Qwen3OmniMoeForConditionalGeneration,
        )

        if not getattr(Qwen3OmniMoeForConditionalGeneration, "_tpu_patched", False):
            orig_init = Qwen3OmniMoeForConditionalGeneration.__init__

            @wraps(orig_init)
            def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
                orig_init(self, *args, **kwargs)
                if getattr(self, "model_stage", None) == "thinker" and not bool(
                    os.environ.get("VLLM_OMNI_FORCE_STAGED_RUN")
                ):
                    self.is_staged_run = False
                    self.has_multimodal_outputs = False

            def _stub_talker_preprocess(self: Any, *args: Any, **kwargs: Any) -> Any:
                # TODO(qwen3-omni-stage1): Implement TPU Talker prefill/decode preprocessing.
                raise NotImplementedError(
                    "talker_preprocess is not implemented for Thinker-only stage on TPU."
                )

            def _stub_generate_audio(self: Any, *args: Any, **kwargs: Any) -> Any:
                # TODO(qwen3-omni-stage2): Implement TPU Code2Wav audio generation.
                raise NotImplementedError(
                    "generate_audio is not implemented for Thinker-only stage on TPU."
                )

            Qwen3OmniMoeForConditionalGeneration.__init__ = patched_init
            Qwen3OmniMoeForConditionalGeneration.talker_preprocess = (
                _stub_talker_preprocess
            )
            Qwen3OmniMoeForConditionalGeneration.generate_audio = _stub_generate_audio
            Qwen3OmniMoeForConditionalGeneration._tpu_patched = True
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch Qwen3OmniMoe skipped: %s", e)

    try:
        import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as omni_thinker

        from vllm_torchtpu.models.vllm.qwen3_omni_moe_thinker_patch import (
            apply_qwen3_omni_thinker_module_patches,
        )

        apply_qwen3_omni_thinker_module_patches(omni_thinker)
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch qwen3_omni_moe_thinker skipped: %s", e)


def patch_qwen3_code_predictor() -> None:
    """Stub for Stage 1 Talker code predictor TPU/CPU delegation."""
    # TODO(qwen3-omni-stage1): Implement Talker CodePredictorWrapper TPU patch.
    raise NotImplementedError(
        "patch_qwen3_code_predictor is not implemented for Thinker-only stage on TPU."
    )


def patch_stage_input_processors() -> None:
    """Stub for Thinker->Talker->Code2Wav stage input processor patches."""
    # TODO(qwen3-omni-stage1): Implement thinker2talker and talker2code2wav processor patches.
    raise NotImplementedError(
        "patch_stage_input_processors is not implemented for Thinker-only stage on TPU."
    )


@run_once
def apply_omni_tpu_patches() -> None:
    """Extend async output timeout and patch TPU unsupported ops."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Lazy execution attributes failure to sync point: In PyTorch/XLA
    #    and torch_tpu, forward passes are queued lazily in the computation
    #    graph. diffusion_worker emits COMPUTE_DONE prematurely before the TPU
    #    hardware finishes execution. When the async output background loop
    #    triggers host synchronization (.cpu()), it blocks on multi-step diffusion
    #    graph execution and initial compilation.
    # 2. Recompilation storms & JIT compile latency: Initial JIT/AOT
    #    compilation and weight distribution for large diffusion models (e.g.
    #    Wan 14B) take >600s, exceeding vLLM-Omni's default 600s timeout.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed/reduced once:
    # 1. OmniTpuPlatform.record_device_event() implements hardware synchronization
    #    (torch.tpu.synchronize()) to ensure execution completes before COMPUTE_DONE.
    # 2. AOT compilation caching stabilizes startup compile times.
    # -------------------------------------------------------------------------
    import os

    os.environ.setdefault("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", "1800.0")
    logger.info("Applied TPU patch: extended VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT to 1800s.")

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Missing aten::upsample_linear1d.out operator in torch_tpu.
    # Calling torch.nn.functional.interpolate(..., mode="linear") on 3D tensors
    # fails with:
    #   RuntimeError: operator 'aten::upsample_linear1d.out' is not implemented for TPU
    # Used in temporal/audio diffusion alignment (Wan 2.2 S2V, LongCat, audio vocoders).
    #
    # UPSTREAM STATUS & REMOVAL CONDITION:
    # Fixed upstream in torch_tpu commit a0b184ed27d
    # ("Implement aten::upsample_linear1d.out", Sep 14, 2026). This CPU fallback patch
    # can be deleted once vllm-torchtpu bumps to a torch_tpu development wheel
    # >= 0.1.1.dev20260914 containing that commit.
    # -------------------------------------------------------------------------
    orig_interpolate = torch.nn.functional.interpolate

    def _safe_interpolate(
        input: torch.Tensor,
        size: Any = None,
        scale_factor: Any = None,
        mode: str = "nearest",
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input.device.type == "tpu" and mode == "linear":
            return orig_interpolate(
                input.cpu(), size, scale_factor, mode, *args, **kwargs
            ).to(input.device)
        return orig_interpolate(input, size, scale_factor, mode, *args, **kwargs)

    torch.nn.functional.interpolate = _safe_interpolate
    logger.info(
        "Applied TPU patch: safe torch.nn.functional.interpolate for 1D linear TPU ops."
    )

    patch_omni_platform_interface()
    patch_orchestrator_timeouts()
    patch_deploy_config_loader()
    patch_qwen3_omni_models()


@run_once
def apply_omni_model_specific_patches() -> None:
    """Apply model-specific runtime patches."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Pickling a device tensor aborts process: In torch_tpu, calling
    #    pickle.dumps on a device tensor hits
    #    ABSL_CHECK(tensor.storage().allocator() == nullptr) in
    #    csrc/eager/tensor_to_buffer.cc:260-261, terminating the worker process
    #    with SIGABRT (exit code 134) with no Python traceback.
    # 2. libtpu per-process lockfile prevents cross-process init:
    #    Holding TPU tensors across process boundaries causes the CPU-only
    #    orchestrator process to trigger PJRT backend initialization during
    #    unpickling, colliding with the worker on /tmp/libtpu_lockfile.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed once torch_tpu supports safe inter-process tensor
    # serialization / proxy unpickling without triggering PJRT initialization
    # in host processes that do not own TPU hardware.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion import ipc

    orig_pack_tensor = ipc._pack_tensor_if_large

    def _safe_pack_tensor_if_large(
        val: torch.Tensor,
        d2h_stream: Any = None,
    ) -> Any:
        return orig_pack_tensor(val.cpu(), d2h_stream=d2h_stream)

    ipc._pack_tensor_if_large = _safe_pack_tensor_if_large
    logger.info("Applied TPU patch: safe CPU serialization for diffusion IPC tensors.")

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # In LTX-2, LTXRuntime._decode_output produces a (video, audio) tuple. Audio
    # tensors are sub-threshold (<1MB) and returned directly in the output object,
    # bypassing SHM packing. If audio remains on TPU, unpickling in the orchestrator
    # triggers PJRT re-initialization and /tmp/libtpu_lockfile collision, while
    # pickling in the worker risks tensor_to_buffer.cc ABSL_CHECK aborts.
    # Moving video and audio to CPU prevents device tensors entering the IPC stream.
    #
    # REMOVAL / FIX CONDITION:
    # Can be removed once torch_tpu provides safe cross-process device tensor
    # serialization, or once vllm_omni universally ensures all diffusion output
    # payloads are moved to host CPU before IPC serialization.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    orig_decode_output = ltx2_runtime.LTXRuntime._decode_output

    def _safe_decode_output(self: Any, *args: Any, **kwargs: Any):
        res = orig_decode_output(self, *args, **kwargs)
        video, audio = res.output
        res.output = (
            video.cpu() if isinstance(video, torch.Tensor) else video,
            audio.cpu() if isinstance(audio, torch.Tensor) else audio,
        )
        return res

    ltx2_runtime.LTXRuntime._decode_output = _safe_decode_output
    logger.info("Applied TPU patch: safe LTX-2 CPU decode output.")


apply_omni_patches = apply_omni_tpu_patches
