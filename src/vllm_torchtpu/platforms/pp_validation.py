# SPDX-License-Identifier: Apache-2.0
"""Static checks for pipeline parallelism on TPU."""

from typing import Any


def _legacy_ray_executor() -> bool:
    import vllm.envs as vllm_envs

    from vllm_torchtpu import envs
    return (envs.TPU_MULTIHOST_BACKEND == "ray"
            and not vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND)


def validate_pipeline_parallel_config(vllm_config: Any) -> None:
    """Reject configurations the TPU pipeline-parallel path does not serve.

    Raises NotImplementedError naming the unsupported feature.
    """
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size <= 1:
        return

    unsupported = None
    remedy = ""
    if parallel_config.tensor_parallel_size > 1:
        # On torch_tpu (dev20260813) a stage's TP collectives stall on the
        # first serving step, when the other stages are not launching the
        # same collectives; warmup, where every stage runs them, passes.
        unsupported = "tensor parallelism inside a stage"
        remedy = " Use --tensor-parallel-size 1."
    elif vllm_config.scheduler_config.async_scheduling:
        # Stages before the last never see the sampled tokens the async
        # placeholder substitution reads from the device.
        unsupported = "async scheduling"
        remedy = " Pass --no-async-scheduling."
    elif vllm_config.speculative_config is not None:
        unsupported = "speculative decoding"
    elif vllm_config.kv_transfer_config is not None:
        # The KV transfer connectors assume every worker holds every layer.
        unsupported = "KV transfer connectors"
    elif _legacy_ray_executor():
        # The pipeline's hand-off pushes go through the executor hook that
        # the multiprocess executor and the Ray V2 executor share; the
        # legacy Ray executor has no such hook.
        unsupported = "the legacy Ray executor"
        remedy = " Set VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1."
    if unsupported is not None:
        raise NotImplementedError(
            f"Pipeline parallelism on TPU does not support {unsupported} "
            f"yet.{remedy}")
