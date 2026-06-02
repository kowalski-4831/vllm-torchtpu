# SPDX-License-Identifier: Apache-2.0

# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import tpu_inference.env_override  # noqa: F401  # isort: skip

import os
import time
from typing import TYPE_CHECKING, Optional, Tuple, Union

import portpicker
import torch
import vllm.envs as vllm_envs
# Ensure the "tpu" torch.compile backend is registered before vllm
# tries to use it (e.g. in @torch.compile decorators at import time).
from torch_tpu._internal import compile as _register_tpu_backend  # noqa: F401
from torch_tpu._internal.utils import hardware
from vllm.platforms.interface import Platform, PlatformEnum

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingConfigManager
from tpu_inference.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.inputs import ProcessorInputs
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams, SamplingType
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    ModelConfig = None
    VllmConfig = None
    ProcessorInputs = None
    PoolingParams = None
    AttentionBackendEnum = None
    SamplingParams = None
    SamplingType = None

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Modules/attrs whose @torch.compile(dynamic=True) wrappers must be removed.
# Each entry is (module_path, function_name).
# ---------------------------------------------------------------------------
_DYNAMIC_COMPILE_TARGETS: list[tuple[str, str]] = [
    ("vllm.model_executor.layers.vocab_parallel_embedding",
     "get_masked_input_and_mask"),
    ("vllm.v1.sample.ops.logprobs", "batched_count_greater_than"),
    ("vllm.v1.sample.ops.topk_topp_sampler", "compiled_random_sample"),
    ("vllm.utils.deep_gemm", "per_block_cast_to_fp8"),
]

_dynamic_compile_unwrapped = False


def _configure_torchtpu_eager_mode() -> None:
    from torch_tpu._internal import execution_mode
    previous_mode = execution_mode.eager_mode
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    logger.info("TorchTPU eager mode configured: %s (previous=%s)",
                execution_mode.eager_mode.name, previous_mode.name)


def _unwrap_dynamic_compile_fns() -> None:
    """Unwrap all known ``@torch.compile(dynamic=True)`` decorators in vllm.

    Safe to call multiple times; only the first call per process does work.
    Failures to import a module are silently ignored (the module may not be
    used at all in this configuration).
    """
    global _dynamic_compile_unwrapped
    if _dynamic_compile_unwrapped:
        return
    _dynamic_compile_unwrapped = True

    import importlib

    for module_path, fn_name in _DYNAMIC_COMPILE_TARGETS:
        try:
            mod = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError):
            continue
        fn = getattr(mod, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(mod, fn_name, fn.__wrapped__)
            logger.debug("Unwrapped @torch.compile(dynamic=True) from %s.%s",
                         module_path, fn_name)


def apply_tpu_patches() -> None:
    """Apply all module-level patches required for TorchTPU.

    Must run in **every** process (including spawned workers) because
    module-level state is re-initialized on import in spawned processes.
    All patches are idempotent.
    """
    from tpu_inference import (_patch_default_moe_runner_select_forward,
                               _patch_vllm_tpu_group_custom_ops)
    from tpu_inference.layers.vllm.custom_ops import _register_custom_ops
    _register_custom_ops()
    _patch_vllm_tpu_group_custom_ops()
    _patch_default_moe_runner_select_forward()
    _configure_torchtpu_eager_mode()
    _unwrap_dynamic_compile_fns()


class TpuPlatform(Platform):
    # Registered via the `vllm.platform_plugins` entry point in pyproject.toml.
    _enum = PlatformEnum.OOT
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "TPU"
    dist_backend: str = "tpu_dist"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "tpu"

    supported_quantization: list[str] = [
        "tpu_int8", "compressed-tensors", "awq", "fp8", "mxfp4"
    ]

    additional_env_vars: list[str] = [
        "PHASED_PROFILING_DIR", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS",
        "TPU_MULTIHOST_BACKEND", "VLLM_MLA_DISABLE", "TPU_BACKEND_TYPE",
        "NEW_MODEL_DESIGN", "ENABLE_QUANTIZED_MATMUL_KERNEL",
        "REQUANTIZE_BLOCK_SIZE", "REQUANTIZE_WEIGHT_DTYPE",
        "MOE_REQUANTIZE_BLOCK_SIZE", "MOE_REQUANTIZE_WEIGHT_DTYPE",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES", "TORCH_TPU_TOPOLOGY"
    ]

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        del parser
        from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                         register_backend)

        register_backend(
            AttentionBackendEnum.FLASH_ATTN,
            "tpu_inference.layers.vllm.attention.PallasAttentionBackend",
        )
        # Experimental batched RPA — opt in via `--attention-backend CUSTOM`.
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "tpu_inference.layers.vllm.attention."
            "PallasBatchedRPAAttentionBackend",
        )

    @classmethod
    def get_worker_distributed_backend(cls, world_size: int) -> str:
        """Pick torch.distributed backend used by worker bootstrap.

        TPU collectives are not needed when world_size==1, and single-rank
        `tpu_dist` bootstrap is unstable in CI. Use `gloo` in that case while
        keeping `tpu_dist` for multi-rank runs.
        """
        if world_size == 1 and cls.dist_backend == "tpu_dist":
            return "gloo"
        return cls.dist_backend

    @classmethod
    def _prepare_singlehost_tpu_env(cls, world_size: int) -> None:
        """Set TORCH_TPU_* env vars needed by PjRt initialization.

        TPUWorker.init_device() always sets WORLD_SIZE in the env, which
        causes PjRt to require TORCH_TPU_SLICEBUILDER_ADDRESSES and
        TORCH_TPU_TOPOLOGY. For world_size > 1, topology is looked up
        via PCI scan using world_size (not auto-detected chip count) so
        slicebuilder and topology match the actual number of workers.
        """
        os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID",
                              str(time.time_ns()))

        sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
            f"localhost:{p}" for p in sb_ports)

        if world_size == 1:
            os.environ["TORCH_TPU_TOPOLOGY"] = "1,1,1"
        else:
            os.environ["TORCH_TPU_TOPOLOGY"] = \
                cls._get_tpu_topology(world_size)

    @classmethod
    def _get_tpu_topology(cls, world_size: int) -> str:
        """Return the torch_tpu-reported topology for the requested world size."""
        topology = hardware.get_tpu_topology(world_size)
        if topology is None:
            raise ValueError("No TPU devices found.")
        return topology

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: "AttentionBackendEnum",
                             attn_selector_config: "AttentionSelectorConfig",
                             **kwargs) -> str:
        backend_name = getattr(selected_backend, "name", None)
        if backend_name == "CUSTOM":
            logger.info("Using TPU Pallas attention backend (batched-RPA "
                        "variant via CUSTOM; requires block_size=256).")
            return ("tpu_inference.layers.vllm.attention."
                    "PallasBatchedRPAAttentionBackend")

        if backend_name not in (None, "FLASH_ATTN"):
            logger.info("Cannot use %s backend on TPU.", selected_backend)

        logger.info("Using TPU Pallas attention backend via FLASH_ATTN.")
        return "tpu_inference.layers.vllm.attention.PallasAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return hardware.get_tpu_device_name()

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.get_device_name().lower() == "tpu v6e":
            logger.info(
                "Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.")
            return torch.float8_e5m2
        return torch.float8_e4m3fn

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU"

    @classmethod
    def get_infinity_values(cls, dtype) -> Tuple[float, float]:
        return torch.finfo(dtype).min, torch.finfo(dtype).max

    @classmethod
    def can_update_inplace(cls):
        return False

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        return 1

    @classmethod
    def inference_mode(cls):
        return True

    @classmethod
    def get_compile_backend(cls) -> str:
        return "tpu_inference.compilation.tpu_compiler.TpuCompilerAdaptor"

    @classmethod
    def _initialize_sharding_config(cls, vllm_config: VllmConfig) -> None:

        sharding_config = ShardingConfigManager.from_vllm_config(vllm_config)
        vllm_config.sharding_config = sharding_config
        logger.info(f"Initialized sharding configuration: {sharding_config}")

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        apply_tpu_patches()

        if vllm_envs.VLLM_TPU_USING_PATHWAYS:
            raise NotImplementedError(
                "Pathways is not supported by torchtpu-vllm. "
                "Unset VLLM_TPU_USING_PATHWAYS.")
        cls._initialize_sharding_config(vllm_config)

        from vllm.config import CompilationMode
        compilation_config = vllm_config.compilation_config
        if compilation_config.mode == CompilationMode.NONE:
            # --enforce-eager is set
            pass
        else:
            compilation_config.mode = CompilationMode.VLLM_COMPILE

        # No graph splitting — TPU handles the full graph including
        compilation_config.splitting_ops = []

        # Disable inductor-specific fusion passes (only available on CUDA).
        compilation_config.pass_config.fuse_norm_quant = False
        compilation_config.pass_config.fuse_act_quant = False
        compilation_config.pass_config.fuse_attn_quant = False
        compilation_config.pass_config.eliminate_noops = False

        # Set compile_sizes to match the token padding buckets used by
        # TPUModelRunner. These are the exact shapes PiecewiseBackend
        # will compile for.
        if compilation_config.compile_sizes is None:
            scheduler_config = vllm_config.scheduler_config
            compilation_config.compile_sizes = _get_token_paddings(
                min_token_size=16,
                max_token_size=scheduler_config.max_num_batched_tokens,
                padding_gap=vllm_envs.VLLM_TPU_BUCKET_PADDING_GAP,
            )
        else:
            compilation_config.compile_sizes = sorted(
                compilation_config.compile_sizes)
        # Clear compile_ranges_split_points — TPU always pads to exact
        # compile_sizes so catch-all ranges are never used.
        compilation_config.compile_ranges_split_points = []

        model_config = vllm_config.model_config
        if model_config is not None and model_config.dtype in (
                torch.float16,
                torch.float32,
        ):
            logger.warning(
                "The TPU backend currently does not support %s. "
                "Using bfloat16 instead.",
                model_config.dtype,
            )
            model_config.dtype = torch.bfloat16

        cache_config = vllm_config.cache_config
        # vLLM's CacheConfig._apply_block_size_default has already populated
        # block_size with DEFAULT_BLOCK_SIZE if the user didn't pass one, so
        # `block_size is None` is never true by this point. The authoritative
        # signal is the `user_specified_block_size` flag pydantic sets.
        block_size_was_unspecified = not getattr(
            cache_config, "user_specified_block_size", cache_config.block_size
            is None)

        from tpu_inference.layers.vllm.attention import (
            PallasAttentionBackend, PallasBatchedRPAAttentionBackend)
        attn_backend = getattr(getattr(vllm_config, "attention_config", None),
                               "backend", None)
        selected_name = getattr(attn_backend, "name", None)
        backend_cls = (PallasBatchedRPAAttentionBackend if selected_name
                       == "CUSTOM" else PallasAttentionBackend)
        is_hybrid = getattr(vllm_config.model_config, "is_hybrid", False)
        cls._is_hybrid = is_hybrid
        cls._speculative_enabled = vllm_config.speculative_config is not None
        if cls._speculative_enabled and \
                vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError(
                "Speculative decoding with async scheduling is not yet "
                "supported on TPU; run with async_scheduling=False.")
        if not is_hybrid and block_size_was_unspecified:
            default = backend_cls.get_page_size(vllm_config)
            cache_config.block_size = (  # type: ignore[assignment]
                backend_cls.get_preferred_block_size(default))

        min_page_size = backend_cls.get_min_page_size(vllm_config)
        if min_page_size > cache_config.block_size:
            logger.warning(
                "Increase the page size from %s to %s to make sure there's"
                "no SMEM OOM",
                cache_config.block_size,
                min_page_size,
            )
            cache_config.block_size = min_page_size  # type: ignore[assignment]
        logger.info("Using KV cache block size: %s", cache_config.block_size)

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        parallel_config.worker_cls = \
                        "tpu_inference.worker.tpu_worker.TPUWorker"

        multihost_backend = envs.TPU_MULTIHOST_BACKEND
        if not multihost_backend:  # Single host
            cls._prepare_singlehost_tpu_env(parallel_config.world_size)
            if (parallel_config.pipeline_parallel_size == 1
                    and parallel_config.tensor_parallel_size == 1):
                logger.info("Force using UniProcExecutor for TPU on \
                        single host without tensor/pipeline parallelism.")
                parallel_config.distributed_executor_backend = "uni"
            else:
                logger.info("Force using TpuMultiprocExecutor for TPU on \
                        single host with tensor/pipeline parallelism.")
                from tpu_inference.executors.tpu_multiproc_executor import \
                    TpuMultiprocExecutor
                parallel_config.distributed_executor_backend = TpuMultiprocExecutor
        elif multihost_backend == "ray":
            from tpu_inference.executors.ray_distributed_executor import \
                RayDistributedExecutor
            parallel_config.distributed_executor_backend = RayDistributedExecutor
            logger.info(
                "Force using RayDistributedExecutor for TPU on multihost.")
        else:
            logger.warning(
                f"Unknown TPU multihost backend: {multihost_backend}. "
                "Using uniproc_executor.")
            parallel_config.distributed_executor_backend = "uni"

        if scheduler_config.is_multimodal_model and not \
            scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            _TPU_SUPPORTED_KV_CONNECTORS = {
                "TPUConnector",
                "TPUConnectorHMA",
                "OffloadingConnector",
            }
            assert kv_transfer_config.kv_connector in \
                _TPU_SUPPORTED_KV_CONNECTORS, (
                f"TPU only supports the following KV connectors: "
                f"{_TPU_SUPPORTED_KV_CONNECTORS}, but got "
                f"'{kv_transfer_config.kv_connector}'."
            )

        from tpu_inference.core.sched.dp_scheduler import \
            update_vllm_config_for_dp_scheduler
        update_vllm_config_for_dp_scheduler(vllm_config)

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        # TODO: TPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # vLLM's default TPU communicator depends on torch_xla. For TorchTPU,
        # use the generic communicator on top of torch.distributed groups, with
        # `dist_backend=tpu_dist` for device-side collectives.
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on TPU is experimental
        return True

    @classmethod
    def validate_request(
        cls,
        processed_inputs: ProcessorInputs,
        params: Union["SamplingParams", PoolingParams],
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        from vllm.sampling_params import SamplingParams, SamplingType

        del processed_inputs
        if isinstance(params, SamplingParams):
            if params.sampling_type == SamplingType.RANDOM_SEED:
                raise ValueError("JAX does not support per-request seed.")
            if params.sampling_type not in (SamplingType.GREEDY,
                                            SamplingType.RANDOM):
                raise ValueError(
                    f"Sampling type {params.sampling_type} is not supported on TPU."
                )
            if getattr(cls, "_speculative_enabled",
                       False) and params.sampling_type != SamplingType.GREEDY:
                raise NotImplementedError(
                    "Speculative decoding currently only supports greedy "
                    "sampling (temperature=0) on TPU.")
            if params.top_k != 0 or params.top_p != 1.0:
                logger.warning(
                    "Top-K and Top-P are not yet supported on TPU and will be ignored."
                )

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    _model_config: ModelConfig) -> bool:
        supported = {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"}
        return kv_cache_dtype in supported

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        """
        Returns if the current platform needs to sync weight loader.
        """
        # TODO: Fix this
        return False

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return getattr(cls, "_is_hybrid", False)


def _get_token_paddings(min_token_size: int, max_token_size: int,
                        padding_gap: int) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size

    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice,
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        logger.info("Using exponential token paddings:")
        while True:
            logger.info("    %d", num)
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        logger.info("Using incremental token paddings:")
        while num <= padding_gap:
            logger.info("    %d", num)
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            logger.info("    %d", num)
            paddings.append(num)

    return paddings
