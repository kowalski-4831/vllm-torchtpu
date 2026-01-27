# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import vllm.envs as vllm_envs
from torch_tpu import api
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention, MLAAttention
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.tasks import SupportedTask
from vllm.v1 import utils as vllm_utils
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MLAAttentionSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from tpu_inference import envs, utils
from tpu_inference.distributed import jax_parallel_state
from tpu_inference.distributed.utils import get_node_id
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


@dataclass
class PPConfig:
    rank: int
    ip: str
    prev_worker_ip: str
    pp_world_size: int

    # default env vars for
    # TPU_PROCESS_BOUNDS, TPU_CHIPS_PER_PROCESS_BOUNDS, TPU_VISIBLE_CHIPS
    # if PP is used in single host.
    default_tpu_process_bounds: str = field(init=False)
    default_tpu_chips_per_process_bounds: str = field(init=False)
    default_tpu_visible_chips: str = field(init=False)

    def __post_init__(self):
        self.default_tpu_process_bounds = f"1,{self.pp_world_size},1"
        self.default_tpu_chips_per_process_bounds = "1,1,1"
        self.default_tpu_visible_chips = f"{self.rank}"


class TPUWorker:

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        devices=None,
        ip: str = "localhost",
        prev_worker_ip: str = "localhost",
    ):
        # If we use vLLM's model implementation in PyTorch, we should set it
        # with torch version of the dtype.
        impl = envs.MODEL_IMPL_TYPE
        if impl != "vllm":
            raise ValueError("Only vLLM models are supported")

        self.vllm_config = vllm_config
        # TODO: Fix this when enabling TorchTPU compilation
        from vllm.config import CompilationMode
        self.vllm_config.compilation_config.mode = CompilationMode.NONE

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.devices = devices if devices is not None else []
        self.device_ranks = set(device.id for device in self.devices)
        self.pp_config = PPConfig(rank, ip, prev_worker_ip,
                                  self.parallel_config.pipeline_parallel_size)

        self.kv_cache_dtype = self.model_config.dtype

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils.import_utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Delay profiler initialization to the start of the profiling.
        # This is because in vLLM V1, MP runtime is initialized before the
        # TPU Worker is initialized. The profiler server needs to start after
        # MP runtime is initialized.
        self.profile_dir = None
        if (vllm_envs.VLLM_TORCH_PROFILER_DIR and self.rank < 1
                and self.pp_config.pp_world_size == 1):
            if not self.devices or 0 in self.device_ranks:
                # For TPU, we can only have 1 active profiler session for 1 profiler
                # server. So we only profile on rank0.
                self.profile_dir = vllm_envs.VLLM_TORCH_PROFILER_DIR
                logger.info("Profiling enabled. Traces will be saved to: %s",
                            self.profile_dir)

        # For PP, we use MPMD so we want to profile every worker.
        if self.pp_config.pp_world_size > 1 and vllm_envs.VLLM_TORCH_PROFILER_DIR:
            self.profile_dir = os.path.join(
                vllm_envs.VLLM_TORCH_PROFILER_DIR,
                f"pprank_{self.rank}_ppworldsize_{self.pp_config.pp_world_size}",
            )
            os.makedirs(self.profile_dir, exist_ok=True)

        # step_counter is used to calculate uuid to transfer intermediate tensors.
        self.step_counter = 0

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(
        self,
        tpu_process_bounds="",
        tpu_chips_per_process_bounds="",
        tpu_visible_chips="",
    ):
        # set tpu visible devices for Jax runtime in single host PP.
        multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "").lower()
        if (multihost_backend != "ray"
                and self.parallel_config.pipeline_parallel_size > 1):
            tpu_ports = [
                jax_parallel_state.BASE_JAX_PORT + i
                for i in range(self.pp_config.pp_world_size)
            ]
            os.environ["TPU_PROCESS_ADDRESSES"] = ",".join(
                [f"localhost:{port}" for port in tpu_ports])
            os.environ["TPU_PROCESS_PORT"] = f"{tpu_ports[self.rank]}"
            os.environ["CLOUD_TPU_TASK_ID"] = f"{self.rank}"

            # Note: Below is the setting for v6e8 host (8 chips of v6e)
            # Replace with your own topology.
            # There are 2 ways of subslicing a v6e
            # 1) 2 slices with 4 TPU chips each, we can do PP=2, TP=1/2/3/4
            #   TPU_PROCESS_BOUNDS = "1,1,1"
            #   TPU_CHIPS_PER_PROCESS_BOUNDS = "1,4,1"
            #   TPU_VISIBLE_CHIPS = "0,1,2,3" or "4,5,6,7"
            # 2) 1 chip for each subslice, with at most 8 subslices,
            #    we can do TP=1, PP=1/2/3/4/5/6/7/8
            os.environ["TPU_PROCESS_BOUNDS"] = (
                tpu_process_bounds if tpu_process_bounds else
                self.pp_config.default_tpu_process_bounds)
            os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = (
                tpu_chips_per_process_bounds if tpu_chips_per_process_bounds
                else self.pp_config.default_tpu_chips_per_process_bounds)
            os.environ["TPU_VISIBLE_CHIPS"] = (
                tpu_visible_chips if tpu_visible_chips else
                self.pp_config.default_tpu_visible_chips)

        if not self.devices:
            self.devices = []
            self.devices.append(api.tpu_device())

        # Initialize the vLLM distribution layer as a single chip environment,
        # we'll swap the model's parallel modules with TPU SPMD equivalents.
        with set_current_vllm_config(self.vllm_config):
            temp_file = tempfile.mkstemp()[1]
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{temp_file}",
                backend="gloo",
            )
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )

        # jax_parallel_state.init_pp_distributed_environment(
        #    self.pp_config.ip,
        #    self.rank,
        #    self.parallel_config.pipeline_parallel_size,
        #    self.devices[0],
        #    need_pp=self.parallel_config.pipeline_parallel_size > 1)

        # ensure_kv_transfer_initialized(self.vllm_config)

        is_first_rank = True
        is_last_rank = True
        if self.parallel_config.pipeline_parallel_size > 1:
            is_first_rank = self.rank == 0
            is_last_rank = self.rank == self.pp_config.pp_world_size - 1

        # TODO: Fix device assignment
        self.model_runner = TPUModelRunner(self.vllm_config, self.devices[0])
        logger.info(f"Init worker | "
                    f"rank={self.rank} | "
                    f"is_first_rank={is_first_rank} | "
                    f"is_last_rank={is_last_rank} | "
                    f"node_id={get_node_id()} | "
                    f"is_driver_worker={self.is_driver_worker} | ")
        # f"hbm={utils.hbm_usage_gb(self.devices)}GiB")
        vllm_utils.report_usage_stats(self.vllm_config)

    def initialize_pp_transfer_connect(self):
        if self.rank == 0:
            return
        jax_parallel_state.connect(self.pp_config.prev_worker_ip,
                                   self.rank - 1)

    def determine_available_memory(self) -> int:
        gpu_memory_utilization = self.cache_config.gpu_memory_utilization

        # TODO: Fix device memory when corresponding APIs become available
        hbm_usage = [
        ]  # 15 * 1024 *1024 *1024#utils.hbm_usage_bytes(self.devices)
        total_hbm_limit = total_hbm_used = 0
        for used, limit in hbm_usage:
            total_hbm_used += used
            total_hbm_limit += limit

        total_hbm_used = 15 * 1024 * 1024 * 1024
        total_hbm_limit = 28 * 1024 * 1024 * 1024

        total_hbm_limit_cap = total_hbm_limit * gpu_memory_utilization
        total_hbm_avail = int(total_hbm_limit_cap - total_hbm_used)

        total_hbm_limit_gb = round(total_hbm_limit / utils.GBYTES, 2)
        total_hbm_limit_cap_gb = round(total_hbm_limit_cap / utils.GBYTES, 2)
        total_hbm_used_gb = round(total_hbm_used / utils.GBYTES, 2)
        total_hbm_avail_gb = round(total_hbm_avail / utils.GBYTES, 2)

        logger.info(f"Memory statistics | "
                    f"{total_hbm_limit_gb=}GiB | "
                    f"{total_hbm_limit_cap_gb=}GiB | "
                    f"{total_hbm_used_gb=}GiB | "
                    f"{total_hbm_avail_gb=}GiB")

        if total_hbm_avail <= 0:
            raise ValueError(f"{total_hbm_used_gb=}GiB exceeds "
                             f"{total_hbm_limit_cap_gb=}GiB by "
                             f"{-total_hbm_avail_gb}GiB. Please consider "
                             f"increasing --gpu-memory-utilization from "
                             f"{gpu_memory_utilization} to a larger value.")
        return total_hbm_avail

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Optional[ModelRunnerOutput]:
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self,
                      grammar_output: GrammarOutput) -> ModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    def add_lora(
        self,
        lora_request: LoRARequest,
    ) -> bool:
        raise NotImplementedError("TODO")

    def profile(self, is_start: bool = True):
        # TODO: Currently still have jax dependency. We are supposed to use
        # torch_tpu api, but this torchtpu API only works in google3 for now. See b/470479047.

        if self.profile_dir is None:
            logger.warning("Profile directory is not set. Skipping profiling.")
            return

        if is_start:
            logger.info(
                f"Starting JAX profiler trace at {self.profile_dir}...")
            options = jax.profiler.ProfileOptions()
            # default: https://docs.jax.dev/en/latest/profiling.html#general-options
            options.python_tracer_level = envs.PYTHON_TRACER_LEVEL
            options.host_tracer_level = int(os.getenv("HOST_TRACER_LEVEL", 1))
            jax.profiler.start_trace(self.profile_dir,
                                     profiler_options=options)
        else:
            logger.info("Stopping JAX profiler trace...")
            jax.profiler.stop_trace()

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        # TODO: Fix
        self.model_runner.capture_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        # self.model_runner._init_random()

    def reset_mm_cache(self) -> None:
        pass

    def get_model(self):
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        block_size = self.vllm_config.cache_config.block_size
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            # Classic Attention path
            if isinstance(attn_module, Attention):
                if (kv_tgt_layer :=
                        attn_module.kv_sharing_target_layer_name) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                if attn_module.attn_type == AttentionType.DECODER:
                    if isinstance(attn_module, ChunkedLocalAttention):
                        logger.warning_once(
                            "Using irope in Pallas is not supported yet, it "
                            "will fall back to global attention for long context."
                        )
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                            sliding_window=attn_module.sliding_window,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                        )
                elif attn_module.attn_type in (
                        AttentionType.ENCODER,
                        AttentionType.ENCODER_ONLY,
                ):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(
                        f"Unknown attention type: {attn_module.attn_type}")
            # MLAAttention path
            elif isinstance(attn_module, MLAAttention):
                if layer_name in kv_cache_spec:
                    continue
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    cache_dtype_str=cache_dtype_str,
                )
            else:
                continue

        return kv_cache_spec

    def initialize_from_config(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def get_node_kv_ip_port(self) -> tuple[int, str, int]:
        pass

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def sync_weights(
        self,
        updated_weights,
        mappings: Dict[str, Tuple[str, Tuple[str]]],
        transpose_keys: Dict[str, Tuple[int]],
        reshard_fn=None,
    ) -> None:
        """Sync the updated weights to the model runner."""
        return self.model_runner._sync_weights(
            updated_weights=updated_weights,
            mappings=mappings,
            transpose_keys=transpose_keys,
            reshard_fn=reshard_fn,
        )

    def shutdown(self) -> None:
        return

    # Ray executor do not need handshake metadata
    # as we pass the kv_parameters through proxy server
    def get_kv_connector_handshake_metadata(self) -> None:
        pass
