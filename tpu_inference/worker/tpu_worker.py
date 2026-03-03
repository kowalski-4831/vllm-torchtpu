# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import torch
import vllm.envs as vllm_envs
from torch_tpu import api
from torch_tpu._internal.profiler import profiler_api
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
        self.profile_context = None
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

        # For natural vLLM MP TP on single-host, pin each worker process to
        # one TPU chip based on local_rank.
        if (multihost_backend != "ray"
                and self.parallel_config.pipeline_parallel_size == 1
                and self.parallel_config.tensor_parallel_size >= 1):
            if tpu_visible_chips:
                os.environ["TPU_VISIBLE_CHIPS"] = tpu_visible_chips
            else:
                os.environ["TPU_VISIBLE_CHIPS"] = str(self.local_rank)
            logger.info(
                "Pin TPU worker to local chip | rank=%s | local_rank=%s "
                "| TPU_VISIBLE_CHIPS=%s", self.rank, self.local_rank,
                os.environ["TPU_VISIBLE_CHIPS"])

            # Single-host TP assumption: workers are spawned locally, one
            # process per chip. TorchTPU bootstrap expects torchrun-style env
            # vars before TPU runtime initialization (api.tpu_device()).
            # TODO: (geyuhao) support multihost
            os.environ["RANK"] = str(self.rank)
            os.environ["LOCAL_RANK"] = str(self.local_rank)
            os.environ["WORLD_SIZE"] = str(self.parallel_config.world_size)
            os.environ["LOCAL_WORLD_SIZE"] = str(
                self.parallel_config.world_size)

            parsed = urlparse(self.distributed_init_method)
            if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
                raise ValueError(
                    "Expected tcp://<host>:<port> distributed_init_method for "
                    "single-host TPU MP, got: "
                    f"{self.distributed_init_method!r}")
            os.environ.setdefault("MASTER_ADDR", parsed.hostname)
            os.environ.setdefault("MASTER_PORT", str(parsed.port))

        if not self.devices:
            self.devices = []
            self.devices.append(api.tpu_device())

        # Initialize vLLM distributed state using true rank/world-size so TP
        # uses native vLLM model-parallel groups.
        from vllm.platforms import current_platform
        dist_backend = current_platform.get_worker_distributed_backend(
            self.parallel_config.world_size)

        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment(
                world_size=self.parallel_config.world_size,
                rank=self.rank,
                local_rank=self.local_rank,
                distributed_init_method=self.distributed_init_method,
                backend=dist_backend,
            )
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=self.parallel_config.
                tensor_parallel_size,
                pipeline_model_parallel_size=self.parallel_config.
                pipeline_parallel_size,
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
        # VLLM directive of the percentage of HBM memory the model executor can use
        gpu_memory_utilization = self.cache_config.gpu_memory_utilization

        self.model_runner.profile_run(self.model_runner.max_num_tokens)

        total_hbm_limit = total_hbm_used = 0
        for device in self.devices:
            free_memory, limit_memory = torch.accelerator.get_memory_info(
                device)
            total_hbm_used += (limit_memory - free_memory)
            total_hbm_limit += limit_memory

        total_hbm_limit_cap = total_hbm_limit * gpu_memory_utilization

        # HACK: The profiling run executes without KV cache since it is used to infer it.
        # To do so the attention kernel returns early.
        # As of Feb 23rd 2026 the attention kernel in the regular forward pass will create
        # a copy of the KV cache. Since the profiling run has no way of catching this
        # we adjust for this copy by dividing the `total_hbm_avail` with 2 to account
        # for the extra copy in the attention kernel.
        # This should be removed once the copy in the attention kernel is removed.
        total_hbm_avail = int(total_hbm_limit_cap - total_hbm_used) // 2

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
        if self.profile_dir is None:
            logger.warning("Profile directory is not set. Skipping profiling.")
            return

        if is_start:
            logger.info(
                f"Starting TorchTPU profiler trace at {self.profile_dir}...")
            handler = profiler_api.xprof_trace_handler(
                dir_name=self.profile_dir)
            self.profile_context = profiler_api.profile(activities=[
                profiler_api.ProfilerActivity.CPU,
                profiler_api.ProfilerActivity.TPU
            ],
                                                        on_trace_ready=handler)
            self.profile_context.__enter__()
        else:
            if self.profile_context is not None:
                logger.info("Stopping TorchTPU profiler trace...")
                self.profile_context.__exit__(None, None, None)
                logger.info(f"Profiler trace saved to {self.profile_dir}")
            else:
                logger.warning(
                    "Profiler context is not set. Cannot stop profiler.")

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
