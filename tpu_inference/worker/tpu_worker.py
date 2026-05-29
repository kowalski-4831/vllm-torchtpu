# SPDX-License-Identifier: Apache-2.0

# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import tpu_inference.env_override  # noqa: F401  # isort: skip

import os
import time
from typing import Dict, Tuple
from urllib.parse import urlparse

import torch
import torch_tpu  # noqa: F401
from torch_tpu._internal.profiler import profiler_api
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.v1 import utils as vllm_utils
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.worker_base import WorkerBase

from tpu_inference import envs, utils
from tpu_inference.distributed import jax_parallel_state
from tpu_inference.distributed.utils import get_node_id
from tpu_inference.layers.vllm.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class TPUWorker(WorkerBase):

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
        # Re-apply patches that were done in check_and_update_config.
        # Workers may be spawned (not forked) so module-level patches
        # from the parent process are lost.
        from tpu_inference.platforms.tpu_platform import apply_tpu_patches
        apply_tpu_patches()

        if envs.MODEL_IMPL_TYPE != "vllm":
            raise ValueError("Only vLLM models are supported")

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)
        # WorkerBase initializes self.device=None which makes vLLM's
        # MultiprocExecutor.async_output_busy_loop call
        # current_platform.set_device(None) → TPU has no torch device-switching
        # API (pinning is via TPU_VISIBLE_CHIPS env var). Drop the attribute
        # so hasattr(self.worker, "device") returns False and the call skips.
        if hasattr(self, "device"):
            del self.device

        # TPU compilation requires static shapes, so clear dynamic compile
        # ranges vLLM might have added.
        self.compilation_config.compile_ranges_endpoints = []

        # TPU-specific extras not in WorkerBase.
        self.devices = devices if devices is not None else []
        self.device_ranks = set(device.id for device in self.devices)
        self.prev_worker_ip = prev_worker_ip

        if self.cache_config.cache_dtype == "auto":
            model_dtype = self.model_config.dtype
            self.kv_cache_dtype = (TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
                                   if isinstance(model_dtype, str) else
                                   model_dtype)
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # Lazy import to avoid importing torch before initializing.
            from vllm.utils.import_utils import init_cached_hf_modules
            init_cached_hf_modules()

        # TPU profiler: only on rank 0 single-host, or every PP worker.
        self.profile_dir: str | None = None
        self.profile_context = None
        torch_profiler_dir = os.getenv("VLLM_TORCH_PROFILER_DIR")
        pp_size = self.parallel_config.pipeline_parallel_size
        if torch_profiler_dir and pp_size == 1 and self.rank < 1 and (
                not self.devices or 0 in self.device_ranks):
            # Only 1 active profiler session per server is allowed.
            self.profile_dir = torch_profiler_dir
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        self.profile_dir)
        elif pp_size > 1 and torch_profiler_dir:
            # PP uses MPMD: profile every worker.
            self.profile_dir = os.path.join(
                torch_profiler_dir,
                f"pprank_{self.rank}_ppworldsize_{pp_size}")
            os.makedirs(self.profile_dir, exist_ok=True)

        # step_counter is used to calc uuid for intermediate tensor transfer.
        self.step_counter = 0

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        # vLLM's MultiprocExecutor passes rank/local_rank as constructor
        # args, but torch_tpu's C++ layer (discovery.cc) reads torchrun-style
        # env vars. Bridge the gap by setting them here before any TPU
        # device usage in this worker.
        # TORCH_TPU_SLICEBUILDER_ADDRESSES and TORCH_TPU_TOPOLOGY are
        # already inherited from the parent process (set by
        # prepare_tpu_environment() in tpu_platform.py).
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["WORLD_SIZE"] = str(self.parallel_config.world_size)
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        if not local_world_size:
            local_world_size = self.parallel_config.world_size
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

        parsed = urlparse(self.distributed_init_method)
        if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
            raise ValueError(
                "Expected tcp://<host>:<port> distributed_init_method, "
                f"got: {self.distributed_init_method!r}")
        os.environ.setdefault("MASTER_ADDR", parsed.hostname)
        os.environ.setdefault("MASTER_PORT", str(parsed.port))

        if not self.devices:
            self.devices = [torch.device("tpu")]

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

        # TODO: Enable PP support. The old JAX-based PP init
        # (jax_parallel_state.init_pp_distributed_environment) was removed
        # during the torch_tpu migration. PP will need KV transfer init
        # and proper rank assignment via torch.distributed.
        is_first_rank = True
        is_last_rank = True
        if self.parallel_config.pipeline_parallel_size > 1:
            is_first_rank = self.rank == 0
            is_last_rank = (
                self.rank == self.parallel_config.pipeline_parallel_size - 1)

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
        jax_parallel_state.connect(self.prev_worker_ip, self.rank - 1)

    def determine_available_memory(self) -> int:
        # VLLM directive of the percentage of HBM memory the model executor can use
        self.model_runner.profile_run(self.model_runner.max_num_tokens)

        gpu_memory_utilization = self.cache_config.gpu_memory_utilization
        budget = utils.compute_hbm_budget(self.devices, gpu_memory_utilization)

        total_hbm_limit_gb = round(budget.total_limit / utils.GBYTES, 2)
        total_hbm_limit_cap_gb = round(budget.cap / utils.GBYTES, 2)
        total_hbm_used_gb = round(budget.total_used / utils.GBYTES, 2)
        kv_cache_headroom_gb = round(budget.headroom / utils.GBYTES, 2)
        total_hbm_avail_gb = round(budget.available / utils.GBYTES, 2)
        logger.info(f"Memory statistics | "
                    f"{total_hbm_limit_gb=}GiB | "
                    f"{total_hbm_limit_cap_gb=}GiB | "
                    f"{total_hbm_used_gb=}GiB | "
                    f"{kv_cache_headroom_gb=}GiB | "
                    f"{total_hbm_avail_gb=}GiB")

        if budget.available <= 0:
            raise ValueError(f"{total_hbm_used_gb=}GiB exceeds "
                             f"{total_hbm_limit_cap_gb=}GiB by "
                             f"{-total_hbm_avail_gb}GiB. Please consider "
                             f"increasing --gpu-memory-utilization from "
                             f"{gpu_memory_utilization} to a larger value, "
                             "or decreasing TPU_KV_CACHE_HEADROOM_MIB if "
                             "this run has a known smaller TPU runtime "
                             "headroom requirement.")
        return budget.available

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self):
        return self.model_runner.take_draft_token_ids()

    def profile(self,
                is_start: bool = True,
                profile_prefix: str | None = None):
        if self.profile_dir is None:
            logger.warning("Profile directory is not set. Skipping profiling.")
            return

        profile_dir = self.profile_dir
        if profile_prefix:
            profile_dir = os.path.join(profile_dir, profile_prefix)
            os.makedirs(profile_dir, exist_ok=True)

        if is_start:
            logger.info(
                f"Starting TorchTPU profiler trace at {profile_dir}...")
            handler = profiler_api.xprof_trace_handler(dir_name=profile_dir)
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
                logger.info(f"Profiler trace saved to {profile_dir}")
            else:
                logger.warning(
                    "Profiler context is not set. Cannot stop profiler.")

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> float:
        start = time.perf_counter()
        self.model_runner.capture_model()
        compilation_time = time.perf_counter() - start
        self.compilation_config.compilation_time = compilation_time
        return compilation_time

    def get_model(self):
        return self.model_runner.get_model()

    def get_supported_tasks(self):
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self):
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def get_node_kv_ip_port(self) -> tuple[int, str, int]:
        pass

    def sync_weights(self,
                     updated_weights,
                     mappings: Dict[str, Tuple[str, Tuple[str]]],
                     transpose_keys: Dict[str, Tuple[int]],
                     reshard_fn=None) -> None:
        return self.model_runner._sync_weights(
            updated_weights=updated_weights,
            mappings=mappings,
            transpose_keys=transpose_keys,
            reshard_fn=reshard_fn,
        )

    # Ray executor doesn't need handshake metadata — kv_parameters go
    # through the proxy server.
    def get_kv_connector_handshake_metadata(self) -> None:
        pass
