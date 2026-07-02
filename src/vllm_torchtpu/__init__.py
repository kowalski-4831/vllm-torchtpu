# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm_torchtpu.env_override  # noqa: F401
from vllm_torchtpu import envs
from vllm_torchtpu import tpu_info as ti
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _patch_vllm_tpu_group_custom_ops() -> None:
    """Disable vLLM custom collective ops for TPU in this TorchTPU integration.

    vLLM's `torch.ops.vllm.all_reduce/all_gather` custom ops are not available
    on the TorchTPU backend in this repo. Force GroupCoordinator to use
    communicator-backed collectives instead.

    TODO (geyuhao): Do we actually need to support torch.ops.vllm.* on TPU? Currently
    this patch will use torch.distributed.all_reduce/all_gather.
    """

    from vllm.distributed.parallel_state import GroupCoordinator

    if getattr(GroupCoordinator, "_tpu_no_custom_collective_patch", False):
        return

    original_init = GroupCoordinator.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.use_custom_op_call = False

    GroupCoordinator.__init__ = patched_init
    GroupCoordinator._tpu_no_custom_collective_patch = True
    logger.info("Applied TPU patch: disable vLLM custom collective ops.")


def _patch_default_moe_runner_select_forward() -> None:
    """Keep DefaultMoERunner on the direct ``_moe_forward`` path under OOT.

    Upstream gates this on ``is_tpu()``; under our OOT plugin that's False
    and we'd take ``torch.ops.vllm.moe_forward``, which captures weight
    tensors as closures and trips torch_tpu's MLIR builder.
    """
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as _dmr

    if getattr(_dmr.MoERunner, "_tpu_select_forward_patch", False):
        return

    def patched_select(self):
        return (_dmr._moe_forward
                if self.shared_experts is None else _dmr._moe_forward_shared)

    _dmr.MoERunner._select_forward = patched_select
    _dmr.MoERunner._tpu_select_forward_patch = True
    logger.info(
        "Applied TPU patch: DefaultMoERunner uses direct _moe_forward.")


def _patch_vllm_disable_compile_ranges() -> None:
    """Force TPU to compile only fixed bucketed sizes, never dynamic ranges.

    ``VllmConfig._set_compile_ranges()`` always appends
    ``scheduler_config.max_num_batched_tokens`` to ``compile_ranges_endpoints``,
    so ``PiecewiseBackend`` builds a dynamic ``Range(1, max)`` and compiles it
    with backed ``SymInt`` shapes. The torch_tpu backend rejects SymInts ("does
    not support dynamic shape" / "No shape env"). The TPU runner pads every shape
    to an exact ``compile_size``, so dynamic ranges are never needed.

    Patch the consumption point: ``CompilationConfig.get_compile_ranges() -> []``.
    Clearing the field (e.g. in the worker) does not stick --
    ``_set_compile_ranges`` rewrites it on every config reconstruction
    (Qwen3-VL ``with_hf_config`` -> ``replace``), and the compile-time
    compilation_config is not the instance a platform/worker hook can reach --
    so a class-level override is the robust fix.
    """
    from vllm.config.compilation import CompilationConfig

    def get_compile_ranges(self):
        return []

    CompilationConfig.get_compile_ranges = get_compile_ranges
    logger.info("Applied TPU patch: disable dynamic compile_ranges (static "
                "compile_sizes only).")


def _patch_disable_sequence_parallel_moe() -> None:
    """Disable vLLM's sequence-parallel MoE on TPU.

    The sequence-parallel path is tied to vLLM all2all MoE backends. The TPU
    communicator handles EP with unsplit token buffers.
    """
    from vllm.config.parallel import ParallelConfig

    if getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        return

    ParallelConfig.use_sequence_parallel_moe = property(lambda self: False)
    ParallelConfig._tpu_no_sp_moe_patch = True
    logger.info("Applied TPU patch: disable sequence-parallel MoE.")


def _patch_disable_dp_ubatch() -> None:
    """Disable DP microbatching (DBO) on TPU.

    TPU does not support vLLM DP microbatching yet. Force the post-sync ubatch
    decision off so DP uses one forward per engine step.
    """
    from vllm.v1.worker import dp_utils

    if getattr(dp_utils, "_tpu_no_ubatch_patch", False):
        return

    dp_utils._post_process_ubatch = lambda tensor, num_ubatches: False
    dp_utils._tpu_no_ubatch_patch = True
    logger.info("Applied TPU patch: disable DP microbatching.")


def _patch_moe_no_ep_tp_scope() -> None:
    """Keep non-EP MoE tensor parallelism scoped to each DP engine.

    Upstream vLLM flattens TP across DP for MoE when expert parallelism is
    disabled. This temporary TPU patch avoids that path until the TPU dummy
    batch path can keep no-EP MoE ranks in the same lockstep as upstream
    model-internal DP.
    """
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.fused_moe.layer import \
        FusedMoEParallelConfig

    if getattr(FusedMoEParallelConfig, "_tpu_no_ep_tp_scope_patch", False):
        return

    original_make = FusedMoEParallelConfig.make

    # TODO: Remove this patch in a follow-up PR. TorchTPU should match vLLM's
    # no-EP MoE convention by flattening DP*TP and extending the coordinated
    # dummy/lockstep execution used for EP to no-EP MoE as well.
    def patched_make(tp_size_, pcp_size_, dp_size_, sp_size_,
                     vllm_parallel_config):
        use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1
                  and vllm_parallel_config.enable_expert_parallel)
        if use_ep:
            return original_make(tp_size_, pcp_size_, dp_size_, sp_size_,
                                 vllm_parallel_config)

        dp_rank = (parallel_state.get_dp_group().rank_in_group
                   if dp_size_ > 1 else 0)
        pcp_rank = (parallel_state.get_pcp_group().rank_in_group
                    if pcp_size_ > 1 else 0)
        tp_rank = (0 if tp_size_ == 1 else
                   parallel_state.get_tensor_model_parallel_rank())
        return FusedMoEParallelConfig(
            tp_size=tp_size_,
            tp_rank=tp_rank,
            pcp_size=pcp_size_,
            pcp_rank=pcp_rank,
            dp_size=dp_size_,
            dp_rank=dp_rank,
            ep_size=1,
            ep_rank=0,
            sp_size=sp_size_,
            use_ep=False,
            all2all_backend=vllm_parallel_config.all2all_backend,
            enable_eplb=vllm_parallel_config.enable_eplb,
        )

    FusedMoEParallelConfig.make = staticmethod(patched_make)
    FusedMoEParallelConfig._tpu_no_ep_tp_scope_patch = True
    logger.info(
        "Applied TPU patch: scope non-EP MoE TP inside each DP engine.")


def _patch_multiproc_worker_global_rank_env() -> None:
    """Set TorchTPU binding env on spawned workers before torch_tpu import.

    Native vLLM DP passes per-engine rank/local_rank to workers and computes
    global torch.distributed rank later from ParallelConfig. TorchTPU currently
    reads RANK/LOCAL_RANK/WORLD_SIZE earlier for physical chip/PjRt binding, so
    this shim exposes the unified single-host DP*TP slice to TorchTPU without
    changing the worker rank arguments passed through vLLM.
    """
    import os as _os

    from vllm.v1.executor.multiproc_executor import WorkerProc

    if getattr(WorkerProc, "_tpu_global_rank_env_patch", False):
        return
    _orig = WorkerProc.make_worker_process

    def _wrapped(vllm_config, local_rank, rank, *args, **kwargs):
        pc = vllm_config.parallel_config
        dp_size = int(_os.environ.get("TORCH_TPU_DP_SIZE",
                                      "0")) or pc.data_parallel_size
        if dp_size > 1:
            dp_rank = getattr(pc, "data_parallel_index", None)
            if dp_rank is None:
                dp_rank = pc.data_parallel_rank or 0
            lw = pc.world_size
            global_rank = lw * dp_rank + rank
            chip_rank = lw * dp_rank + local_rank
            global_world = lw * dp_size
            _os.environ["RANK"] = str(global_rank)
            # Single-host TorchTPU indexes chips in the unified DP*TP slice.
            _os.environ["LOCAL_RANK"] = str(chip_rank)
            _os.environ["WORLD_SIZE"] = str(global_world)
            _os.environ["LOCAL_WORLD_SIZE"] = str(global_world)
            logger.info(
                "Applied TPU patch: worker spawn env RANK=%d LOCAL_RANK=%d "
                "WORLD_SIZE=%d (dp_rank=%d rank=%d local_rank=%d)",
                global_rank, chip_rank, global_world, dp_rank, rank,
                local_rank)
        else:
            _os.environ["RANK"] = str(rank)
            _os.environ["LOCAL_RANK"] = str(local_rank)
            _os.environ["WORLD_SIZE"] = str(pc.world_size)
            _os.environ["LOCAL_WORLD_SIZE"] = str(pc.world_size)
        return _orig(vllm_config, local_rank, rank, *args, **kwargs)

    WorkerProc.make_worker_process = staticmethod(_wrapped)
    WorkerProc._tpu_global_rank_env_patch = True
    logger.info("Applied TPU patch: MultiprocExecutor worker binding env.")


if "proxy" in envs.JAX_PLATFORMS:
    logger.info("Running vLLM on TPU via Pathways proxy.")
    # Must run pathwaysutils.initialize() before any JAX operations
    try:
        import traceback

        import pathwaysutils
        import vllm
        from vllm.platforms import (resolve_current_platform_cls_qualname,
                                    resolve_obj_by_qualname)
        pathwaysutils.initialize()
        logger.info("Module pathwaysutils is imported.")

        # Pathways requires eager resolution of vllm.current_platform instead of
        # lazy resolution in the normal code path. Since this part involves
        # global topology discovery across multiple hosts, the platform
        # resolution must happen before other components are loaded.
        logger.info("Eagerly resolving vLLM current_platform for Pathways.")
        platform_cls_qualname = resolve_current_platform_cls_qualname()
        resolved_platform_instance = resolve_obj_by_qualname(
            platform_cls_qualname)()
        vllm.platforms._current_platform = resolved_platform_instance
        vllm.platforms._init_trace = "".join(traceback.format_stack())
        logger.info(
            f"vLLM platform resolved to: {resolved_platform_instance.__class__.__name__}"
        )

    except Exception as e:
        logger.error(
            f"Error occurred while importing pathwaysutils or logging TPU info: {e}"
        )
else:
    # Either running on TPU or CPU
    try:
        logger.info(f"TPU info: node_name={ti.get_node_name()} | "
                    f"tpu_type={ti.get_tpu_type()} | "
                    f"worker_id={ti.get_node_worker_id()} | "
                    f"num_chips={ti.get_num_chips()} | "
                    f"num_cores_per_chip={ti.get_num_cores_per_chip()}")
    except Exception as e:
        logger.error(
            f"Error occurred while logging TPU info: {e}. Are you running on CPU?"
        )
