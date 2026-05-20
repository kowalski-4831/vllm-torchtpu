# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import tpu_inference.env_override  # noqa: F401
from tpu_inference import envs
from tpu_inference import tpu_info as ti
from tpu_inference.logger import init_logger

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
    from vllm.model_executor.layers.fused_moe.runner import \
        default_moe_runner as _dmr

    if getattr(_dmr.DefaultMoERunner, "_tpu_select_forward_patch", False):
        return

    def patched_select(self, layer):
        return (_dmr._moe_forward
                if self.shared_experts is None else _dmr._moe_forward_shared)

    _dmr.DefaultMoERunner._select_forward = patched_select
    _dmr.DefaultMoERunner._tpu_select_forward_patch = True
    logger.info(
        "Applied TPU patch: DefaultMoERunner uses direct _moe_forward.")


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
