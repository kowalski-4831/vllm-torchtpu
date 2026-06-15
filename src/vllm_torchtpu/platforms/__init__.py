# ruff: noqa
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


def register_tpu_platform() -> str:
    """vLLM out-of-tree platform plugin entry point."""
    return "vllm_torchtpu.platforms.tpu_platform.TpuPlatform"
