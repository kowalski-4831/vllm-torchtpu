# ruff: noqa
from tpu_inference.platforms.tpu_platform import TpuPlatform


def register_tpu_platform() -> str:
    """vLLM out-of-tree platform plugin entry point."""
    return "tpu_inference.platforms.tpu_platform.TpuPlatform"
