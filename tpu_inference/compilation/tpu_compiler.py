# SPDX-License-Identifier: Apache-2.0
"""TPU compiler adaptor for vLLM's VLLM_COMPILE mode.

This module implements the CompilerInterface for TPU, enabling
PiecewiseBackend to compile FX graphs via TorchTPU's FX->MLIR->PjRt
pipeline. Each shape bucket gets its own compiled executable.
"""

import copy
import hashlib
import logging
import os
import pickle
from typing import Any, Callable

import torch
import torch.fx as fx
from torch_tpu._internal.compile._backend import TpuBackend
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Type alias for the compile range tuple
Range = tuple[int, int]

_tpu_backend = TpuBackend()


def _ensure_tuple_output(graph: fx.GraphModule) -> tuple[fx.GraphModule, bool]:
    """Wrap a graph module to return a tuple if it returns a single tensor.

    aot_autograd requires graph outputs to be tuples. PiecewiseBackend
    sub-graphs may return a single tensor.

    Returns (wrapped_graph, was_wrapped).
    """
    output_node = None
    for node in graph.graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return graph, False

    output_args = output_node.args[0]
    if isinstance(output_args, (tuple, list)):
        return graph, False

    # Single return value — wrap in a tuple
    with graph.graph.inserting_before(output_node):
        output_node.args = ((output_args, ), )
    graph.graph.lint()
    graph.recompile()
    return graph, True


class TpuCompilerAdaptor(CompilerInterface):
    """CompilerInterface implementation for TPU.

    Delegates to TpuBackend which handles aot_autograd decomposition
    and FX->MLIR->PjRt compilation. Supports per-shape disk caching
    via pickle serialization of compiled executables.
    """

    name = "tpu"

    def __init__(self):
        self.cache_dir: str | None = None
        self._disable_cache = False

    def initialize_cache(self,
                         cache_dir: str,
                         disable_cache: bool = False,
                         prefix: str = "") -> None:
        self.cache_dir = cache_dir
        self._disable_cache = disable_cache

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """Hash torch_tpu version for cache invalidation."""
        import torch_tpu
        version = getattr(torch_tpu, "__version__", "unknown")

        hash_str = hashlib.sha256(
            f"torch_tpu={version}".encode()).hexdigest()[:10]
        return hash_str

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        """Compile a Dynamo-level FX graph to a TPU PjRt executable.

        Delegates to TpuBackend which handles aot_autograd + fx_to_mlir
        + compile_mlir internally.
        """
        logger.info(
            "[TpuCompilerAdaptor] Compiling FX graph for range %s",
            compile_range,
        )

        # aot_autograd (inside TpuBackend) requires tuple outputs.
        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        compiled_fn = _tpu_backend(graph, example_inputs)

        if was_wrapped:
            inner_fn = compiled_fn

            def unwrap_fn(*args, **kwargs):
                result = inner_fn(*args, **kwargs)
                if isinstance(result, (tuple, list)) and len(result) == 1:
                    return result[0]
                return result

            compiled_fn = unwrap_fn

        # Save the per-shape PjRt executable to disk.
        # We pickle the inner _TorchTpuCompiledExecutable (which has
        # __reduce__), not the aot_autograd wrapper (which has
        # unpicklable closures).
        handle = None
        if (key is not None and self.cache_dir is not None
                and not self._disable_cache
                and _tpu_backend._compiled_executables):
            try:
                inner_exe = _tpu_backend._compiled_executables[-1]
                save_path = os.path.join(self.cache_dir, key)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(inner_exe, f)
                handle = (key, save_path, was_wrapped)
                logger.info(
                    "[TpuCompilerAdaptor] Saved compiled executable to %s",
                    save_path,
                )
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] Failed to save executable: %s", e)

        return compiled_fn, handle

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]:
        """Load a compiled executable from disk.

        Unpickles the _TorchTpuCompiledExecutable, then wraps it with
        aot_autograd (using the same graph) to reconstruct the full
        callable. This skips FX->MLIR->compile but still sets up the
        aot_autograd runtime wrapper that handles input mutations.
        """
        assert isinstance(handle, tuple) and len(handle) == 3
        key, path, was_wrapped = handle
        logger.info("[TpuCompilerAdaptor] Loading compiled executable from %s",
                    path)

        with open(path, "rb") as f:
            inner_exe = pickle.load(f)

        # Wrap to filter non-tensor args if needed (matching compile path)
        has_non_tensor = any(not isinstance(a, torch.Tensor)
                             for a in example_inputs)
        if has_non_tensor:
            from torch_tpu._internal.compile._backend import \
                _TensorFilterExecutable
            cached_exe = _TensorFilterExecutable(inner_exe)
        else:
            cached_exe = inner_exe

        # TODO: this can be further optimized
        # Re-create the aot_autograd wrapper around the cached executable.
        # The compile path runs: aot_autograd(fw_compiler=...)(graph, inputs)
        # which sets up runtime logic to handle input mutations  by mapping
        # extra outputs back to input tensors.
        from torch._dynamo.backends.common import aot_autograd

        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        def _cached_compiler(gm, example_inputs):
            return cached_exe

        compiled_fn = aot_autograd(
            fw_compiler=_cached_compiler,
            keep_inference_input_mutations=False,
        )(graph, example_inputs)

        if was_wrapped:
            inner_fn = compiled_fn

            def unwrap_fn(*args, **kwargs):
                result = inner_fn(*args, **kwargs)
                if isinstance(result, (tuple, list)) and len(result) == 1:
                    return result[0]
                return result

            compiled_fn = unwrap_fn

        return compiled_fn
