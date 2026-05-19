# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import contextlib
import dataclasses
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, cast

# TODO: Remove this after jax dependency is removed
import jax
import numpy as np
import torch
import torch.nn as nn
import torch_tpu  # noqa: F401
import vllm.envs as envs
# TODO: Remove this after jax dependency is removed
from jax.sharding import Mesh
from packaging import version
from torch_tpu._internal import sync
from vllm.config import (CUDAGraphMode, ParallelConfig, VllmConfig,
                         get_layers_from_vllm_config, set_current_vllm_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import (Attention,
                                                  ChunkedLocalAttention,
                                                  MLAAttention)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec, SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsLists,
                             LogprobsTensors, ModelRunnerOutput)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import AttentionGroup, bind_kv_cache

from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilder,
    AttentionMetadataBuilderContext)
from tpu_inference.layers.vllm.attention import (TPU_STR_DTYPE_TO_TORCH_DTYPE,
                                                 PallasAttentionBackend)
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from tpu_inference.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID, AsyncPreResults, AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput


@dataclass
class ExecuteModelState:
    scheduler_output: "SchedulerOutput"
    logits_list: list[torch.Tensor]
    num_reqs_list: list[int]


def _substitute_placeholder_token(
        input_ids: torch.Tensor, token_in_tpu_cur_input_indices: torch.Tensor,
        token_in_tpu_pre_next_tokens_indices: torch.Tensor,
        next_tokens: torch.Tensor):
    """Substitute placeholder tokens from TPU for async scheduler.

    Padding scheme (set up by `_apply_async_token_substitution`):
      - `token_in_tpu_cur_input_indices`: real-slot indices followed by all
        other indices in ascending order; the tail entries re-target slots
        that should keep their current values.
      - `token_in_tpu_pre_next_tokens_indices`: real `next_tokens` indices
        followed by `-1` sentinels at the padded positions.
    The `-1` sentinel is the mask source; the program is shape-only so it
    does not recompile when the active count changes.

    Args:
        input_ids: device tensor [N] where N is the bucketed input size.
        token_in_tpu_cur_input_indices: int tensor [N], destination slots.
        token_in_tpu_pre_next_tokens_indices: int tensor [N], source slots
            in `next_tokens`; -1 marks a padding slot to leave unchanged.
        next_tokens: int tensor of tokens from the previous async step.
    Return:
        input_ids with real placeholders replaced; padding slots untouched.
    """
    assert input_ids.shape[0] == token_in_tpu_cur_input_indices.shape[
        0] == token_in_tpu_pre_next_tokens_indices.shape[0]
    mask = token_in_tpu_pre_next_tokens_indices >= 0
    # clamp_min(0) gives a safe in-range gather index for the -1 sentinel
    # slots; their gathered values are discarded by `mask` in `where`.
    safe_idx = torch.clamp_min(token_in_tpu_pre_next_tokens_indices, 0)
    new_token_values = next_tokens[safe_idx].to(input_ids.dtype)
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = torch.where(mask, new_token_values, original_values)
    input_ids.scatter_(0, token_in_tpu_cur_input_indices, update_values)
    return input_ids


logger = init_logger(__name__)

# Smallest output size
MIN_NUM_SEQS = 8
SAMPLING_EPS = 1e-5


def _validate_libtpu_version() -> None:
    """Validate libtpu opportunistically if it is present."""
    libtpu_version = importlib_metadata.version("libtpu")
    parsed_version = version.parse(libtpu_version)
    if parsed_version < version.parse("0.0.35"):
        raise RuntimeError(
            "Argmax is having accuracy issue with libtpu < 0.0.35")
    if parsed_version < version.parse("0.0.36"):
        logger.warning_once(
            "libtpu < 0.0.36 may enable "
            "--xla_tpu_impure_enable_large_2nd_minor_layout by default, "
            "which can hurt performance. "
            "Upgrade libtpu to >= 0.0.36, or set "
            "LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=false."
        )


@contextlib.contextmanager
def _torch_tpu_wrapper():
    """Alias torch.cuda.* to torch.tpu.* during inherited GPUModelRunner init.

    Mirrors vllm/v1/worker/xpu_model_runner.py:_torch_cuda_wrapper. The parent
    GPUModelRunner constructs torch.cuda.Stream/Event for comm and async copy;
    this wrapper makes those resolve to the TPU equivalents (verified present
    on torch.tpu: Stream, Event, current_stream, default_stream, stream,
    set_stream, synchronize, set_device, current_device, device_count).

    The 5 APIs not on torch.tpu are stubbed:
      - mem_get_info -> torch.accelerator.get_memory_info (works on TPU)
      - empty_cache  -> no-op (XLA manages HBM)
      - graph / CUDAGraph / graph_pool_handle -> never reached because we
        disable cudagraphs in __init__ via compilation_config overrides.
    """
    saved = {}
    aliased = ("Stream", "Event", "current_stream", "default_stream", "stream",
               "set_stream", "synchronize", "set_device", "current_device",
               "device_count", "is_available")
    for name in aliased:
        saved[name] = getattr(torch.cuda, name, None)
        setattr(torch.cuda, name, getattr(torch.tpu, name))
    saved["mem_get_info"] = getattr(torch.cuda, "mem_get_info", None)
    torch.cuda.mem_get_info = lambda *a, **kw: torch.accelerator.get_memory_info(
    )
    saved["empty_cache"] = getattr(torch.cuda, "empty_cache", None)
    torch.cuda.empty_cache = lambda: None
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                if hasattr(torch.cuda, k):
                    delattr(torch.cuda, k)
            else:
                setattr(torch.cuda, k, v)


@contextlib.contextmanager
def _bypass_torch_compile(model: nn.Module):
    """HACK: vLLM wraps the model backbone with
    `TorchCompileWithNoGuardsWrapper` which overrides `__call__` to route
    through `torch.compile`, making neither
    `torch.compiler.set_stance("force_eager")` nor
    `torch._dynamo.config.patch(disable=True)` work here.
    """
    if hasattr(model, "get_language_model"):
        compiled_model = model.get_language_model().model
    else:
        compiled_model = model.model

    compiled_model_cls = compiled_model.__class__
    original_call = compiled_model_cls.__call__
    compiled_model_cls.__call__ = compiled_model_cls.forward
    try:
        yield
    finally:
        compiled_model_cls.__call__ = original_call


# Recompilation-avoidance contract:
#   1. Input prep happens on CPU; H2D via `cpu_tensor.to(xla_device)`.
#   2. Forward is split into 4 `@torch.compile(backend="tpu")` subgraphs
#      (backbone, compute_selected_logits, sample_from_logits/structured_decode,
#      gather_logprobs) so dummy_run and execute_model trace identically.
#   3. `_dummy_run` exercises every padding bucket so all shapes are AOT-
#      compiled before the first real request.
class TPUModelRunner(GPUModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        original_parallel_config: ParallelConfig | None = None,
    ):
        # Disable cudagraphs before parent init so its dispatch self-disables.
        # TPU uses AOT bucket precompile (_precompile_* methods) instead.
        vllm_config.compilation_config.cudagraph_capture_sizes = []
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        with _torch_tpu_wrapper():
            super().__init__(vllm_config, device)
        _validate_libtpu_version()

        # Parent already set: vllm_config, *_config, device, pin_memory, dtype,
        # max_model_len, max_num_reqs, max_num_tokens, num_query_heads,
        # inputs_embeds_size, mm_registry, uses_mrope, supports_mm_inputs,
        # kv_caches, encoder_cache, shared_kv_cache_layers, requests,
        # num_prompt_logprobs, input_batch, etc. The block below is only the
        # TPU-specific delta + overrides.

        # TPU-only custom kwarg + alias kept for backwards-compat with worker.
        self.original_parallel_config = original_parallel_config
        self.device_config = vllm_config.device_config

        # Set by `_update_mamba_page_size_padded` for hybrid attention+mamba
        # models so vLLM sees a uniform page size across groups.
        self._hybrid_uniform_page_size_bytes: int | None = None

        # TPU env-var flags.
        self.check_recompilation = envs.VLLM_XLA_CHECK_RECOMPILATION
        self.use_spmd = envs.VLLM_XLA_USE_SPMD

        # XLA graph tracker (TPU-specific debug aid).
        self.enforce_eager = self.model_config.enforce_eager
        self.num_xla_graphs = 0
        self._update_num_xla_graphs("init")
        # torchTPU doesn't support SymInt yet, so widen Dynamo cache limit.
        torch._dynamo.config.cache_size_limit = 1024

        # Override parent's kv_cache_dtype with the TPU mapping (handles
        # auto -> bfloat16 for our supported dtype set).
        cache_config = self.cache_config
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = (TPU_STR_DTYPE_TO_TORCH_DTYPE[self.dtype] if
                                   isinstance(self.dtype, str) else self.dtype)
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]
        self._hidden_states_dtype = self.dtype

        # Compile bucket + padding (TPU AOT compile rounds inputs up).
        self.sliding_window = self.model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.most_model_len = envs.VLLM_TPU_MOST_MODEL_LEN
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.num_blocks_per_most_len_req = (cdiv(
            self.most_model_len, self.block_size) if self.most_model_len
                                            is not None else None)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(self.max_num_reqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = vllm_config.compilation_config.compile_sizes
        # Override parent's max_num_tokens with the last (largest) padding
        # bucket so all CPU staging buffers cover it.
        self.max_num_tokens = self.num_tokens_paddings[-1]
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention")
        self.num_kv_heads = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.vocab_size = self.model_config.get_vocab_size()

        # Pallas attention bookkeeping.
        self._attention_kernels_initialized = False

        # CPU staging tensors (TPU prepares inputs on CPU then transfers).
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_np = self.positions_cpu.numpy()
        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu")
        if self.uses_mrope:
            # Override parent's int64 mrope buffer with int32 so the H2D copy is
            # dtype-identical to what the TPU model expects. Parent's
            # _calc_mrope_positions writes via .cpu and .np — both route here.
            self.mrope_positions = self._make_buffer(3,
                                                     self.max_num_tokens + 1,
                                                     dtype=torch.int32)
        self.query_start_loc_cpu = torch.zeros(self.max_num_tokens + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        if self.supports_mm_inputs:
            self.is_mm_embed_cpu = torch.zeros(self.max_num_tokens,
                                               dtype=torch.bool,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        # Pallas SMEM-aware num_reqs caps.
        self.num_reqs_most_model_len = (min(
            PallasAttentionBackend.get_max_num_seqs(
                self.most_model_len, self.block_size), self.max_num_reqs,
            self.max_num_tokens) if self.most_model_len is not None else None)
        self.num_reqs_max_model_len = min(
            PallasAttentionBackend.get_max_num_seqs(self.max_model_len,
                                                    self.block_size),
            self.max_num_reqs, self.max_num_tokens)

        # Structured decoding staging tensors.
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory)
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory)
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory)
        self.sample_from_logits_func = self.sample_from_logits

        # TPU async-scheduling state (passed between execute_model and
        # sample_tokens, mirroring vLLM's async path).
        self.mm_embed_inputs: tuple[list[torch.Tensor],
                                    torch.Tensor] | None = None
        self.execute_model_state: ExecuteModelState | None = None
        self._pre_async_results: AsyncPreResults | None = None

        # Caches for decode-fast-path: avoid re-creating identical device
        # tensors across consecutive decode steps.
        self._attn_layer_names: list[str] | None = None
        self._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
        self._decode_device_cache_key: tuple | None = None
        self._cached_query_start_loc: torch.Tensor | None = None
        self._cached_logits_indices: torch.Tensor | None = None
        self._cached_request_distribution: torch.Tensor | None = None

        # JAX Mesh for shard_map ops in TPU kernels (data=1, model=tp_size).
        self.mesh = self._create_mesh_for_parallelism()

    # ----- Backend hooks overridden from GPUModelRunner -----

    def _init_device_properties(self) -> None:
        # GPU sets self.num_sms from torch.cuda.get_device_properties; TPU
        # has no SM concept. The parent's call here is during __init__ but
        # we still want a no-op override so subclass invariants hold.
        pass

    def _sync_device(self) -> None:
        torch.tpu.synchronize()

    def maybe_setup_kv_connector(self, scheduler_output) -> None:
        if not has_kv_transfer_group():
            return
        kv_connector = get_kv_transfer_group()
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata)
        # forward_context is unused by TPUConnector; pass None.
        kv_connector.start_load_kv(None)

    def maybe_wait_for_kv_save(self) -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    def get_finished_kv_transfers(self, scheduler_output):
        if not has_kv_transfer_group():
            return None, None
        kv_connector = get_kv_transfer_group()
        finished = kv_connector.get_finished(scheduler_output.finished_req_ids)
        # Mirror KVConnectorModelRunnerMixin._get_kv_connector_output:
        # metadata is bound per-step and must be cleared after use.
        kv_connector.clear_connector_metadata()
        return finished

    def _create_mesh_for_parallelism(self) -> Mesh:
        # Per-worker JAX mesh is always single-chip; vLLM native multiprocess
        # handles TP>1.
        if self.parallel_config.world_size == 1 and \
                self.parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "Single-process TPU mesh TP>1 is not supported in this path. "
                "Use vLLM multiprocess mode for --tensor-parallel-size > 1.")
        local_devices = list(jax.local_devices())
        if not local_devices:
            raise ValueError("No TPU devices are visible to create JAX mesh.")
        mesh_devices = np.asarray(local_devices[:1]).reshape((1, 1))
        mesh = Mesh(mesh_devices, axis_names=("data", "model"))
        logger.info("Init mesh | tp_size=1 | device_id=%s",
                    getattr(local_devices[0], "id", str(local_devices[0])))
        return mesh

    def _update_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        stats = torch.tpu._get_cache_stats()
        total_graphs = len(stats.per_entry_stats)
        new_compiled_graphs = total_graphs - self.num_xla_graphs
        if new_compiled_graphs == 0:
            logger.info(f"No new compiled graphs for case: {case_str}")
            return

        logger.info(f"Total Requests: {stats.num_cache_reqs}")
        logger.info(f"Total Hits: {stats.num_cache_hits}")
        logger.info(
            f"Total number of cached graphs: {total_graphs}, new: {new_compiled_graphs}, case: {case_str}"
        )
        self.num_xla_graphs += new_compiled_graphs

    def _reorder_batch_for_rpa(self,
                               scheduler_output: "SchedulerOutput") -> int:
        """Reorder active requests into an RPA-friendly decode-first layout.

        decode-only requests come first and all remaining requests stay in the
        mixed bucket. We do not create a dedicated prefill-only bucket here.

        Returns:
            Number of decode-only requests after reordering.
        """
        num_reqs = self.input_batch.num_reqs
        if num_reqs <= 0:
            return 0

        # Two-pointer partition: move decode requests (1 scheduled token) to
        # the front while preserving the existing mixed-mode fallback for the
        # remaining requests.
        i, j = 0, num_reqs - 1
        while i < j:
            i_req_id = self.input_batch.req_ids[i]
            j_req_id = self.input_batch.req_ids[j]
            assert i_req_id is not None
            assert j_req_id is not None

            if scheduler_output.num_scheduled_tokens[i_req_id] == 1:
                i += 1
            elif scheduler_output.num_scheduled_tokens[j_req_id] > 1:
                j -= 1
            else:
                self.input_batch.swap_states(i, j)
                i += 1
                j -= 1

        last_req_id = self.input_batch.req_ids[i]
        assert last_req_id is not None
        return i + int(scheduler_output.num_scheduled_tokens[last_req_id] == 1)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        layers = get_layers_from_vllm_config(
            self.vllm_config,
            (AttentionLayerBase, MambaBase),  # type: ignore[type-abstract]
        )
        block_size = self.vllm_config.cache_config.block_size
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        has_attention = any(isinstance(m, Attention) for m in layers.values())
        has_mamba = any(isinstance(m, MambaBase) for m in layers.values())
        if has_attention and has_mamba:
            self._update_mamba_page_size_padded(layers)

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            # Linear Attention path
            if isinstance(attn_module, MambaBase):
                spec = attn_module.get_kv_cache_spec(self.vllm_config)
                if spec is not None:
                    kv_cache_spec[layer_name] = spec
            # Classic Attention path
            elif isinstance(attn_module, Attention):
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
                    page_size_padded = (
                        self._hybrid_uniform_page_size_bytes if
                        self._hybrid_uniform_page_size_bytes is not None else
                        PallasAttentionBackend.get_kv_cache_page_size_bytes(
                            block_size,
                            attn_module.num_kv_heads,
                            attn_module.head_size,
                            self.kv_cache_dtype,
                        ))
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                            page_size_padded=page_size_padded,
                            sliding_window=attn_module.sliding_window,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                            page_size_padded=page_size_padded,
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
                page_size_padded = (
                    self._hybrid_uniform_page_size_bytes
                    if self._hybrid_uniform_page_size_bytes is not None else
                    PallasAttentionBackend.get_kv_cache_page_size_bytes(
                        block_size,
                        1,
                        attn_module.head_size,
                        self.kv_cache_dtype,
                    ))
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    cache_dtype_str=cache_dtype_str,
                    page_size_padded=page_size_padded,
                )
            else:
                continue

        return kv_cache_spec

    def _update_mamba_page_size_padded(
            self, layers: dict[str, AttentionLayerBase]) -> None:
        """Pad attention and mamba page sizes so vLLM's num_blocks matches
        what the TPU allocates per layer.

        For hybrid attention+mamba models, vLLM groups a tensor's memory so
        that one `KVCacheTensor` is `shared_by` one layer from each kv-cache
        group (e.g., Qwen3.5: 1 full-attn + 3 linear-attn per shared_by).
        vLLM's scheduler assumes these layers share a single physical
        tensor at the byte level — each layer's block_table indexes into
        disjoint slots of the same backing allocation, and device kernels
        reinterpret the bytes as attention KV or mamba state depending on
        which layer is accessing the slot.

        TPU `jax.Array`s are strongly typed, so we cannot overlay an
        attention tensor and a mamba tensor on the same bytes.
        `initialize_kv_cache` therefore allocates one physical array per
        layer in the `shared_by` group, carving the group's byte budget
        into separate per-layer tensors. Without the compensation done
        here, vLLM's block pool would hold `num_shared_layers`× more
        block IDs than each per-layer array has slots — the scheduler
        would hand out block IDs beyond a layer's leading dimension,
        JAX's indexed writes would silently clip them, and multiple
        requests' mamba recurrent states would collapse onto the same
        slot (corrupted state → gibberish generation).

        The fix: set every layer's reported `page_size_padded` equal to the
        full per-`shared_by` footprint — `num_attn_groups × attn_page +
        num_mamba_groups × mamba_unpadded`, where `attn_page` is the
        TPU-actual per-block bytes (from `get_attention_page_size_bytes`,
        which accounts for dtype packing like fp8) and `mamba_unpadded` is
        the natural `prod(shape) × dtype_size`. vLLM then computes a
        smaller `num_blocks` that exactly matches what we allocate per layer
        on the TPU side. HBM usage is unchanged; only the block-ID
        accounting lines up.

        Args:
            layers: A dictionary mapping layer names to their corresponding
                attention module instances (e.g., `MambaBase`, `Attention`).
        """
        attn_modules = [
            m for m in layers.values()
            if isinstance(m, (Attention, MLAAttention))
        ]
        if not attn_modules:
            return

        first_attn_module = attn_modules[0]
        num_kv_heads = first_attn_module.num_kv_heads if isinstance(
            first_attn_module, Attention) else 1
        attn_page_size_bytes = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            self.block_size,
            num_kv_heads,
            first_attn_module.head_size,
            self.kv_cache_dtype,
        )

        mamba_modules = [
            m for m in layers.values() if isinstance(m, MambaBase)
        ]
        if not mamba_modules:
            # Not hybrid; set `mamba_page_size_padded` to the attention
            # page size as a no-op default (vLLM's platform interface sets
            # this too when it detects hybrid). No layer duplication will
            # happen without mamba layers, so no block-ID mismatch to fix.
            self.cache_config.mamba_page_size_padded = attn_page_size_bytes
            return

        # Compute the unpadded mamba page size from an actual mamba module's
        # spec (shapes × dtype-size), ignoring any existing padding.
        first_mamba_spec = mamba_modules[0].get_kv_cache_spec(self.vllm_config)
        assert isinstance(first_mamba_spec, MambaSpec)
        unpadded_mamba_page_size = dataclasses.replace(
            first_mamba_spec, page_size_padded=None).page_size_bytes

        # Derive vLLM's kv-cache group layout. vLLM splits each type into
        # equal-sized groups of `group_size` layers, then allocates
        # `group_size` `KVCacheTensor`s, each `shared_by` one layer from
        # every group — so each tensor covers `num_attn_groups +
        # num_mamba_groups` layers.
        #
        # Choosing `group_size` trades off padding vs. number of groups:
        #   * group_size = max_count → fewer groups (often 1 per type),
        #     but the smaller side pads its group up to max_count layers
        #     (wastes space if max ≫ min).
        #   * group_size = min_count → no padding, but the larger side
        #     splits into `ceil(max/min)` groups.
        # vLLM's rule: pick max_count only when counts are close enough
        # that the padding is minor (max < 1.5 × min), else min_count.
        #   e.g. 12 sliding-window + 13 full-attn → max (1 group each)
        #   e.g. 10 full-attn      + 30 mamba     → min (1 attn + 3 mamba)
        #
        # This duplicates the heuristic from
        # `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`.
        # We can't call it directly because vLLM's grouping needs a fully
        # populated spec dict, while we need the group layout *before* we
        # can finish creating the specs (padding depends on grouping,
        # spec creation depends on padding). Keep in sync if that
        # heuristic ever changes — it has been stable since the hybrid
        # allocator landed.
        num_attn = len(attn_modules)
        num_mamba = len(mamba_modules)
        min_count = min(num_attn, num_mamba)
        max_count = max(num_attn, num_mamba)

        # Match vLLM exactly: float comparison, no int() truncation (matters
        # at e.g. min=3, max=4, where 4 < 4.5 but 4 < int(4.5)==4 differs).
        if max_count < min_count * 1.5:
            group_size = max_count
        else:
            group_size = min_count

        num_attn_groups = (num_attn + group_size - 1) // group_size
        num_mamba_groups = (num_mamba + group_size - 1) // group_size

        uniform_page_size_bytes = (num_attn_groups * attn_page_size_bytes +
                                   num_mamba_groups * unpadded_mamba_page_size)

        logger.info(
            "Hybrid KV cache: padding every layer spec to %d bytes "
            "(num_attn_groups=%d × attn_page=%d + "
            "num_mamba_groups=%d × mamba_unpadded=%d). This makes vLLM's "
            "num_blocks match per-layer TPU allocation when mamba layers "
            "cannot be truly shared.", uniform_page_size_bytes,
            num_attn_groups, attn_page_size_bytes, num_mamba_groups,
            unpadded_mamba_page_size)

        self._hybrid_uniform_page_size_bytes = int(uniform_page_size_bytes)
        self.cache_config.mamba_page_size_padded = int(uniform_page_size_bytes)

        # Pin vLLM's num_blocks via a two-step flooring that keeps peak
        # HBM within `gpu_memory_utilization × total_hbm` at high
        # utilization. See `_maybe_set_num_blocks_override` for the
        # formula and rationale; the short version is that vLLM's
        # single-step `floor(avail / (uniform × group_size))` can land
        # one block higher than the two-step value, and that extra
        # block × group_size × uniform bytes is enough to push past the
        # budget against imprecision in vLLM's `avail` estimate.
        self._maybe_set_num_blocks_override(attn_page_size_bytes,
                                            int(uniform_page_size_bytes),
                                            group_size)

    def _maybe_set_num_blocks_override(self, attn_page_size_bytes: int,
                                       uniform_page_size_bytes: int,
                                       group_size: int) -> None:
        """Pin `cache_config.num_gpu_blocks_override` to the two-step
        flooring value that keeps peak HBM within the user-set
        `gpu_memory_utilization` budget at high utilization.

        Formula:
          `num_blocks_attn = floor(avail / (attn_page × group_size))`
          `num_blocks_tpu  = floor(attn_page × num_blocks_attn / uniform)`

        The two-step flooring can land 1 block lower than vLLM's
        single-step `floor(avail / (uniform × group_size))`. Since
        `uniform > attn_page`, that 1-block gap costs `group_size × uniform`
        bytes of HBM, which — against the imprecision in vLLM's `avail`
        estimate — is enough to tip high-utilization configurations into
        OOM. Pinning to `num_blocks_tpu` preserves the headroom the
        single-step formula silently removes.

        No internal safety margin is applied — `gpu_memory_utilization` is
        the knob users already have for reserving headroom. Adding a
        silent reduction here would conflict with their explicit budget.

        Skipped if the user has explicitly set `num_gpu_blocks_override` or
        if HBM usage isn't readable (e.g. in tests without real devices).
        Spec padding alone still fixes the OOB bug in that case; only the
        ~1-block-per-tensor flooring-boundary precision is lost.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block for one
                attention layer, from `get_attention_page_size_bytes`
                (accounts for dtype packing like fp8).
            uniform_page_size_bytes: bytes per block for one `KVCacheTensor`
                shared across `num_attn_groups + num_mamba_groups` layers
                (the `_hybrid_uniform_page_size_bytes` value set above).
            group_size: number of layers per vLLM kv-cache group, used by
                vLLM to compute `num_blocks` from the attention tensor size.

        Returns:
            None. Side effect: sets `cache_config.num_gpu_blocks_override`
            if all preconditions hold; otherwise leaves it unset.
        """
        cache_config = self.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return

        try:
            free_memory, limit_memory = torch.accelerator.get_memory_info(
                self.device)
            total_used = limit_memory - free_memory
            total_limit = limit_memory
        except Exception as exc:
            logger.debug(
                "Skipping num_gpu_blocks_override: hbm_usage_bytes failed "
                "(%s).", exc)
            return

        gpu_mem_util = cache_config.gpu_memory_utilization
        avail = int(total_limit * gpu_mem_util - total_used)
        if avail <= 0:
            return

        naive_vllm_num_blocks = avail // (attn_page_size_bytes * group_size)
        if naive_vllm_num_blocks <= 0:
            return
        naive_tensor_size = attn_page_size_bytes * naive_vllm_num_blocks
        num_blocks_tpu = naive_tensor_size // uniform_page_size_bytes
        if num_blocks_tpu <= 0:
            return

        cache_config.num_gpu_blocks_override = int(num_blocks_tpu)
        logger.info(
            "Hybrid KV cache: setting num_gpu_blocks_override=%d to align "
            "the scheduler's block pool with per-layer TPU allocation "
            "(avail=%d, naive_vllm_num_blocks=%d).", num_blocks_tpu, avail,
            naive_vllm_num_blocks)

    def _prepare_async_token_substitution_indices(
        self, start_index: int, num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._pre_async_results is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        token_in_tpu_cur_input_indices_list = []
        token_in_tpu_pre_next_tokens_indices_list = []
        acc_cur_len = 0

        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[start_index + i]
            acc_cur_len += num_scheduled_tokens_per_req[i]
            assert req_id is not None
            if req_id not in self._pre_async_results.req_id_to_index_copy:
                continue

            token_in_tpu_cur_input_indices_list.append(acc_cur_len - 1)
            token_in_tpu_pre_next_tokens_indices_list.append(
                self._pre_async_results.req_id_to_index_copy[req_id])

        if len(token_in_tpu_cur_input_indices_list) > 0:
            return (np.array(token_in_tpu_cur_input_indices_list,
                             dtype=np.int32),
                    np.array(token_in_tpu_pre_next_tokens_indices_list,
                             dtype=np.int32))
        else:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    def _apply_async_token_substitution(
            self, input_ids: torch.Tensor,
            token_in_tpu_cur_input_indices: np.ndarray,
            token_in_tpu_pre_next_tokens_indices: np.ndarray) -> torch.Tensor:
        """Apply async token substitution if needed."""
        if len(token_in_tpu_cur_input_indices) == 0:
            return input_ids

        idx_pad_len = len(input_ids) - len(token_in_tpu_cur_input_indices)

        # Pad according to the instructions written inside _substitute_placeholder_token
        full_range = np.arange(0, len(input_ids), dtype=np.int32)
        missing_values = np.setdiff1d(full_range,
                                      token_in_tpu_cur_input_indices)
        padded_token_in_tpu_cur_input_indices = np.concatenate(
            (token_in_tpu_cur_input_indices, missing_values))

        padded_token_in_tpu_pre_next_tokens_indices = np.pad(
            token_in_tpu_pre_next_tokens_indices, (0, idx_pad_len),
            mode='constant',
            constant_values=-1)

        cur_input_indices = torch.from_numpy(
            padded_token_in_tpu_cur_input_indices).to(self.device,
                                                      non_blocking=True)
        pre_next_tokens_indices = torch.from_numpy(
            padded_token_in_tpu_pre_next_tokens_indices).to(self.device,
                                                            non_blocking=True)

        return _substitute_placeholder_token(
            input_ids, cur_input_indices, pre_next_tokens_indices,
            self._pre_async_results.next_tokens_tpu)

    def _modify_prev_results(self):
        if self._pre_async_results is None:
            return

        pre_req_ids = self._pre_async_results.req_ids
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_sampled_tokens_req_indices = self._pre_async_results.discard_sampled_tokens_req_indices

        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()
        assert pre_next_tokens_cpu is not None

        pre_next_tokens_cpu = pre_next_tokens_cpu[:len(pre_req_ids)]
        max_gen_len = pre_next_tokens_cpu.shape[-1]

        if max_gen_len == 1:
            valid_sampled_token_ids = pre_next_tokens_cpu.tolist()
        else:
            valid_mask = pre_next_tokens_cpu != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in pre_next_tokens_cpu[valid_mask].split(gen_lens)
            ]

        for i in pre_discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        for pre_req_idx, req_state, seq_len, req_id in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if not sampled_ids:
                continue

            # Replace the 0 placeholder we appended in the previous step
            req_state.output_token_ids[-1] = sampled_ids[0]
            if len(sampled_ids) > 1:
                req_state.output_token_ids.extend(sampled_ids[1:])

            if req_id not in self.input_batch.req_id_to_index:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]

            if len(sampled_ids) > 1:
                self.input_batch.num_tokens_no_spec[req_idx] += len(
                    sampled_ids) - 1

            target_slice = slice(seq_len - len(sampled_ids) + 1, seq_len + 1)
            self.input_batch.token_ids_cpu[req_idx, target_slice] = sampled_ids

    def _update_placeholder(self, discard_sampled_tokens_req_indices,
                            request_seq_lens):
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_set = set(discard_sampled_tokens_req_indices)
        for req_idx, req_state, seq_len, req_id in request_seq_lens:
            if req_idx in discard_set:
                continue

            end_idx = seq_len + 1
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_state.output_token_ids.append(0)

            placeholder_req_id_to_index[req_state.req_id] = req_idx

        return placeholder_req_id_to_index

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput",
                        start_index: int, num_decode_reqs: int):
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        assert start_index < num_reqs

        # Get the number of scheduled tokens for each request.
        use_max_model_len = self.most_model_len is None
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        end_index = start_index

        # Use either most_model_len or max_model_len depending on request size.
        for i in range(start_index, num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if (not use_max_model_len and self.most_model_len is not None
                    and num_tokens > self.most_model_len):
                use_max_model_len = True
            num_scheduled_tokens_per_req.append(num_tokens)
        if use_max_model_len:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_max_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[:
                                                                            self
                                                                            .
                                                                            num_reqs_max_model_len]
                end_index = start_index + self.num_reqs_max_model_len
            else:
                end_index = num_reqs
        else:
            assert self.num_reqs_most_model_len is not None
            if len(num_scheduled_tokens_per_req
                   ) > self.num_reqs_most_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[:
                                                                            self
                                                                            .
                                                                            num_reqs_most_model_len]
                end_index = start_index + self.num_reqs_most_model_len
            else:
                end_index = num_reqs
        max_num_scheduled_tokens_all_reqs = max(num_scheduled_tokens_per_req)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req,
                                                dtype=np.int32)
        total_num_scheduled_tokens = sum(num_scheduled_tokens_per_req)
        assert max_num_scheduled_tokens_all_reqs > 0

        num_reqs = len(num_scheduled_tokens_per_req)

        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Fast path for decode-only: all requests have exactly 1 token.
        if max_num_scheduled_tokens_all_reqs == 1:
            # Pure decode: each request schedules exactly 1 token.
            # req_indices = [0, 1, ..., num_reqs-1]
            req_indices = self.arange_np[start_index:start_index + num_reqs]
            # positions = num_computed_tokens for each request
            positions_np = self.positions_np[:num_reqs]
            np.copyto(
                positions_np, self.input_batch.
                num_computed_tokens_cpu[start_index:start_index + num_reqs])
            # token_indices = positions + req_index * max_model_len
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:num_reqs],
            )
        else:
            # General path: mixed prefill + decode.
            # Get request indices.
            req_indices = np.repeat(
                self.arange_np[start_index:start_index + num_reqs],
                num_scheduled_tokens_per_req)

            # Get batched arange.
            # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            arange = np.concatenate(
                [self.arange_np[:n] for n in num_scheduled_tokens_per_req])

            # Get positions.
            positions_np = self.positions_np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )

            # Get token indices.
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])

            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:total_num_scheduled_tokens],
            )

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        # Keep padded entries equal to the last valid location so padded
        # requests have zero length instead of a negative q_len.
        self.query_start_loc_np[num_reqs +
                                1:] = self.query_start_loc_np[num_reqs]

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[start_index:start_index +
                                                     num_reqs] +
            num_scheduled_tokens_per_req)

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens] = 0
        self.input_ids = self.input_ids_cpu[:
                                            padded_total_num_scheduled_tokens].to(
                                                self.device, non_blocking=True)
        if self.uses_mrope:
            self.mrope_positions.cpu[:, total_num_scheduled_tokens:
                                     padded_total_num_scheduled_tokens] = 0
            self.position_ids = self.mrope_positions.cpu[:, :
                                                         padded_total_num_scheduled_tokens].to(
                                                             self.device,
                                                             non_blocking=True)
        else:
            self.position_ids = self.positions_cpu[:
                                                   padded_total_num_scheduled_tokens].to(
                                                       self.device,
                                                       non_blocking=True)
        if use_max_model_len:
            seq_lens = self.seq_lens_cpu[:self.num_reqs_max_model_len].to(
                self.device, non_blocking=True)
            target_num_reqs = self.num_reqs_max_model_len
        else:
            assert self.num_reqs_most_model_len is not None
            seq_lens = self.seq_lens_cpu[:self.num_reqs_most_model_len].to(
                self.device, non_blocking=True)
            target_num_reqs = self.num_reqs_most_model_len

        # For decode-only case, cache constant device tensors to skip H2D.
        # query_start_loc, logits_indices, and request_distribution don't
        # change between decode steps for a given (num_reqs, padded_num_reqs).
        is_decode_only = (max_num_scheduled_tokens_all_reqs == 1)
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)
        decode_cache_key = (num_reqs, padded_num_reqs, use_max_model_len)

        if is_decode_only and decode_cache_key == self._decode_device_cache_key:
            # Reuse cached device tensors — skip 3 H2D copies.
            query_start_loc = self._cached_query_start_loc
            logits_indices = self._cached_logits_indices
            request_distribution = self._cached_request_distribution
        else:
            # Compute and copy to device (first call or shape changed).
            if use_max_model_len:
                query_start_loc = self.query_start_loc_cpu[:self.
                                                           num_reqs_max_model_len
                                                           + 1].to(
                                                               self.device,
                                                               non_blocking=True
                                                           )
            else:
                query_start_loc = self.query_start_loc_cpu[:self.
                                                           num_reqs_most_model_len
                                                           + 1].to(
                                                               self.device,
                                                               non_blocking=True
                                                           )

            # Indices at which we sample (positions of last token in the
            # sequence). Padded to avoid recompiling when `num_reqs` varies.
            logits_indices = (self.query_start_loc_cpu[1:padded_num_reqs + 1] -
                              1).to(self.device, non_blocking=True)

            # For the V3 kernel, request_distribution is
            # [decode_end, prefill_end, mixed_end]. We put decode requests first,
            # no dedicated prefill-only bucket, and all remaining requests in mixed mode.
            chunk_num_decode = max(
                0, min(num_decode_reqs - start_index, num_reqs))
            self._request_distribution_cpu[0] = chunk_num_decode
            self._request_distribution_cpu[1] = chunk_num_decode
            self._request_distribution_cpu[2] = num_reqs
            request_distribution = self._request_distribution_cpu.to(
                self.device, non_blocking=True)

            if is_decode_only:
                # Cache for future decode steps.
                self._decode_device_cache_key = decode_cache_key
                self._cached_query_start_loc = query_start_loc
                self._cached_logits_indices = logits_indices
                self._cached_request_distribution = request_distribution

        self._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=start_index,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )
        slot_mappings = self.empty_slot_mappings
        per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = self._build_attention_metadata(
            num_tokens=total_num_scheduled_tokens,
            num_reqs=target_num_reqs,
            max_query_len=max_num_scheduled_tokens_all_reqs,
            num_tokens_padded=padded_total_num_scheduled_tokens,
            num_reqs_padded=target_num_reqs,
            slot_mappings=slot_mappings,
        )

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens)

            self.set_active_loras(self.input_batch,
                                  padded_num_scheduled_tokens_per_req)

        # Prepare token substitution indices
        cur_input_indices, pre_next_tokens_indices = self._prepare_async_token_substitution_indices(
            start_index, num_reqs, num_scheduled_tokens_per_req)

        return (
            per_layer_attn_metadata,
            logits_indices,
            padded_num_reqs,
            num_reqs,
            end_index,
            cur_input_indices,
            pre_next_tokens_indices,
        )

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ):
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            return None, inputs_embeds
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            return input_ids, None

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
        # Update cached state
        self._update_states(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)

        mm_embed_inputs = self.mm_embed_inputs
        self.mm_embed_inputs = None

        num_decode_reqs = self._reorder_batch_for_rpa(scheduler_output)

        start_index = 0
        logits_list = []
        num_reqs_list = []

        # NOTE: setup current batch's metadata for kv connector.
        # Currently, only verified with NixlConnector
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)

        while start_index < self.input_batch.num_reqs:
            (attn_metadata, logits_indices, padded_num_reqs, num_reqs,
             end_index, cur_input_indices,
             pre_next_tokens_indices) = (self._prepare_inputs(
                 scheduler_output, start_index, num_decode_reqs))

            input_ids = self._apply_async_token_substitution(
                self.input_ids, cur_input_indices, pre_next_tokens_indices)

            input_ids, inputs_embeds = self._get_model_inputs(
                input_ids, mm_embed_inputs)
            # Run the decoder
            # set_forward_context: vLLM's native context for attention metadata
            # set_vllm_model_wrapper_context: TPU-specific context for mesh info
            with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
            ), set_vllm_model_wrapper_context(mesh=self.mesh):
                hidden_states = self.forward_model(
                    input_ids=input_ids,
                    positions=self.position_ids,
                    inputs_embeds=inputs_embeds,
                )

            logits = self.compute_selected_logits(hidden_states,
                                                  logits_indices)

            logits_list.append(logits)
            num_reqs_list.append(num_reqs)

            start_index = end_index

        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits_list=logits_list,
            num_reqs_list=num_reqs_list,
        )
        return None

    @torch.no_grad()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            return None  # type: ignore[return-value]

        state = self.execute_model_state
        scheduler_output = state.scheduler_output
        self.execute_model_state = None

        # Prepare inputs, the requests might be split into multiple
        # executions, combine the result of each execution.

        max_num_logprobs = self.input_batch.max_num_logprobs
        if max_num_logprobs == -1:
            raise NotImplementedError(
                "TPU runner does not support full logprobs (`logprobs=-1`) "
                "with the merged vLLM v1 sampler path yet.")
        needs_logprobs = max_num_logprobs is not None

        combined_selected_tokens: list[torch.Tensor] = []
        combined_logprobs: list[Any] = []

        cur_start_idx = 0
        req_ids = cast(list[str],
                       self.input_batch.req_ids[:self.input_batch.num_reqs])
        all_greedy = self.input_batch.all_greedy
        for logits, num_reqs in zip(state.logits_list, state.num_reqs_list):
            cur_end_idx = cur_start_idx + num_reqs
            if grammar_output is not None:
                require_struct_decoding, grammar_bitmask_padded, arange = (
                    self.prepare_structured_decoding_input(
                        logits, grammar_output))
                logits = self.structured_decode(require_struct_decoding,
                                                grammar_bitmask_padded, logits,
                                                arange)
            if all_greedy:
                dummy_placeholder = torch.empty((1, 1),
                                                dtype=logits.dtype,
                                                device=logits.device)
                selected_token_ids = self.sample_from_logits_func(
                    logits,
                    dummy_placeholder,
                    dummy_placeholder,
                    all_greedy=True)
            else:
                temperatures_tpu = self._build_padded_temperatures(
                    cur_start_idx, cur_end_idx, logits)
                u = torch.rand_like(logits)
                selected_token_ids = self.sample_from_logits_func(
                    logits, temperatures_tpu, u, all_greedy=all_greedy)
            # NOTE (NickLucche) Use the original logits (before any penalties or
            # temperature scaling) for the top-k logprobs. We can't enforce it
            # due to recompilations outside torch.compiled code, so just make
            # sure `sample_from_logits` does not modify the logits in-place.
            logprobs = (self.gather_logprobs(logits, selected_token_ids)
                        if needs_logprobs else None)

            combined_selected_tokens.append(selected_token_ids[:num_reqs])
            if needs_logprobs:
                sliced_logprobs = LogprobsTensors(
                    logprobs.logprob_token_ids[:num_reqs],
                    logprobs.logprobs[:num_reqs],
                    logprobs.selected_token_ranks[:num_reqs],
                    logprobs.cu_num_generated_tokens)
                combined_logprobs.append(sliced_logprobs)

            self._update_num_xla_graphs("decoding_step")
            cur_start_idx = cur_end_idx

        # NOTE: current kv load and save get h2d/d2h copies involved.
        # Those copies are blocking. Once they become async., kv_save
        # should be called right after each single forward pass,
        # instead of the forwards of the entire input batch.
        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfers(
            scheduler_output)

        logprobs = []
        if needs_logprobs and len(combined_logprobs):
            # TODO: concatenate the LogprobsTensors (torch) first, then
            # call .tolists() once to match the GPU path and avoid the
            # per-chunk numpy round-trip.
            combined_logprobs_lists = []
            for i in range(len(combined_logprobs)):
                logprobs = combined_logprobs[i].tolists()
                combined_logprobs_lists.append(logprobs)

            logprobs_lists = LogprobsLists(
                logprob_token_ids=np.concatenate(
                    [lp.logprob_token_ids for lp in combined_logprobs_lists]),
                logprobs=np.concatenate(
                    [lp.logprobs for lp in combined_logprobs_lists]),
                sampled_token_ranks=np.concatenate([
                    lp.sampled_token_ranks for lp in combined_logprobs_lists
                ]),
            )
            logprobs = logprobs_lists

        discard_sampled_tokens_req_indices = []
        num_reqs = self.input_batch.num_reqs
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        request_seq_lens = []
        for i, req_id in enumerate(req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            # Ignore the sampled token from the partial request.
            # Rewind the generator state as if the token was not sampled.
            if seq_len < req_state.num_tokens:
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)
            else:
                request_seq_lens.append((i, req_state, seq_len, req_id))

        kv_connector_output = (None if (finished_sending is None
                                        and finished_recving is None) else
                               KVConnectorOutput(
                                   finished_sending=finished_sending,
                                   finished_recving=finished_recving,
                               ))

        next_tokens = None
        if len(combined_selected_tokens) > 1:
            next_tokens = torch.cat(combined_selected_tokens, dim=0)
        elif len(combined_selected_tokens) == 1:
            next_tokens = combined_selected_tokens[0]

        next_tokens_tpu = None
        copy_state = None
        if next_tokens is not None:
            copy_state = AsyncTPUCopyState.from_device(next_tokens)

        if self.scheduler_config.async_scheduling:
            self._modify_prev_results()
            req_id_to_index_copy = self._update_placeholder(
                discard_sampled_tokens_req_indices, request_seq_lens)
            if next_tokens is not None:
                next_tokens_tpu = next_tokens.view(-1)
                self._pre_async_results = AsyncPreResults(
                    req_ids=req_ids,
                    next_tokens_tpu=next_tokens_tpu,
                    request_seq_lens=request_seq_lens,
                    discard_sampled_tokens_req_indices=
                    discard_sampled_tokens_req_indices,
                    req_id_to_index_copy=req_id_to_index_copy,
                    copy_state=copy_state,
                )
            else:
                self._pre_async_results = None

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            # Snapshot of the req_id_to_index for the VLLM scheduler.
            req_id_to_index=dict(self.input_batch.req_id_to_index),
            sampled_token_ids=
            [],  # Filled in AsyncTPUModelRunnerOutput get_output
            logprobs=logprobs,
            prompt_logprobs_dict={req_id: None
                                  for req_id in req_ids},
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

        async_output = AsyncTPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            copy_state=copy_state,
            discard_sampled_tokens_req_indices=
            discard_sampled_tokens_req_indices)

        if not self.scheduler_config.async_scheduling:
            final_output = async_output.get_output()
            for i, req_state, seq_len, req_id in request_seq_lens:
                if i in discard_sampled_tokens_req_indices:
                    continue
                valid_tokens = final_output.sampled_token_ids[i]
                if not valid_tokens:
                    continue
                req_idx = self.input_batch.req_id_to_index[req_id]
                self.input_batch.num_tokens_no_spec[req_idx] += len(
                    valid_tokens)
                target_slice = slice(seq_len - len(valid_tokens) + 1,
                                     seq_len + 1)
                self.input_batch.token_ids_cpu[req_idx,
                                               target_slice] = valid_tokens
                req_state.output_token_ids.extend(valid_tokens)
            return final_output

        return async_output

    def load_model(self) -> None:
        self.device_config.device = torch.device("tpu")
        self.device = self.device_config.device
        self.vllm_config.device_config = self.device_config

        # Set TPU-specific quantization config before model loading.
        # This ensures that TPU-compatible quantization methods are used
        # instead of the default GPU-based ones (e.g., MXFP4 for MoE models).
        # For unquantized models (quantization=None), we still need to apply
        # our TPU config to override vLLM's default UnquantizedFusedMoEMethod
        # which uses torch_xla.
        logger.info("Setting TPU quantization config for: %s",
                    self.model_config.quantization)
        self.vllm_config.quant_config = get_tpu_quantization_config(
            self.vllm_config)

        model_loader = get_model_loader(self.load_config)
        logger.info("Loading model from scratch...")
        with set_vllm_model_wrapper_context(mesh=self.mesh), \
             set_current_vllm_config(self.vllm_config):
            model = model_loader.load_model(vllm_config=self.vllm_config,
                                            model_config=self.model_config)
        self.model = model
        # Ensure attention custom ops exist before any compile/inference path,
        self._initialize_attention_kernels()

    def _initialize_attention_kernels(self) -> None:
        """Pre-build Pallas RPA attention kernels before torch.compile.

        Must be called after model weights are loaded and before the first
        forward_model() call (which triggers torch.compile tracing).
        """
        if self._attention_kernels_initialized:
            return
        from tpu_inference.layers.vllm.attention import \
            PallasAttentionBackendImpl

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        with set_vllm_model_wrapper_context(mesh=self.mesh):
            for layer_name, attn_layer in layers.items():
                if isinstance(attn_layer.impl, PallasAttentionBackendImpl):
                    attn_layer.impl.initialize_kernel(attn_layer)
                    logger.info("Pre-built RPA kernel for layer: %s",
                                layer_name)
        self._attention_kernels_initialized = True

    @torch.no_grad()
    def _dummy_run(self,
                   num_tokens: int,
                   num_reqs: int,
                   num_blocks: int,
                   use_max_model_len: bool = True) -> None:
        if self.supports_mm_inputs:
            input_ids = None
            inputs_embeds = torch.zeros(
                (num_tokens, self.inputs_embeds_size),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            input_ids = torch.zeros((num_tokens),
                                    dtype=torch.int32).to(self.device)
            inputs_embeds = None
        actual_num_reqs = min(num_tokens, num_reqs)
        if self.uses_mrope:
            position_ids = torch.zeros((3, num_tokens),
                                       dtype=torch.int32).to(self.device)
        else:
            position_ids = torch.zeros(num_tokens,
                                       dtype=torch.int32).to(self.device)
        query_lens = [1] * num_reqs
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(self.device)
        seq_lens = torch.ones((num_reqs, ), dtype=torch.int32).to(self.device)
        # V3: request_distribution = [decode_end, prefill_end, mixed_end].
        # Dummy runs use one scheduled token per active request, so model
        # them as pure decode to match the real single-chip path.
        request_distribution = torch.tensor(
            [actual_num_reqs, actual_num_reqs, actual_num_reqs],
            dtype=torch.int32).to(self.device)

        if getattr(self, "kv_cache_config", None) is not None:
            self._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
                num_reqs=num_reqs,
                start_index=0,
                use_max_model_len=use_max_model_len,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                position_ids_override=position_ids,
            )
            slot_mappings = self.empty_slot_mappings
            per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = self._build_attention_metadata(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_query_len=1,
                num_tokens_padded=num_tokens,
                num_reqs_padded=num_reqs,
                slot_mappings=slot_mappings,
            )
        else:
            # Pre-init path (called before initialize_kv_cache): use the
            # caller-supplied num_blocks and a single shared metadata.
            if self._attn_layer_names is None:
                self._attn_layer_names = list(
                    get_layers_from_vllm_config(self.vllm_config,
                                                (Attention, MambaBase)).keys())
            block_tables = torch.zeros((num_reqs * num_blocks, ),
                                       dtype=torch.int32).to(self.device)
            attn_metadata = AttentionMetadata(
                input_positions=position_ids,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )

            per_layer_attn_metadata = {
                layer_name: attn_metadata
                for layer_name in self._attn_layer_names
            }

        with (
                self.maybe_select_dummy_loras(
                    self.lora_config, np.array([num_tokens], dtype=np.int32)),
                set_forward_context(per_layer_attn_metadata, self.vllm_config,
                                    0),
                set_vllm_model_wrapper_context(mesh=self.mesh),
        ):
            out = self.forward_model(input_ids=input_ids,
                                     positions=position_ids,
                                     inputs_embeds=inputs_embeds)
            sync.synchronize(out, wait=True)
        self._hidden_states_dtype = out.dtype

    @contextlib.contextmanager
    def _precompile_timed(self, name: str):
        """Common boilerplate for each AOT bucket precompile pass."""
        logger.info("Compiling %s with different input shapes.", name)
        start = time.perf_counter()
        yield
        logger.info("Compilation finished in %.2f [secs].",
                    time.perf_counter() - start)
        self._update_num_xla_graphs(name)

    def _dummy_logits(self, num_reqs: int) -> torch.Tensor:
        return torch.zeros((num_reqs, self.vocab_size),
                           device=self.device,
                           dtype=self._hidden_states_dtype)

    def _precompile_compute_selected_logits(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_selected_logits"):
            for num_tokens in self.num_tokens_paddings:
                dummy_hidden = torch.zeros((num_tokens, hsize),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                for num_reqs in self.num_reqs_paddings:
                    indices = torch.zeros(num_reqs,
                                          dtype=torch.int32,
                                          device=self.device)
                    out = self.compute_selected_logits(dummy_hidden, indices)
                    sync.synchronize(out, wait=True)
                    logger.info("  -- num_tokens: %d, num_seqs: %d",
                                num_tokens, num_reqs)
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_structured_decoding(self) -> None:
        with self._precompile_timed("structured_decoding"):
            arange = self.structured_decode_arange.to(self.device)
            for num_reqs in self.num_reqs_paddings:
                out = self.structured_decode(
                    self.require_structured_out_cpu[:num_reqs].to(self.device),
                    self.grammar_bitmask_cpu[:num_reqs].to(self.device),
                    self._dummy_logits(num_reqs),
                    arange,
                )
                sync.synchronize(out, wait=True)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_sample_from_logits(self) -> None:
        with self._precompile_timed("sample_from_logits"):
            for num_reqs in self.num_reqs_paddings:
                dummy_logits = self._dummy_logits(num_reqs)
                dummy_temperatures = torch.ones(
                    (num_reqs, 1),
                    dtype=self._hidden_states_dtype,
                    device=self.device)
                dummy_u = torch.rand_like(dummy_logits)
                for all_greedy in [False, True]:
                    out = self.sample_from_logits_func(dummy_logits,
                                                       dummy_temperatures,
                                                       dummy_u,
                                                       all_greedy=all_greedy)
                    sync.synchronize(out, wait=True)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_gather_logprobs(self) -> None:
        with self._precompile_timed("gather_logprobs"):
            for num_reqs in self.num_reqs_paddings:
                out = self.gather_logprobs(
                    self._dummy_logits(num_reqs),
                    torch.zeros((num_reqs, 1),
                                dtype=torch.int64).to(self.device),
                )
                sync.synchronize(out.logprobs, wait=True)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_sampling_subgraphs(self) -> None:
        """Compile sampling-path subgraphs so their bottom-HBM reservations
        are visible to vLLM's available-memory probe in profile_run."""
        self._precompile_compute_selected_logits()
        self._precompile_structured_decoding()
        self._precompile_sample_from_logits()
        self._precompile_gather_logprobs()

    def capture_model(self) -> None:
        """Precompile every torch.compile subgraph across all input buckets."""
        if self.enforce_eager:
            return
        with self.maybe_setup_dummy_loras(self.lora_config):
            with self._precompile_timed("model backbone"):
                for num_tokens in self.num_tokens_paddings:
                    logger.info("  -- num_tokens: %d", num_tokens)
                    self._dummy_run(num_tokens,
                                    self.num_reqs_max_model_len,
                                    self.max_num_blocks_per_req,
                                    use_max_model_len=True)
                    if self.most_model_len is not None:
                        self._dummy_run(num_tokens,
                                        self.num_reqs_most_model_len,
                                        self.num_blocks_per_most_len_req,
                                        use_max_model_len=False)

            self._precompile_sampling_subgraphs()

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        self._initialize_attention_kernels()

        # KV cache isn't allocated yet; torch.compile would specialize on
        # the kv_cache.numel() == 0 early-return in attention. Compile the
        # backbone later in capture_model() after KV alloc.
        with _bypass_torch_compile(self.model):
            self._dummy_run(num_tokens,
                            self.num_reqs_max_model_len,
                            self.max_num_blocks_per_req,
                            use_max_model_len=True)

        # Sampling-path subgraphs don't depend on kv_cache; compile them
        # now so their bottom-HBM reservations are visible to vLLM's
        # available-memory probe and counted in the KV-cache budget.
        if not self.enforce_eager:
            with self.maybe_setup_dummy_loras(self.lora_config):
                self._precompile_sampling_subgraphs()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        # Mirror GPUModelRunner.initialize_kv_cache: needed by inherited
        # _update_states -> _may_reorder_batch which reads kv_cache_config.
        self.kv_cache_config = kv_cache_config

        # Dummy slot mapping, not used anywhere in the TPU code flow. But needed
        #  for upstream `_build_attention_metadata` call.
        self.empty_slot_mappings = {
            gid: torch.empty(0, device=self.device)
            for gid in range(len(self.kv_cache_config.kv_cache_groups))
        }

        attn_block_size = None
        has_attention = False
        has_mamba = False
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                has_mamba = True
                # We can safely ignore block size for Mamba layers since they only use a single cache state per sequence.
                continue
            if isinstance(spec, AttentionSpec):
                has_attention = True
            elif len(kv_cache_config.kv_cache_groups) > 1:
                raise NotImplementedError(
                    "Only AttentionSpec and MambaSpec are supported in KV cache groups > 1."
                )

            block_size = getattr(spec, "block_size", None)
            if block_size is not None:
                if attn_block_size is None:
                    attn_block_size = block_size
                assert attn_block_size == block_size, "Block size across attention groups must be the same."

        block_sizes = []
        for group in kv_cache_config.kv_cache_groups:
            block_sizes.append(
                getattr(group.kv_cache_spec, "block_size", attn_block_size)
                or self.block_size)

        self.may_reinitialize_input_batch(kv_cache_config, block_sizes)

        # Populate self.attn_groups directly with a single shared TPU builder
        # per group; bypasses parent's initialize_attn_backend (per-backend
        # get_builder_cls dispatch, cudagraph-mode resolution, cp compat
        # checks) — all GPU-relevant and unused on TPU.
        self.attn_groups = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            builder = AttentionMetadataBuilder(group.kv_cache_spec,
                                               group.layer_names,
                                               self.vllm_config,
                                               self.device,
                                               runner=self,
                                               kv_cache_group_id=gid)
            self.attn_groups.append([
                AttentionGroup(
                    backend=None,
                    layer_names=list(group.layer_names),
                    kv_cache_spec=group.kv_cache_spec,
                    kv_cache_group_id=gid,
                    metadata_builders=[builder],
                )
            ])

        # Verify dtype compatibility between block_table_cpu and every
        # per-group block table that downstream code may index.
        for group_id in range(len(kv_cache_config.kv_cache_groups)):
            assert (self.block_table_cpu.dtype == self.input_batch.
                    block_table[group_id].get_cpu_tensor().dtype)

        layer_name_to_spec = {}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                layer_name_to_spec[layer_name] = group.kv_cache_spec

        kv_caches: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # If the KV cache tensor is shared by multiple layers, then we
            # duplicate cache for each layer and `num_blocks` is calculated
            # based on the total size of the shared cache.
            # Otherwise, `num_blocks` is calculated based on the size of the
            # single layer's KV cache spec.
            tensor_size = kv_cache_tensor.size
            shared_by = kv_cache_tensor.shared_by
            if len(shared_by) > 1:
                assert has_attention and has_mamba, "KV cache duplication is only supported for hybrid models with Mamba and Full Attention layers."
                total_group_page_size = 0
                for name in shared_by:
                    spec = layer_name_to_spec[name]
                    # Use the per-layer *TPU-actual* per-block bytes so the
                    # sum equals the `page_size_padded` that
                    # `update_mamba_page_size_padded` installed on every
                    # spec (== attn_page + N × mamba_unpadded). For
                    # attention, the TPU-actual size includes dtype-
                    # specific packing (e.g., fp8 KV packs 4 elements per
                    # 32-bit word) which `spec.real_page_size_bytes`
                    # doesn't account for — on fp8 models they differ by
                    # 2×, which would break the num_blocks match here.
                    if isinstance(spec, MambaSpec):
                        total_group_page_size += dataclasses.replace(
                            spec, page_size_padded=None).page_size_bytes
                    elif isinstance(spec, AttentionSpec):
                        total_group_page_size += PallasAttentionBackend.get_kv_cache_page_size_bytes(
                            spec.block_size,
                            spec.num_kv_heads,
                            spec.head_size,
                            spec.dtype,
                        )
                    else:
                        raise NotImplementedError
                num_blocks = tensor_size // total_group_page_size
            else:
                page_size_bytes = layer_name_to_spec[
                    shared_by[0]].page_size_bytes
                assert tensor_size % page_size_bytes == 0
                num_blocks = tensor_size // page_size_bytes

            for layer_name in shared_by:
                kv_cache_spec = layer_name_to_spec[layer_name]

                if isinstance(kv_cache_spec, MambaSpec):
                    mamba_states = []
                    for _, (shape, dtype) in enumerate(
                            zip(kv_cache_spec.shapes, kv_cache_spec.dtypes)):
                        cache_shape = (num_blocks, *shape)
                        mamba_states.append(
                            torch.zeros(cache_shape,
                                        dtype=dtype).to(self.device))
                    kv_caches[layer_name] = tuple(mamba_states)
                elif isinstance(kv_cache_spec, AttentionSpec):
                    if self.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        assert self.original_parallel_config is not None
                        tp_size = self.original_parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode")
                    kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        kv_cache_spec.dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    tpu_kv_cache = torch.zeros(kv_cache_shape,
                                               dtype=dtype).to(self.device)

                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        # Cross-layer KV cache sharing: shared_kv_cache_layers is populated
        # by get_kv_cache_spec but not yet wired through here. This is a TODO
        # to call GPU's maybe_add_kv_sharing_layers_to_kv_cache_groups once
        # we verify the signature + invariants line up on TPU.

        # Mark KV cache buffers as donation candidates outside torch.compile
        # regions to avoid Dynamo tracing through pybind calls.
        # TODO(geyuhao): Comment out for now as TorchTPU does not support this right now
        # for kv_cache in kv_caches.values():
        #     pallas.set_buffer_donor_(kv_cache, True)

        # Reset kv_caches list (bind_kv_cache expects empty list)
        self.kv_caches = []

        # Use bind_kv_cache to bind KV caches to attention layers using layer names
        # This is the native vLLM pattern and avoids the 'layer_id' attribute error
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        if self.use_spmd:
            # Shard KV Cache
            for cache in self.kv_caches:
                continue
                # xs.mark_sharding(cache, self.mesh, (None, "x", None, None))

        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.register_kv_caches(kv_caches)
            # TPUConnector reads runner.kv_caches lazily and doesn't need
            # set_host_xfer_buffer_ops; only call it on connectors that
            # expose it.
            if hasattr(kv_connector, "set_host_xfer_buffer_ops"):
                kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
            if hasattr(kv_connector, "register_runner"):
                kv_connector.register_runner(self)

    def forward_model(self, input_ids, positions, inputs_embeds=None):
        # @support_torch_compile annotations will be put on the vLLM model if it
        # supports torch.compile
        out = self.model(input_ids=input_ids,
                         positions=positions,
                         inputs_embeds=inputs_embeds)
        # @support_torch_compile may return a list/tuple; extract tensor
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def _build_padded_temperatures(self, cur_start_idx: int, cur_end_idx: int,
                                   logits: torch.Tensor) -> torch.Tensor:
        # Stage [padded_num_reqs] on CPU (neutral 1.0 for padding slots),
        # then one fixed-shape H2D copy to keep decode shape-stable.
        padded_num_reqs = logits.shape[0]
        temps_cpu = torch.ones(padded_num_reqs, dtype=logits.dtype)
        temps_cpu[:cur_end_idx - cur_start_idx].copy_(
            self.input_batch.temperature_cpu_tensor[cur_start_idx:cur_end_idx])
        return temps_cpu.unsqueeze(1).to(logits.device, non_blocking=True)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def compute_selected_logits(
            self, hidden_states: torch.Tensor,
            indices_do_sample: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states[indices_do_sample])

    def _apply_temperature(self, logits: torch.Tensor,
                           temperatures: torch.Tensor) -> torch.Tensor:
        safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
        return logits / safe_temperatures

    # TODO: Under SPMD mode, sample_from_logits has correctness issue.
    #       Re-enable the torch.compile once the issue is fixed in torchxla.
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def sample_from_logits(self,
                           logits: torch.Tensor,
                           temperatures: torch.Tensor,
                           u: torch.Tensor,
                           all_greedy: bool = False) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced
        separately from `forward` for lighter compilation overhead.
        """
        if all_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        is_greedy = temperatures <= SAMPLING_EPS
        scaled_logits = self._apply_temperature(logits, temperatures)
        u_clamped = torch.clamp(u,
                                min=torch.finfo(u.dtype).tiny,
                                max=1.0 - torch.finfo(u.dtype).eps)
        gumbel_noise = -torch.log(-torch.log(u_clamped))
        noisy_logits = scaled_logits + gumbel_noise
        final_logits = torch.where(is_greedy, logits, noisy_logits)
        return torch.argmax(final_logits, dim=-1, keepdim=True)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def gather_logprobs(self, logits: torch.Tensor,
                        sampled_tokens: torch.Tensor) -> LogprobsTensors:
        """
        Gather the top_logprobs with corresponding tokens. Use a fixed number
        of logprobs as an alternative to having multiple pre-compiled graphs.
        Select the number of logprobs actually demanded by each request on CPU.
        """
        token_ids = sampled_tokens.to(torch.int64)
        token_logits = logits.gather(-1, token_ids)
        token_ranks = (logits >= token_logits).sum(dim=-1, dtype=torch.int32)
        log_normalizers = torch.logsumexp(logits, dim=-1, keepdim=True)
        token_logprobs = (token_logits - log_normalizers).to(torch.float32)

        max_logprobs = self.model_config.max_logprobs
        if max_logprobs > 0:
            topk_logits, topk_indices = torch.topk(logits,
                                                   max_logprobs,
                                                   dim=-1)
            topk_logprobs = (topk_logits - log_normalizers).to(torch.float32)
            logprob_token_ids = torch.cat((token_ids, topk_indices), dim=1)
            logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        else:
            logprob_token_ids = token_ids
            logprobs = token_logprobs

        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids.to(torch.int32),
            logprobs=logprobs,
            selected_token_ranks=token_ranks,
        )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            require_struct_decoding,
            self.apply_grammar_bitmask(logits, grammar_bitmask, arange),
            logits,
        )

    def apply_grammar_bitmask(self, logits: torch.Tensor,
                              grammar_bitmask: torch.Tensor,
                              arange: torch.Tensor):
        assert logits.shape[0] == grammar_bitmask.shape[0]
        logits_cloned = logits.clone()
        for i in range(logits.shape[0]):
            unpacked_bitmask = (torch.bitwise_right_shift(
                grammar_bitmask[i][:, None], arange[None, :])
                                & 1) == 0
            unpacked_bitmask = unpacked_bitmask.reshape(-1)[:self.vocab_size]
            logits_cloned[i] = logits_cloned[i].masked_fill(
                unpacked_bitmask, -float("inf"))
        return logits_cloned

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, grammar_output: "GrammarOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grammar_bitmask = grammar_output.grammar_bitmask
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        cumulative_mask_idx = 0
        for req_id in grammar_output.structured_output_request_ids:
            if req_id not in self.input_batch.req_id_to_index:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            self.grammar_bitmask_cpu[batch_index] = torch.from_numpy(
                grammar_bitmask[cumulative_mask_idx])
            # It's not guaranteed that all requests in this batch require
            # structured output, so create a bool tensor to represent
            # the requests that need structured output.
            self.require_structured_out_cpu[batch_index] = True
            cumulative_mask_idx += 1

        return (
            self.require_structured_out_cpu[:num_reqs].to(logits.device),
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device),
            self.structured_decode_arange.to(logits.device),
        )


def _get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    logger.info("Preparing request paddings:")
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = _get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def _get_padded_num_kv_cache_update_slices(num_tokens: int, max_num_reqs: int,
                                           page_size: int) -> int:
    """Calculates the padded number of KV cache update slices to avoid
    recompilation."""
    # NOTE(chengjiyao): let's say R_i is the token num for i-th request,
    # so it occupies most 2 + R_i // page_size pages. The total maximum
    # possible number of pages needed is sum(2 + R_i // page_size), which
    # is <= 2 * max_num_reqs + sum(R_i) // page_size
    # = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    return padded_num_slices


def _prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def _get_num_slices_per_kv_cache_update_block(page_size_bytes: int) -> int:
    """Find the optimum number of slices to copy per Pallas program instance.

    Increasing the number of slices copied in one instance of the kernel program
    will increase HBM bandwidth utilization via more in-flight DMAs.

    However, it will also use more VMEM, and experimentally, we observed
    performance regression at 128 slices on v6e, likely due to running
    out of scalar registers. Thus this function will limit the number of
    slices to 64.
    """
    # The default vmem_limit_bytes of a pallas kernel is 32MB. Here we
    # calculate num_slices_per_block based on 16MB in case any register spills.
    vmem_limit = 16 * 1024 * 1024
    num_slices_per_block = vmem_limit // page_size_bytes
    assert num_slices_per_block > 0, "Number of slices should be positive"
    num_slices_per_block = _prev_power_of_2(num_slices_per_block)
    if num_slices_per_block > 64:
        num_slices_per_block = 64
    return num_slices_per_block
