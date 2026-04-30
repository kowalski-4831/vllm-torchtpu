# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import contextlib
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, cast

# TODO: Remove this after jax dependency is removed
import jax
import numpy as np
import torch
import torch.nn as nn
import vllm.envs as envs
# TODO: Remove this after jax dependency is removed
from jax.sharding import Mesh
from packaging import version
from torch_tpu import api
from torch_tpu._internal import sync
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import (ParallelConfig, VllmConfig,
                         get_layers_from_vllm_config, set_current_vllm_config,
                         update_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import (Attention,
                                                  ChunkedLocalAttention,
                                                  MLAAttention)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.interfaces import (SupportsMultiModal,
                                                   supports_transcription)
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model, is_text_generation_model)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        MLAAttentionSpec, SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsLists,
                             LogprobsTensors, ModelRunnerOutput)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin, KVConnectorOutput)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.tpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import bind_kv_cache

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
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
        next_tokens: torch.Tensor, placeholder_num: int):
    """Substitute placeholder tokens from TPU for async scheduler

    Padding for parallelisation of the substitute_placeholder_token_fn
    [1, 3] => [1, 3, 0, 2, 4, 5, 6, 7, 8]
    The reason for such a special padding instead of padding with -1 is:
    An edge case when the end index needs to be updated and padding is required.
    If we pad the array with -1, the _substitute_placeholder_token_fn will repeatedly update the end element with the original value
    Although such a scenario is unlikely to happen in vLLM, it is best to eliminate any potential risks.

    Args:
        input_ids: possible input_ids size
        token_in_tpu_cur_input_indices: replace holder idx in input_ids. Length the same to input_ids.
        token_in_tpu_pre_next_tokens_indices: value idx in next_tokens. Length the same to input_ids.
        next_tokens: next tokens on the TPU from previous step.
        placeholder_num: number of placeholders. placeholder_num <= len(token_in_tpu_cur_input_indices)
    Return:
        input_ids after replace placeholder tokens
    """
    assert input_ids.shape[0] == token_in_tpu_cur_input_indices.shape[
        0] == token_in_tpu_pre_next_tokens_indices.shape[0]
    device = input_ids.device
    mask = torch.arange(input_ids.shape[0], device=device) < placeholder_num
    new_token_values = next_tokens[token_in_tpu_pre_next_tokens_indices].to(
        input_ids.dtype)
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


#########################################################
# Ways to avoid recompilation
#########################################################
#
# The model executor has two primary components:
# 1. preparing the model and sampler inputs
# 2. executing the model and sampler.
# The core idea is to avoid any TPU computation during input preparation. For
# better compilation tracking and increased flexibility, the model execution and
# sampler are divided into several distinct components.
#
# Below are the detailed steps:
#
# Step 1
# It is recommended to avoid TPU operations when preparing the model and sampler
# inputs. CPU tensors can be prepared and transferred to the XLA device using
# cpu_tensor.to(xla_device), which only triggers CPU to TPU transfers and avoids
# compilation.
#
# Step 2
# The TPU execution should be decomposed into subgraphs (4 at the moment):
# 1. the main model
# 2. selected-logits computation for each request
# 3. sampler / logprob post-processing
# 4. encoder.
# Each subgraph should be decorated in a torch.compile. This is used to make
# sure that we have the same subgraph topology in both dummy_run and
# xecute_model. The results from these subgraphs should either be passed to
# other subgraphs, or transferred from TPU to CPU using xla_tensor.cpu() for
# subsequent processing on the CPU.
#
# Step 3
# The dummy_run should be comprehensive, ensuring all potential input shapes and
# branch predictions are included as subgraph inputs to facilitate
# pre-compilation.
class TPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        original_parallel_config: ParallelConfig | None = None,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.original_parallel_config = original_parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config
        _validate_libtpu_version()

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.check_recompilation = envs.VLLM_XLA_CHECK_RECOMPILATION

        # SPMD Related
        # TODO: Fix this
        self.use_spmd = envs.VLLM_XLA_USE_SPMD

        self.enforce_eager = model_config.enforce_eager

        self.num_xla_graphs = 0
        self._update_num_xla_graphs("init")
        # TODO: this is a temp fix as we don't support SymInt in torchTPU
        torch._dynamo.config.cache_size_limit = 1024

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            model_dtype = self.dtype
            if isinstance(model_dtype, str):
                self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            else:
                self.kv_cache_dtype = model_dtype
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]
        self._hidden_states_dtype = self.dtype

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.most_model_len = envs.VLLM_TPU_MOST_MODEL_LEN
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.num_blocks_per_most_len_req = (cdiv(
            self.most_model_len, self.block_size) if self.most_model_len
                                            is not None else None)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(scheduler_config.max_num_seqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = vllm_config.compilation_config.compile_sizes
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, "attention")
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.vocab_size = model_config.get_vocab_size()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config)
        # TODO: Support M-RoPE (e.g, Qwen2-VL)
        assert not self.uses_mrope, "TPU does not support M-RoPE yet."

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        self._attention_kernels_initialized = False
        # mm_hash -> encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # Initialize input batch early to avoid AttributeError in _update_states
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
        )

        # Cached torch/numpy tensor
        # The pytorch tensor and numpy array share the same buffer.
        # Sometimes the numpy op is faster so we create both.
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
            device="cpu",
        )
        # adjust num_reqs to avoid SMEM OOM.
        self.num_reqs_most_model_len = (min(
            PallasAttentionBackend.get_max_num_seqs(self.most_model_len,
                                                    self.block_size),
            self.max_num_reqs,
            self.max_num_tokens,
        ) if self.most_model_len is not None else None)
        self.num_reqs_max_model_len = min(
            PallasAttentionBackend.get_max_num_seqs(self.max_model_len,
                                                    self.block_size),
            self.max_num_reqs,
            self.max_num_tokens,
        )
        self.query_start_loc_cpu = torch.zeros(
            self.max_num_tokens + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            self.is_mm_embed_cpu = torch.zeros(
                self.max_num_tokens,
                dtype=torch.bool,
                device="cpu",
                pin_memory=self.pin_memory,
            )

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

        # tensors for structured decoding
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory)

        # TODO: torch.compile this
        self.sample_from_logits_func = self.sample_from_logits

        # For passing scheduler_output between successive
        # execute_model() and sample_tokens() calls.
        self.mm_embed_inputs: tuple[list[torch.Tensor],
                                    torch.Tensor] | None = None
        self.execute_model_state: ExecuteModelState | None = None
        self._pre_async_results: AsyncPreResults | None = None

        # Cache attention layer names to avoid calling
        # get_layers_from_vllm_config every step.
        self._attn_layer_names: list[str] | None = None

        # Pre-allocate request_distribution tensor on device to avoid
        # creating a new tensor + H2D copy every step.
        self._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)

        # Cache for decode-constant device tensors to avoid redundant H2D
        # copies. Keyed by (num_reqs, padded_num_reqs, use_max_model_len).
        self._decode_device_cache_key: tuple | None = None
        self._cached_query_start_loc: torch.Tensor | None = None
        self._cached_logits_indices: torch.Tensor | None = None
        self._cached_request_distribution: torch.Tensor | None = None

        # Create JAX Mesh for shard_map operations in TPU kernels.
        # Support TP by shaping the mesh as (data=1, model=tp_size).
        self.mesh = self._create_mesh_for_parallelism()

    def reset_mm_cache(self) -> None:
        pass

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

    def _get_requested_tp_size(self) -> int:
        # This integration supports TP>1 only via vLLM native multiprocess
        # parallelism. Per-worker JAX mesh must stay single-chip.
        if self.parallel_config.world_size == 1 and \
                self.parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "Single-process TPU mesh TP>1 is not supported in this path. "
                "Use vLLM multiprocess mode for --tensor-parallel-size > 1.")
        return 1

    def _create_mesh_for_parallelism(self) -> Mesh:
        tp_size = self._get_requested_tp_size()  # always 1 in current setting
        local_devices = list(jax.local_devices())
        if not local_devices:
            raise ValueError("No TPU devices are visible to create JAX mesh.")

        mesh_devices = np.asarray(local_devices[:tp_size]).reshape(
            (1, tp_size))
        mesh = Mesh(mesh_devices, axis_names=("data", "model"))
        mesh_device_ids = [
            getattr(device, "id", str(device))
            for device in local_devices[:tp_size]
        ]
        logger.info("Init mesh | tp_size=%d | device_ids=%s", tp_size,
                    mesh_device_ids)
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

    def _verify_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        stats = torch.tpu._get_cache_stats()
        curr_cached_graph = len(stats.per_entry_stats)
        assert self.num_xla_graphs == curr_cached_graph, (
            "Recompilation after warm up is detected during {}."
            " num_xla_graphs = {} curr_cached_graph = {}".format(
                case_str, self.num_xla_graphs, curr_cached_graph))

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert (new_req_data.sampling_params
                    is not None), "Pooling is not supported in TPU yet"
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1 else
                    sampling_params.prompt_logprobs)

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[
                req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            # Fill the empty index or append to the end
            req_index = removed_req_indices.pop(
            ) if removed_req_indices else None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

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

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

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
                    page_size_padded = (
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

    def _get_slot_mapping_metadata(self, num_reqs,
                                   num_scheduled_tokens_per_req) -> np.ndarray:
        """
        Computes metadata for mapping slots to blocks in the key-value (KV)
        cache for a batch of requests.

        This function determines, for each request in the batch, how the
        scheduled tokens are distributed across memory blocks, and generates
        metadata needed to map slices of tokens to their corresponding positions
        in the KV cache.

        Args:
            num_reqs (int): Number of requests in the current batch.
            num_scheduled_tokens_per_req (int or np.ndarray): Number of tokens
                to be scheduled for each request.

        Returns:
            np.ndarray: A 2D array of shape (total_block_len, 3), where each row
                contains:
                - kv_cache_start_index (int): The starting index in the KV cache
                  for the corresponding slice.
                - new_kv_start_index (int): The starting index in the new KV
                  cache for the corresponding slice.
                - slice_len (int): The length of the slice.
        """
        slices_start = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        slices_end = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                      num_scheduled_tokens_per_req)
        local_block_start_idx = slices_start // self.block_size
        local_block_end_idx = (slices_end - 1) // self.block_size
        no_repeat_req_indices = self.arange_np[:num_reqs]
        global_block_start_idx = (
            no_repeat_req_indices * self.max_num_blocks_per_req +
            local_block_start_idx)
        block_lens = local_block_end_idx - local_block_start_idx + 1
        global_block_start_idx = np.repeat(global_block_start_idx, block_lens)
        slice_arange = np.concatenate([self.arange_np[:n] for n in block_lens])
        global_block_indices = global_block_start_idx + slice_arange
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[global_block_indices].numpy()
        total_block_len = np.sum(block_lens)
        slot_mapping_slices = np.repeat(np.array([[0, self.block_size]],
                                                 dtype=np.int32),
                                        total_block_len,
                                        axis=0)
        cu_block_lens = np.zeros(len(block_lens) + 1, dtype=np.int32)
        np.cumsum(block_lens, out=cu_block_lens[1:])
        for req_idx in range(num_reqs):
            slot_mapping_slices[cu_block_lens[req_idx]][0] = (
                slices_start[req_idx] % self.block_size)
            slot_mapping_slices[
                cu_block_lens[req_idx + 1] -
                1][1] = (slices_end[req_idx] - 1) % self.block_size + 1
        slice_lens = slot_mapping_slices[:, 1] - slot_mapping_slices[:, 0]
        cu_slices_lens = np.zeros(len(slice_lens) + 1, dtype=np.int32)
        np.cumsum(slice_lens, out=cu_slices_lens[1:])
        kv_cache_start_indices = slot_mapping_slices[:, 0] + (block_numbers *
                                                              self.block_size)
        new_kv_start_indices = cu_slices_lens[:-1]
        slot_mapping_metadata = np.stack(
            [kv_cache_start_indices, new_kv_start_indices, slice_lens], axis=1)
        return slot_mapping_metadata

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
            self._pre_async_results.next_tokens_tpu,
            len(token_in_tpu_cur_input_indices))

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
        self.position_ids = self.positions_cpu[:
                                               padded_total_num_scheduled_tokens].to(
                                                   self.device,
                                                   non_blocking=True)
        if use_max_model_len:
            block_tables = self.block_table_cpu[:self.num_reqs_max_model_len, :
                                                self.max_num_blocks_per_req]
            block_tables[:num_reqs, :self.max_num_blocks_per_req] = (
                self.input_batch.block_table[0].get_cpu_tensor()
                [start_index:start_index + num_reqs])
            seq_lens = self.seq_lens_cpu[:self.num_reqs_max_model_len].to(
                self.device, non_blocking=True)
        else:
            assert self.num_reqs_most_model_len is not None
            block_tables = self.block_table_cpu[:self.
                                                num_reqs_most_model_len, :self.
                                                num_blocks_per_most_len_req]
            block_tables[:num_reqs, :self.num_blocks_per_most_len_req] = (
                self.input_batch.block_table[0].get_cpu_tensor()[
                    start_index:start_index +
                    num_reqs, :self.num_blocks_per_most_len_req])
            seq_lens = self.seq_lens_cpu[:self.num_reqs_most_model_len].to(
                self.device, non_blocking=True)
        # Flatten on CPU before H2D to avoid device-side as_strided/reshape
        # materialization on every decode step.
        block_tables = block_tables.reshape(-1).to(self.device,
                                                   non_blocking=True)

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

        attn_metadata = AttentionMetadata(
            input_positions=self.position_ids,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
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

        # Cache attention layer names on first call to avoid iterating
        # all model layers every step.
        if self._attn_layer_names is None:
            self._attn_layer_names = list(
                get_layers_from_vllm_config(self.vllm_config,
                                            Attention).keys())
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in self._attn_layer_names
        }

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

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # List of tuple (mm_hash, pos_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
                multimodal_cpu_fields=model.multimodal_cpu_fields,
        ):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)

            if isinstance(curr_group_outputs, torch.Tensor):
                encoder_outputs.append(curr_group_outputs)
            else:
                assert isinstance(curr_group_outputs, (list, tuple))
                for output in curr_group_outputs:
                    encoder_outputs.append(output)

        # Cache the encoder outputs.
        # NOTE (NickLucche) here we diverge from logic in other runners, as we
        # assume to only have whole mm items to process. Hence we avoid the
        # intrinsic dynamism that `scatter_mm_placeholders` introduces.
        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            assert (
                pos_info.is_embed is None
            ), "Expected all positions to be contiguous and embeddings."
            self.encoder_cache[mm_hash] = output

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)

        is_mm_embed = self.is_mm_embed_cpu
        is_mm_embed[:padded_total_num_scheduled_tokens] = False
        mm_embeds = list[torch.Tensor]()
        req_start_idx = 0

        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens

            # TODO unroll loop and assume/enforce --disable_chunked_mm_input
            # NOTE (NickLucche) here we diverge from logic in other runners, as
            # we assume to only have whole mm items to process. Hence we avoid
            # the intrinsic dynamism that `gather_mm_placeholders` introduces.
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                assert (
                    pos_info.is_embed is None
                ), "Expected all positions to be contiguous and embeddings."

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                is_mm_embed[req_start_pos + start_idx:req_start_pos +
                            end_idx] = True

                # Only whole mm items are processed
                mm_embeds.append(encoder_output)

            req_start_idx += num_scheduled_tokens

        is_mm_embed = is_mm_embed[:padded_total_num_scheduled_tokens].to(
            self.device)

        return mm_embeds, is_mm_embed

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
        self.input_batch.temperature.copy_(
            self.input_batch.temperature_cpu_tensor)
        for logits, num_reqs in zip(state.logits_list, state.num_reqs_list):
            cur_end_idx = cur_start_idx + num_reqs
            temperatures_tpu = torch.ones((logits.shape[0], 1),
                                          dtype=logits.dtype,
                                          device=logits.device)
            num_active_reqs = cur_end_idx - cur_start_idx
            temperatures_tpu[:num_active_reqs,
                             0] = self.input_batch.temperature[
                                 cur_start_idx:cur_end_idx]
            if grammar_output is not None:
                require_struct_decoding, grammar_bitmask_padded, arange = (
                    self.prepare_structured_decoding_input(
                        logits, grammar_output))
                logits = self.structured_decode(require_struct_decoding,
                                                grammar_bitmask_padded, logits,
                                                arange)
            u = torch.rand_like(logits)
            selected_token_ids = self.sample_from_logits_func(
                logits, temperatures_tpu, u)
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

    def update_config(self, overrides: dict[str, Any]) -> None:
        # TODO: TPU config may need extra validation
        # https://github.com/vllm-project/vllm/pull/20095#discussion_r2201497754
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}")
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self) -> None:
        self.device_config.device = api.tpu_device()
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

    def reload_weights(self) -> None:
        assert (getattr(self, "model", None)
                is not None), "Cannot reload weights before model is loaded."
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        with set_current_vllm_config(self.vllm_config):
            model_loader.load_weights(self.model,
                                      model_config=self.model_config)
        self._attention_kernels_initialized = False
        self._initialize_attention_kernels()

    @torch.no_grad()
    def _dummy_run(self, num_tokens: int, num_reqs: int,
                   num_blocks: int) -> None:
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
        position_ids = torch.zeros(num_tokens,
                                   dtype=torch.int32).to(self.device)
        block_tables = torch.zeros((num_reqs * num_blocks, ),
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
        attn_metadata = AttentionMetadata(
            input_positions=position_ids,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
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

    def _set_active_loras(self, prompt_lora_mapping, token_lora_mapping,
                          lora_requests) -> None:
        super()._set_active_loras(prompt_lora_mapping, token_lora_mapping,
                                  lora_requests)

    def _precompile_backbone(self) -> None:
        logger.info("Compiling the model with different input shapes.")
        start = time.perf_counter()
        for num_tokens in self.num_tokens_paddings:
            logger.info("  -- num_tokens: %d", num_tokens)
            self._dummy_run(num_tokens, self.num_reqs_max_model_len,
                            self.max_num_blocks_per_req)
            if self.most_model_len is not None:
                self._dummy_run(
                    num_tokens,
                    self.num_reqs_most_model_len,
                    self.num_blocks_per_most_len_req,
                )
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("model backbone")

    def _precompile_compute_selected_logits(self) -> None:
        logger.info(
            "Compiling compute_selected_logits with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
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
                logger.info("  -- num_tokens: %d, num_seqs: %d", num_tokens,
                            num_reqs)
                if num_reqs >= min(num_tokens, self.max_num_reqs):
                    break
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("compute_selected_logits")

    def _precompile_structured_decoding(self) -> None:
        logger.info(
            "Compiling structured_decoding with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            dummy_require_struct_decoding = self.require_structured_out_cpu[:num_reqs].to(
                self.device)
            dummy_grammar_bitmask = self.grammar_bitmask_cpu[:num_reqs].to(
                self.device)
            arange = self.structured_decode_arange.to(self.device)
            out = self.structured_decode(
                dummy_require_struct_decoding,
                dummy_grammar_bitmask,
                dummy_logits,
                arange,
            )
            sync.synchronize(out, wait=True)
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("structured_decoding")

    def _precompile_sample_from_logits(self) -> None:
        logger.info(
            "Compiling sample_from_logits with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            dummy_temperatures = torch.ones((num_reqs, 1),
                                            device=self.device,
                                            dtype=self._hidden_states_dtype)
            dummy_u = torch.rand_like(dummy_logits)
            out = self.sample_from_logits_func(dummy_logits,
                                               dummy_temperatures, dummy_u)
            sync.synchronize(out, wait=True)
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("sample_from_logits")

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            dummy_tokens = torch.zeros((num_reqs, 1),
                                       dtype=torch.int64).to(self.device)
            out = self.gather_logprobs(dummy_logits, dummy_tokens)
            sync.synchronize(out.logprobs, wait=True)
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("gather_logprobs")

    def capture_model(self) -> None:
        """
        Precompile all the subgraphs with possible input shapes.
        """
        if self.enforce_eager:
            return

        with self.maybe_setup_dummy_loras(self.lora_config):
            self._precompile_backbone()
            self._precompile_compute_selected_logits()
            self._precompile_structured_decoding()
            self._precompile_sample_from_logits()
            self._precompile_gather_logprobs()

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        self._initialize_attention_kernels()

        # TODO: figure out if this can be fixed
        # Run eagerly (without torch.compile) during profiling.
        # The profiling run executes before KV cache is allocated, so
        # kv_cache.numel() == 0. If torch.compile traces this path, it
        # specializes the graph with the early-return branch in attention.
        # So we will not compile here and instead let the compilation happen in
        # the `precompile_backbone` step.
        with _bypass_torch_compile(self.model):
            self._dummy_run(num_tokens, self.num_reqs_max_model_len,
                            self.max_num_blocks_per_req)

    def maybe_setup_cross_layer_kv_sharing(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        pass

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not supported yet."
            )

        if (kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                != self.block_size):
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=[
                    kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                ],
                kernel_block_sizes=[
                    kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                ],
            )
        # Verify dtype compatibility between block_table_cpu and input_batch
        assert (self.block_table_cpu.dtype ==
                self.input_batch.block_table[0].get_cpu_tensor().dtype)

        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert (
                len(kv_cache_tensor.shared_by) == 1
            ), "KV cache tensor shared by multiple layers is not supported in TPU."
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_caches: dict[str, torch.Tensor] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes  # noqa
                if isinstance(kv_cache_spec, AttentionSpec):
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
                    shape_nbytes = torch.empty((), dtype=dtype).element_size()
                    for dim in kv_cache_shape:
                        shape_nbytes *= dim
                    assert shape_nbytes == tensor_size, (
                        f"KV cache tensor size mismatch for {layer_name}: "
                        f"{shape_nbytes=} {tensor_size=} "
                        f"{kv_cache_shape=} {dtype=} "
                        f"page_size_bytes={kv_cache_spec.page_size_bytes}")

                    tpu_kv_cache = torch.zeros(kv_cache_shape,
                                               dtype=dtype).to(self.device)

                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        # Set up cross-layer KV cache sharing if needed
        self.maybe_setup_cross_layer_kv_sharing(kv_caches, kv_cache_config)

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

    def reset_dynamo_cache(self):
        # NOTE: We check `is_multimodal_model` instead of `supports_mm_inputs`
        # since the compiled model object of the language backbone of a
        # multimodal model needs to be extracted via `get_language_model`.
        if self.model_config.is_multimodal_model:
            compiled_model = self.model.get_language_model().model
        else:
            compiled_model = self.model.model
        if isinstance(compiled_model, TorchCompileWithNoGuardsWrapper):
            logger.info("Clear dynamo cache and cached dynamo bytecode.")
            torch._dynamo.eval_frame.remove_from_cache(
                compiled_model.original_code_object())
            # Reset the wrapper to re-initialize.
            compiled_model.compiled = False
            TorchCompileWithNoGuardsWrapper.__init__(compiled_model)

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
    def sample_from_logits(self, logits: torch.Tensor,
                           temperatures: torch.Tensor,
                           u: torch.Tensor) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced
        separately from `forward` for lighter compilation overhead.
        """
        is_greedy = temperatures <= SAMPLING_EPS
        scaled_logits = self._apply_temperature(logits, temperatures)
        gumbel_noise = -torch.log(-torch.log(u))
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

    def embed_multimodal(self, *args, **kwargs):
        return self.model.embed_multimodal(*args, **kwargs)

    def embed_input_ids(self, *args, **kwargs):
        return self.model.embed_input_ids(*args, **kwargs)

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
