# SPDX-License-Identifier: Apache-2.0
""" KV-cache connector for P/D disagg serving of HMA models (e.g. Qwen 3.5).

  *  Block ids are per-kv-cache-group (``list[list[int]]``).
  * ``runner.kv_caches`` is HMA; an attention layer is a single tensor;
     a Mamba layer is a ``(conv_state, ssm_state)`` tuple of differing
     shape and dtype. We flatten it into per-rank "arrays" and stage
    / transfer / insert each array with its own per-group block count.
"""

import time
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorRole,
                                                               SupportsHMA)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.host_kv_shm_hma import (
    HostKVShmPoolHMA, PoolSpecHMA)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPUConnector, TPUConnectorScheduler, TPUConnectorWorker, get_uuid)
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import (LoadMeta,
                                                                SendMeta)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class _DoneFuture:

    def wait(self) -> None:
        return


_DONE_FUTURE = _DoneFuture()

__all__ = [
    "TPUConnectorHMA",
    "TPUConnectorHMAScheduler",
    "TPUConnectorHMAWorker",
]


class TPUConnectorHMA(TPUConnector, SupportsHMA):
    """TPU connector supporting the hybrid memory allocator (HMA)."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 role: KVConnectorRole,
                 kv_cache_config: Any = None):
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUConnectorHMAScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = TPUConnectorHMAWorker(vllm_config)

    def register_kv_caches(self, kv_caches: dict[str, Any]):
        if self.connector_worker is not None:
            self.connector_worker.named_kv_caches = kv_caches

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(
            request, block_ids)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        raise AssertionError(
            "TPUConnectorHMA implements SupportsHMA and expects "
            "`request_finished_all_groups` to be invoked.")


class TPUConnectorHMAScheduler(TPUConnectorScheduler):

    def _maybe_truncate_for_mamba(self, request: "Request") -> None:
        """P-side: drop the last prompt token so the prefiller computes the
        Mamba recurrent state h(N-1) (state before the last token) instead
        of h(N).

        For attention that recompute is idempotent, but a Mamba layer would
        fold the last token through the conv1d/SSM recurrence a second time
        (P's transferred state already includes it), corrupting the first
        decode logit. By having P prefill shipping h(N-1),
        the decoder's recompute of the last token reproduces h(N) exactly.
        """
        if request.num_prompt_tokens <= 1:
            return
        params = request.kv_transfer_params
        if params is not None and params.get("_p_side_truncated"):
            return
        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        else:
            return
        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        if request.kv_transfer_params is None:
            request.kv_transfer_params = {}
        request.kv_transfer_params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            self._maybe_truncate_for_mamba(request)
            return 0, False
        if not request.kv_transfer_params:
            return 0, False

        # Pull all prompt tokens except the last one. vLLM forces D to
        # recompute the last prompt token (see _maybe_truncate_for_mamba).
        count = max((len(request.prompt_token_ids) - 1) - num_computed_tokens,
                    0)
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            local_block_ids = list(blocks.get_block_ids())
            assert all(isinstance(g, list) for g in local_block_ids), (
                f"Expected list[list[int]] from blocks.get_block_ids() in "
                f"HMA mode; got {local_block_ids}")
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        else:
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        load_meta = self.reqs_to_load[request.request_id]
        logger.info(
            "TPUConnectorHMAScheduler Decode --> load queued | req_id=%s | "
            "uuid=%s | remote_host=%s | remote_port=%s | pending_loads=%d",
            request.request_id, load_meta.uuid, load_meta.remote_host,
            load_meta.remote_port, len(self.reqs_to_load))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self.is_producer:
            return False, None

        # max_tokens is forced to 1 by the proxy on P, so the only clean
        # finish is length cap.
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # Per-group, no trim/rounding for HMA.
        computed_per_group: list[list[int]] = [
            list(one_group) for one_group in block_ids
        ]
        delay_free_blocks = any(len(g) > 0 for g in computed_per_group)

        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = (time.perf_counter() +
                               dist_utils.get_p2p_wait_pull_timeout())
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_per_group,
                expiration_time=expiration_time)
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_per_group,
                                      remote_host=self.kv_ip,
                                      remote_port=self.kv_port)
            logger.info(
                "TPUConnectorHMAScheduler Prefill --> send queued | "
                "req_id=%s | uuid=%s | num_prompt_tokens=%d | "
                "pending_sends=%d", request.request_id, uuid,
                len(request.prompt_token_ids), len(self.reqs_to_send))
        else:
            kv_transfer_params = {}
        return delay_free_blocks, kv_transfer_params

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        raise AssertionError(
            "This scheduler only expects `request_finished_all_groups`.")


class TPUConnectorHMAWorker(TPUConnectorWorker):

    # layer_name -> kv cache dict used to resolve each positional kv cache's
    # group by object id in _map_layers_to_groups.
    named_kv_caches: Optional[dict[str, Any]] = None

    def _extract_kv_layout(self) -> None:
        runner = self.runner
        kv_caches = runner.kv_caches
        groups = runner.kv_cache_config.kv_cache_groups
        self.group_is_mamba: list[bool] = [
            isinstance(g.kv_cache_spec, MambaSpec) for g in groups
        ]
        layer_to_group = self._map_layers_to_groups(kv_caches, groups)

        # Flatten into per-rank arrays. A mamba layer (tuple) contributes one
        # array per state tensor; an attention layer contributes one array.
        self.array_to_runner: list[tuple[int, Optional[int]]] = []
        self.array_to_group: list[int] = []
        self.array_inner_shape: list[tuple] = []
        self.array_dtype: list = []
        for layer_idx, cache in enumerate(kv_caches):
            gid = layer_to_group[layer_idx]
            if isinstance(cache, tuple):
                for sub, t in enumerate(cache):
                    self.array_to_runner.append((layer_idx, sub))
                    self.array_to_group.append(gid)
                    self.array_inner_shape.append(tuple(t.shape[1:]))
                    self.array_dtype.append(t.dtype)
            else:
                self.array_to_runner.append((layer_idx, None))
                self.array_to_group.append(gid)
                self.array_inner_shape.append(tuple(cache.shape[1:]))
                self.array_dtype.append(cache.dtype)

        self.num_arrays = len(self.array_to_runner)
        # num_layers is the per-rank wire frame count in the base class.
        self.num_layers = self.num_arrays
        # Representative shape/dtype for the base's logging only.
        self.shape = [0] + list(self.array_inner_shape[0])
        self.dtype = self.array_dtype[0]
        logger.info(
            "TPUConnectorHMAWorker %s rank%d --> layout | num_groups=%d | "
            "group_is_mamba=%s | num_layers=%d | num_arrays=%d | ",
            self.node_id, self.tp_rank, len(groups), self.group_is_mamba,
            len(kv_caches), self.num_arrays)

    def _map_layers_to_groups(self, kv_caches: list, groups: list) -> list:
        """Map each entry in ``runner.kv_caches`` to its kv-cache-group id.
        """
        named = self.named_kv_caches
        name_to_group: dict[str, int] = {
            name: gid
            for gid, g in enumerate(groups)
            for name in g.layer_names
        }
        id_to_name = {id(cache): name for name, cache in named.items()}
        mapping: list[Optional[int]] = []
        for cache in kv_caches:
            name = id_to_name.get(id(cache))
            assert name is not None, (
                "a runner kv cache was not found by identity in the "
                "registered named_kv_caches dict.")
            mapping.append(name_to_group[name])
        return mapping  # type: ignore[return-value]

    def _build_pool_spec(self) -> PoolSpecHMA:
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        attn_max_blocks = (max_model_len + block_size - 1) // block_size
        # Mamba groups hold a single recurrent state per sequence.
        array_max_blocks = tuple(
            1 if self.group_is_mamba[self.
                                     array_to_group[a]] else attn_max_blocks
            for a in range(self.num_arrays))

        def make_spec(num_slots: int) -> PoolSpecHMA:
            return PoolSpecHMA(
                num_slots=num_slots,
                tp_size=self.tp_size,
                num_arrays=self.num_arrays,
                array_inner_shape=tuple(self.array_inner_shape),
                array_dtype=tuple(self.array_dtype),
                array_max_blocks=array_max_blocks,
                array_to_group=tuple(self.array_to_group),
            )

        probe = make_spec(1)
        per_slot_bytes = probe.per_slot_bytes
        budget_bytes = int(dist_utils.get_kv_shm_pool_gb() * (1024**3))
        num_slots = max(1, budget_bytes // per_slot_bytes)
        logger.info(
            "TPUConnectorHMAWorker %s rank%d --> shm pool budget=%.2fGB "
            "per_slot=%.2fMB -> num_slots=%d", self.node_id, self.tp_rank,
            budget_bytes / (1024**3), per_slot_bytes / (1024**2), num_slots)
        return make_spec(num_slots)

    def _pool_create(self, spec: PoolSpecHMA,
                     shm_name: str) -> HostKVShmPoolHMA:
        return HostKVShmPoolHMA.create(spec, shm_name)

    def _pool_attach(self, spec: PoolSpecHMA,
                     shm_name: str) -> HostKVShmPoolHMA:
        return HostKVShmPoolHMA.attach(spec, shm_name)

    def _blocks_token(self, local_block_ids: Any) -> tuple:
        """Per-kv-cache-group block counts. Indexed by group id so the pool
        can resolve each array's runtime block count via array_to_group."""
        return tuple(len(g) for g in local_block_ids)

    def _pull_response_header(self, entry) -> dict:
        return {
            "tp_size": self.tp_size,
            "num_layers": self.num_layers,
            "num_blocks": entry.num_blocks,
        }

    def _maybe_enable_kv_scatter(self) -> None:
        """The fused kv_scatter kernel is not available for HMA yet."""
        return

    def _warmup_block_sizes(self) -> list[int]:
        """ Skip the kv ops warmup path for now. """
        return []

    # ---- D2H staging (producer) -----------------------------------------
    def _build_d2h_views(self, slot_idx: int, blocks: Any,
                         block_ids: Any) -> tuple[list, list, int]:
        """``blocks`` is the per-group count token; ``block_ids`` is the
        per-group list[list[int]] to gather. One (src, dst) pair per flat
        array."""
        kv_caches = self.runner.kv_caches
        tpu_tensors: list = []
        cpu_tensors: list = []
        total_bytes = 0
        for a in range(self.num_arrays):
            gid = self.array_to_group[a]
            ids = block_ids[gid]
            indices = torch.tensor(ids, dtype=torch.int64, device=self.device)
            layer_idx, sub = self.array_to_runner[a]
            src = kv_caches[layer_idx] if sub is None else kv_caches[
                layer_idx][sub]
            src_shard = torch.index_select(src, 0, indices)
            dest_view = self._coord_pool.layer_view(slot_idx, self.tp_rank, a,
                                                    blocks)
            tpu_tensors.append(src_shard)
            cpu_tensors.append(dest_view)
            total_bytes += dest_view.numel() * dest_view.element_size()
        return tpu_tensors, cpu_tensors, total_bytes

    # The batched DMA primitives (batch_transfer_d2h/h2d) assume a *uniform*
    # batch. This is not working for mamba+attention arrays yet. Use ordinary
    # shape-aware torch copies instead until DMA primitives are available.
    def _stage_d2h(self, slot_idx: int, blocks: Any,
                   block_ids: Any) -> tuple[Any, list, list, int]:
        tpu_tensors, cpu_tensors, total_bytes = self._build_d2h_views(
            slot_idx, blocks, block_ids)
        for src_shard, dest_view in zip(tpu_tensors, cpu_tensors):
            dest_view.copy_(src_shard.cpu())
        return _DONE_FUTURE, [], [], total_bytes

    def _wait_stage(self, future: Any) -> None:
        # D2H/H2D completed synchronously in the helpers above.
        return

    def _h2d_into_device_async(
            self, src_views: list) -> tuple[Any, list[torch.Tensor]]:
        return _DONE_FUTURE, [v.to(self.device) for v in src_views]

    # ---- H2D + insert (consumer) ----------------------------------------
    def _coord_scatter_shard(self, slot_idx: int, blocks: Any,
                             local_blocks: Any) -> None:
        """H2D each array's shm shard into HBM and insert it into the kv
        cache at the per-group local block ids. ``blocks`` is the per-group
        count token; ``local_blocks`` is the per-group list[list[int]].

        This will be replaced by KV scatter Pallas kernel implementation."""
        if not any(len(g) > 0 for g in local_blocks):
            return
        kv_caches = self.runner.kv_caches

        alloc_t0 = time.perf_counter()
        src_views = [
            self._coord_pool.layer_view(slot_idx, self.tp_rank, a, blocks)
            for a in range(self.num_arrays)
        ]
        h2d_total_bytes = sum(v.numel() * v.element_size() for v in src_views)
        alloc_t1 = time.perf_counter()

        h2d_t0 = time.perf_counter()
        device_shards = [v.to(self.device) for v in src_views]
        h2d_t1 = time.perf_counter()

        insert_t0 = time.perf_counter()
        # Insert each array's shard into its kv cache via functional index_put.
        group_indices: dict[int, torch.Tensor] = {}
        for gid, ids in enumerate(local_blocks):
            if ids:
                group_indices[gid] = torch.tensor(ids,
                                                  dtype=torch.int64,
                                                  device=self.device)

        new_attn: dict[int, torch.Tensor] = {}
        new_mamba: dict[int, list] = {}
        for a in range(self.num_arrays):
            indices = group_indices.get(self.array_to_group[a])
            if indices is None:
                continue
            li, sub = self.array_to_runner[a]
            if sub is None:
                new_attn[li] = kv_caches[li].index_put((indices, ),
                                                       device_shards[a])
            else:
                lst = new_mamba.setdefault(li, list(kv_caches[li]))
                lst[sub] = lst[sub].index_put((indices, ), device_shards[a])

        for layer_idx, new_cache in new_attn.items():
            self._synchronize_device(new_cache)
            self._replace_runner_kv_cache(layer_idx, new_cache)
        for layer_idx, lst in new_mamba.items():
            for t in lst:
                self._synchronize_device(t)
            self._replace_runner_kv_cache(layer_idx, tuple(lst))
        insert_t1 = time.perf_counter()

        h2d_ms = (h2d_t1 - h2d_t0) * 1000.0
        insert_ms = (insert_t1 - insert_t0) * 1000.0
        h2d_mb = h2d_total_bytes / (1024 * 1024)
        logger.info(
            "TPUConnectorHMAWorker %s rank%d --> insert slot=%d "
            "blocks_per_group=%s arrays=%d alloc=%.2fms h2d=%.2fms "
            "insert=%.2fms h2d_MiB=%.2f", self.node_id, self.tp_rank, slot_idx,
            list(blocks), self.num_arrays, (alloc_t1 - alloc_t0) * 1000.0,
            h2d_ms, insert_ms, h2d_mb)
