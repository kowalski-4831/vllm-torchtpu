# SPDX-License-Identifier: Apache-2.0
"""
TPUConnector: KV-cache connector for P/D disaggregated serving on TPU.

Ported from tpu_inference/distributed/tpu_connector.py in the tpu-inference
(JAX) project, with the JAX transfer-server data plane replaced by a ZMQ-based
pull-on-demand transport that moves torch tensors over the TPU host network.

The generic shm-staged ZMQ machinery lives in
``tpu_inference.distributed.kv_transfer.zmq_shm_base.ZmqShmKvConnectorBase``;
this file holds the TPU-specific transport hooks plus the scheduler half
of the connector.
"""

import time
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import torch
from torch_tpu._internal.batch_transfer import (batch_transfer_d2h,
                                                batch_transfer_d2h_sync,
                                                batch_transfer_h2d,
                                                batch_transfer_h2d_sync)
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics, KVConnectorStats, PromMetric, PromMetricT)
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import tpu_inference.distributed.utils as dist_utils
from tpu_inference.distributed.kv_transfer import kv_scatter
from tpu_inference.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)
from tpu_inference.distributed.kv_transfer.zmq_shm_base import (
    LoadMeta, ReqId, SendMeta, TPUConnectorMetadata, ZmqShmKvConnectorBase,
    _CoordRecvEntry, _CoordSendEntry)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

__all__ = [
    "TPUConnector",
    "TPUConnectorScheduler",
    "TPUConnectorWorker",
    "TPUConnectorMetadata",
    "SendMeta",
    "LoadMeta",
    "_CoordSendEntry",
    "_CoordRecvEntry",
]


class TPUConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata: Optional[TPUConnectorMetadata] = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = TPUConnectorWorker(vllm_config)

    # ---- Scheduler-side methods -----------------------------------------
    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> TPUConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def get_finished_count(self) -> int:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_finished_count()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
            cls,
            data: dict[str, Any] | None = None) -> KVConnectorStats | None:
        return (TpuKVConnectorStats(
            data=data) if data is not None else TpuKVConnectorStats())

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return TpuKVConnectorPromMetrics(vllm_config, metric_types, labelnames,
                                         per_engine_labelvalues)

    # ---- Worker-side methods --------------------------------------------
    def register_kv_caches(self, kv_caches: list[torch.Tensor]):
        """No-op: we call register_runner() from the runner after
        bind_kv_cache, and read runner.kv_caches lazily in the worker."""
        pass

    def register_runner(self, runner: TPUModelRunner) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self, _, **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.process_send_load(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Layer-wise load is not supported on TPU."""
        pass

    def save_kv_layer(self, *args, **kwargs) -> None:
        """Layer-wise save is not supported on TPU."""
        pass

    def wait_for_save(self):
        """No-op. See comment in tpu_inference's TPUConnector: reqs_to_send is
        only populated after the request finishes prefill, at which point
        total_num_scheduled_tokens may be 0 and wait_for_save is not called
        by the KVConnectorModelRunnerMixin. We run the send from
        start_load_kv -> process_send_load instead."""
        pass

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()


class TPUConnectorScheduler:

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.block_size = vllm_config.cache_config.block_size

        # Populated by request_finished() on P.
        self.reqs_to_send: dict[ReqId, SendMeta] = {}
        # Populated by update_state_after_alloc() on D.
        self.reqs_to_load: dict[ReqId, LoadMeta] = {}

        self.kv_ip = dist_utils.get_kv_ips()
        self.kv_port = dist_utils.get_kv_ports()
        logger.info("TPUConnectorScheduler --> kv_ip=%s | kv_port=%s",
                    self.kv_ip, self.kv_port)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        D workers use this to get the number of new tokens
        that can be loaded from remote P workers.
        No-op for P workers.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that will be loaded from the
                  external KV cache.
                - If async loading. Must be 'False' for TPU connector
                  because TPU pulls KV cache in a blocking way.

        """
        if self.is_producer or not request.kv_transfer_params:
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        # Rounding must match request_finished()'s remote_block_ids computation.
        rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids),
                                               self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        # The pull is blocking at the ZMQ layer, but we wrap it in a thread
        # pool so from the scheduler's perspective it's async.
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
            local_block_ids = blocks.get_block_ids()[0]
            # D must pull the whole prefill blocks regardless of partial
            # prefix-cache hits, because the transport has no RDMA-style
            # partial-pull and P publishes the full payload under a uuid.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        else:
            # Full prefix-cache hit or async pull done -- we still need to
            # notify P so it can free the pending buffer.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        logger.info(
            "TPUConnectorScheduler update_state_after_alloc --> reqs_to_load=%s",
            self.reqs_to_load)

    def build_connector_meta(self) -> TPUConnectorMetadata:
        meta = TPUConnectorMetadata()
        if self.is_producer:
            meta.reqs_to_send = self.reqs_to_send
            self.reqs_to_send = {}
        else:
            meta.reqs_to_load = self.reqs_to_load
            self.reqs_to_load = {}
        return meta

    def get_finished_count(self) -> int:
        return len(self.kv_ip) if isinstance(self.kv_ip, list) else 1

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self.is_producer:
            return False, None

        # max_tokens is forced to 1 by the proxy on P, so the only way the
        # prefill finishes cleanly is length cap.
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # Only transfer full blocks; let D re-prefill the trailing partial
        # block locally.
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = block_ids if all_full else block_ids[:-1]

        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = (time.perf_counter() +
                               dist_utils.get_p2p_wait_pull_timeout())
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_block_ids,
                expiration_time=expiration_time)
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_block_ids,
                                      remote_host=self.kv_ip,
                                      remote_port=self.kv_port)
            logger.info(
                "TPUConnectorScheduler --> reqs_to_send=%s | kv_transfer_params=%s",
                self.reqs_to_send, kv_transfer_params)
        else:
            kv_transfer_params = {}

        return delay_free_blocks, kv_transfer_params


class TPUConnectorWorker(ZmqShmKvConnectorBase):
    """TPU-specific transport hooks for the generic ZmqShmKvConnectorBase.
    """

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the worker.
        """
        # Clear stats for next iteration
        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None

    def _build_d2h_views(self, slot_idx: int, num_blocks: int,
                         block_ids: list[int]) -> tuple[list, list, int]:
        indices = torch.tensor(block_ids,
                               dtype=torch.int64,
                               device=self.device)
        kv_caches = self.runner.kv_caches
        tpu_tensors: list = []
        cpu_tensors: list = []
        d2h_total_bytes = 0
        for layer_idx, cache in enumerate(kv_caches):
            src_shard = torch.index_select(cache, 0, indices)
            dest_view = self._coord_pool.layer_view(slot_idx, self.tp_rank,
                                                    layer_idx, num_blocks)
            tpu_tensors.append(src_shard)
            cpu_tensors.append(dest_view)
            d2h_total_bytes += dest_view.numel() * dest_view.element_size()
        return tpu_tensors, cpu_tensors, d2h_total_bytes

    def _stage_d2h(self, slot_idx: int, num_blocks: int,
                   block_ids: list[int]) -> tuple[Any, list, list, int]:
        tpu_tensors, cpu_tensors, total_bytes = self._build_d2h_views(
            slot_idx, num_blocks, block_ids)
        future = batch_transfer_d2h(tpu_tensors, cpu_tensors)
        return future, tpu_tensors, cpu_tensors, total_bytes

    def _stage_d2h_sync(self, slot_idx: int, num_blocks: int,
                        block_ids: list[int]) -> None:
        tpu_tensors, cpu_tensors, _ = self._build_d2h_views(
            slot_idx, num_blocks, block_ids)
        batch_transfer_d2h_sync(tpu_tensors, cpu_tensors)

    def _wait_stage(self, future: Any) -> None:
        future.wait()

    def _h2d_into_device(self, src_views: list) -> list[torch.Tensor]:
        device_shards = [
            torch.empty(v.shape, dtype=v.dtype, device=self.device)
            for v in src_views
        ]
        batch_transfer_h2d_sync(src_views, device_shards)
        return device_shards

    def _h2d_into_device_async(
            self, src_views: list) -> tuple[Any, list[torch.Tensor]]:
        device_shards = [
            torch.empty(v.shape, dtype=v.dtype, device=self.device)
            for v in src_views
        ]
        future = batch_transfer_h2d(src_views, device_shards)
        return future, device_shards

    def _synchronize_device(self, tensor: torch.Tensor) -> None:
        from torch_tpu._internal.sync import sync as tpu_sync
        tpu_sync.synchronize(tensor)

    def _try_fast_scatter(
            self, device_shards: list[torch.Tensor],
            kv_caches: list[torch.Tensor],
            local_blocks: list[int]) -> Optional[list[torch.Tensor]]:
        if not self._kv_scatter_enabled:
            return None
        dest_blocks_dev = torch.tensor(local_blocks,
                                       dtype=torch.int32,
                                       device=self.device)
        prebuilt = kv_scatter.prepare_scatter_args(dest_blocks_dev,
                                                   self.device)
        return kv_scatter.multi_layer_scatter_into(device_shards,
                                                   kv_caches,
                                                   dest_blocks_dev,
                                                   prebuilt_args=prebuilt)

    def _maybe_enable_kv_scatter(self) -> None:
        """Run the startup smoke test for the multi-layer scatter kernel
        (``kv_scatter``) and flip it on if it passes. Consumer-only. On
        by default; any failure falls back permanently to ``index_put_``
        with a warning. Smoke-tests with a small num_layers; compile for
        the real per-worker num_layers happens on first invocation
        (typically during warmup)."""
        if self.is_producer:
            return
        if not kv_scatter.scatter_available():
            logger.warning(
                "TPUConnectorWorker %s rank%d --> kv_scatter kernel "
                "unavailable; staying on index_put_", self.node_id,
                self.tp_rank)
            return
        trailing = tuple(self.shape[1:]) if len(self.shape) > 1 else (128, )
        ok = kv_scatter.smoke_test_multi_layer_scatter(
            device=self.device,
            num_layers=2,
            num_blocks=4,
            trailing_shape=trailing,
            dtype=self.dtype,
        )
        if ok:
            self._kv_scatter_enabled = True
            logger.info(
                "TPUConnectorWorker %s rank%d --> multi-layer scatter "
                "enabled for insert path", self.node_id, self.tp_rank)
        else:
            logger.warning(
                "TPUConnectorWorker %s rank%d --> multi-layer scatter "
                "smoke test failed; falling back to index_put_", self.node_id,
                self.tp_rank)


def get_uuid() -> int:
    int128 = uuid4().int
    # Stay under 64-bit so JSON-encoded responses through the proxy are safe.
    return int128 >> 78
