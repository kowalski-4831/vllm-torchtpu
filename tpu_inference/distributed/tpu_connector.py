# SPDX-License-Identifier: Apache-2.0
"""
TPUConnector: KV-cache connector for P/D disaggregated serving on TPU.

Ported from tpu_inference/distributed/tpu_connector.py in the tpu-inference
(JAX) project, with the JAX transfer-server data plane replaced by a ZMQ-based
pull-on-demand transport that moves torch tensors over the TPU host network.

vLLM's multiproc_executor forks one process per TP rank, so we can't let
every rank bind the same TCP port. Rank 0 acts as a per-host coordinator
that owns both external ZMQ endpoints and a SharedMemory pool of KV
staging slots; other ranks talk to rank 0 over an ipc:// ROUTER/DEALER
side channel. See _coord_* methods and host_kv_shm.HostKVShmPool.

  P workflow:
    P receives the request (max_output_tokens=1). P prefills; when the
    request is FINISHED_LENGTH_CAPPED, the scheduler's request_finished()
    stores a SendMeta in reqs_to_send and returns kv_transfer_params =
    {uuid, remote_block_ids, remote_host, remote_port} to the proxy server.
    On the next scheduler step rank 0 acquires a shm slot, broadcasts
    STAGE_NOTIFY, and every rank D2Hs its shard into the slot. Rank 0's
    ROUTER serves PULL <uuid> requests by scatter-gather sending the
    per-rank memoryviews straight from shm. D notifies P over a side
    channel when the pull is done; P frees the slot. Unacked entries
    expire after TPU_P2P_WAIT_PULL_TIMEOUT.

  D workflow:
    D receives the request with kv_transfer_params. Its
    get_num_new_matched_tokens() reports the remote-hit count and
    update_state_after_alloc() stores a LoadMeta. On the next scheduler
    step rank 0 acquires a shm slot, opens N parallel DEALERs to P, pulls
    and unpacks directly into shm, then broadcasts LOAD_NOTIFY. On the
    following scheduler step (remote_block_ids=None) every rank scatters
    its shard from shm into HBM and acks COPY_DONE; rank 0 notifies P and
    releases the slot once all ranks have copied.
"""

import pickle
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import tpu_inference.distributed.utils as dist_utils
from tpu_inference import envs
from tpu_inference.distributed import kv_scatter
from tpu_inference.distributed.host_kv_shm import HostKVShmPool, PoolSpec
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

ReqId = str

logger = init_logger(__name__)

# Wire-format tags for the ZMQ data channel.
_MSG_PULL = b"PULL"
_MSG_OK = b"OK"
_MSG_ERR = b"ERR"

# Phases tracked by _LatencyTracker. Kept as module-level strings so call
# sites don't typo-drift.
_LAT_D2H = "d2h"  # producer: device -> host (index_select + copy_)
_LAT_WIRE = "wire"  # consumer: network PULL round-trip
_LAT_UNPACK = "unpack"  # consumer: rank0 scatter payload -> shm
_LAT_H2D = "h2d"  # consumer: host -> device (to(device))
_LAT_INSERT = "insert"  # consumer: index_put_ + synchronize on device
_LAT_STAGE = "stage"  # producer: total staging (D2H across all layers+sync)
_LAT_SCATTER = "scatter"  # consumer: total scatter (H2D+insert across layers)

# Intra-host IPC op tags (rank-0 coordinator <-> other ranks).
_IPC_HELLO = b"HELLO"  # worker -> coord: (rank,)
_IPC_STAGE_NOTIFY = b"STAGE_N"  # coord -> worker: (uuid, slot_idx, num_blocks)
_IPC_STAGE_DONE = b"STAGE_D"  # worker -> coord: (uuid, rank)
_IPC_LOAD_NOTIFY = b"LOAD_N"  # coord -> worker: (uuid, slot_idx, num_blocks, local_blocks)
_IPC_LOAD_SKIP = b"LOAD_S"  # coord -> worker: (uuid,) -- drain w/ no scatter (cache hit)
_IPC_COPY_DONE = b"COPY_D"  # worker -> coord: (uuid, rank)
_IPC_DROP = b"DROP"  # coord -> worker: (uuid,)  -- failure/timeout


@dataclass
class SendMeta:
    uuid: int
    local_block_ids: list[int]
    expiration_time: float


@dataclass
class LoadMeta:
    uuid: int
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str | list[str]
    remote_port: int | list[int]


@dataclass
class TPUConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, SendMeta] = field(default_factory=dict)
    reqs_to_load: dict[ReqId, LoadMeta] = field(default_factory=dict)


# ---- Rank-0 bookkeeping (TP>1 coordinator-mode only) -------------------
@dataclass
class _CoordSendEntry:
    req_id: ReqId
    slot_idx: int
    num_blocks: int
    expiration_time: float
    # Ranks that have sent STAGE_DONE. Once |staged| == tp_size the slot is
    # ready to serve external PULL requests.
    staged: set[int] = field(default_factory=set)
    stage_complete: threading.Event = field(default_factory=threading.Event)
    # Flipped True when the remote side has acked the pull via NOTIFY, so
    # get_finished() can return this req as done_sending.
    pull_acked: bool = False


@dataclass
class _CoordRecvEntry:
    req_id: ReqId
    uuid: int
    slot_idx: int
    num_blocks: int
    local_blocks: list[int]
    remote_blocks: list[int]
    remote_host: str
    remote_port: int
    # Set by the async pull thread once data has been written to shm.
    load_complete: threading.Event = field(default_factory=threading.Event)
    # Ranks that have sent COPY_DONE. Once |copied| == tp_size we can notify
    # the remote producer and release the slot.
    copied: set[int] = field(default_factory=set)
    # Did the pull actually succeed? If False, we skip done_recving.
    pull_ok: bool = False
    # Have we already surfaced this req in get_finished()? Prevents duplicate
    # reporting across scheduler ticks while we wait for per-rank copies.
    reported_done: bool = False


class _LatencyTracker:
    """Running per-phase latency stats + periodic summary log.

    Phases are arbitrary string keys (see _LAT_* module constants). Every
    record() updates n/total/min/max under a lock; the first record() after
    log_interval seconds triggers a summary line. Per-sample lines are up
    to the caller (they often carry req-specific context we'd lose here)."""

    def __init__(self, who: str, log_interval_s: float = 30.0):
        self._who = who
        self._log_interval_s = max(0.0, log_interval_s)
        self._lock = threading.Lock()
        # phase -> [n, total_ms, min_ms, max_ms]
        self._stats: dict[str, list[float]] = {}
        self._last_log = time.perf_counter()

    def record(self, phase: str, duration_ms: float) -> None:
        with self._lock:
            row = self._stats.get(phase)
            if row is None:
                self._stats[phase] = [
                    1.0, duration_ms, duration_ms, duration_ms
                ]
            else:
                row[0] += 1
                row[1] += duration_ms
                if duration_ms < row[2]:
                    row[2] = duration_ms
                if duration_ms > row[3]:
                    row[3] = duration_ms
            if self._log_interval_s <= 0:
                return
            now = time.perf_counter()
            if now - self._last_log < self._log_interval_s:
                return
            parts = []
            for name, (n, total, mn, mx) in self._stats.items():
                avg = total / n if n else 0.0
                parts.append(f"{name}: n={int(n)} avg={avg:.1f}ms "
                             f"min={mn:.1f}ms max={mx:.1f}ms")
            self._last_log = now
        logger.info("%s latency summary | %s", self._who, " | ".join(parts))


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


class TPUConnectorWorker:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.runner: Optional[TPUModelRunner] = None
        self.device: Optional[torch.device] = None
        self.multi_host = envs.TPU_MULTIHOST_BACKEND == "ray"
        self.node_id: int = dist_utils.get_node_id()

        self.tp_rank: int = get_tensor_model_parallel_rank()
        self.tp_size: int = get_tensor_model_parallel_world_size()

        self.host_ip = dist_utils.get_host_ip()
        self.kv_transfer_port = int(dist_utils.get_kv_transfer_port())
        self.side_channel_port = int(dist_utils.get_side_channel_port())

        # One ZMQ I/O thread per parallel data socket. Default io_threads=1
        # funnels all tp_size data sockets through a single kernel-I/O
        # thread, which caps aggregate wire throughput at what one thread
        # can push (~10 Gbps) regardless of socket count.
        io_threads = max(1, self.tp_size)
        self.zmq_cxt = zmq.Context(io_threads=io_threads)
        self._stop_event = threading.Event()

        self._lat = _LatencyTracker(
            who=f"TPUConnectorWorker node={self.node_id} rank{self.tp_rank}",
            log_interval_s=dist_utils.get_kv_latency_log_interval())

        self._init_coord_state()

        logger.info(
            "TPUConnectorWorker --> init | ip=%s | kv_transfer_port=%s | "
            "side_channel_port=%s | is_producer=%s | node_id=%s | "
            "tp_rank=%d | tp_size=%d", self.host_ip, self.kv_transfer_port,
            self.side_channel_port, self.is_producer, self.node_id,
            self.tp_rank, self.tp_size)

    def _init_coord_state(self) -> None:
        """Minimum state needed on both rank-0 and the other ranks. The
        heavy stuff (SHM pool, ZMQ listeners, IPC endpoints) is allocated
        in register_runner once we know the KV cache shapes."""
        # Parallel TCP stream count. Auto -> tp_size (one channel per rank
        # shard). Clamp to [1, tp_size]. Ports used are
        # base..base+n_channels-1, and a rank's payload rides channel
        # `rank % n_channels`.
        n_cfg = dist_utils.get_transfer_channel_number()
        n_channels = self.tp_size if n_cfg <= 0 else min(n_cfg, self.tp_size)
        self._n_channels: int = max(1, n_channels)
        self._kv_transfer_ports: list[int] = [
            self.kv_transfer_port + i for i in range(self._n_channels)
        ]

        # Multi-layer batched HBM->HBM scatter kernel (kv_scatter): one
        # dispatch across all N layers instead of N per-layer
        # ``cache.index_put_`` calls under lazy torch_tpu. On by default
        # when the smoke test passes; falls back to ``index_put_`` on
        # smoke-test or compile failure.
        self._kv_scatter_enabled: bool = False

        self._coord_pool: Optional[HostKVShmPool] = None
        self._coord_ipc_sock: Optional[zmq.Socket] = None
        self._coord_threads: list[threading.Thread] = []
        # Outbound IPC queue: the IPC loop owns the socket and drains this
        # queue each iteration, so send_multipart() is only ever called from
        # one thread (ZMQ sockets are not thread-safe).
        self._coord_ipc_out: queue.Queue = queue.Queue()
        # Reusable executor for both producer staging and consumer pulling.
        self._coord_executor = ThreadPoolExecutor(max_workers=128)

        # Rank-0 authoritative state. Non-zero ranks leave these empty and
        # return empty sets from get_finished; the scheduler unions across
        # workers, so rank 0's report suffices.
        #
        # Producer side: uuid -> _CoordSendEntry.
        self._coord_send: dict[int, _CoordSendEntry] = {}
        # done_sending set produced by the expiration sweeper, consumed by
        # the next get_finished() call.
        self._coord_done_sending: set[ReqId] = set()
        # done_recving set for reqs whose pull failed or couldn't be
        # admitted (shm pool exhausted, scatter exception on rank 0,
        # pull timeout). Without surfacing these, the vLLM scheduler
        # keeps waiting for a done_recving that never arrives -- every
        # failure leaks a request and eventually hangs the engine.
        # We always report them so the scheduler can advance; the KV
        # contents for the affected req are zero/stale, but that's
        # strictly better than a hang.
        self._coord_done_recving: set[ReqId] = set()
        # Consumer side: uuid -> _CoordRecvEntry.
        self._coord_recv: dict[int, _CoordRecvEntry] = {}

        # Identity of each rank's DEALER, keyed by rank, populated by HELLO.
        self._coord_rank_to_identity: dict[int, bytes] = {}
        # Signaled on rank 0 once every non-zero rank has registered via
        # HELLO. We gate the first STAGE_NOTIFY/LOAD_NOTIFY on this to avoid
        # a startup race where a broadcast would miss a slow-joining rank.
        # At tp_size=1 there are no other ranks to wait for; set the event
        # eagerly so the first request doesn't eat a 5s false-positive
        # timeout in _coord_process_send_load.
        self._coord_workers_ready: threading.Event = threading.Event()
        if self.tp_size == 1:
            self._coord_workers_ready.set()

        # Ranks != 0: state for work the coordinator has told us to do.
        # uuid -> (slot_idx, num_blocks, block_ids) -- producer: coord has
        # broadcast a slot assignment; the next process_send_load on this
        # rank will stage inline.
        self._worker_pending_stage: dict[int, tuple[int, int, list[int]]] = {}
        self._worker_pending_stage_cv = threading.Condition()
        # uuid -> (slot_idx, num_blocks, local_blocks) -- consumer: after
        # coord has filled the slot, ranks scatter from shm on next step.
        self._worker_pending_load: dict[int, tuple[int, int, list[int]]] = {}
        self._worker_pending_load_cv = threading.Condition()

        # Lock around the rank-0 bookkeeping dicts above.
        self._coord_lock = threading.Lock()

    def __del__(self):
        self._stop_event.set()
        self._coord_teardown()
        if hasattr(self, "zmq_cxt"):
            self.zmq_cxt.destroy(linger=0)

    # ---- Bring-up -------------------------------------------------------
    def register_runner(self, runner: TPUModelRunner):
        self.runner = runner
        self.device = runner.device

        kv_caches = runner.kv_caches
        assert len(kv_caches) > 0, (
            "register_runner called before kv_caches were allocated")
        kv_layer = kv_caches[0]
        self.num_layers = len(kv_caches)
        self.shape = list(kv_layer.shape)
        self.dtype = kv_layer.dtype
        logger.info(
            "TPUConnectorWorker --> register_runner | node_id=%s | "
            "num_layers=%d | layer_shape=%s | dtype=%s | device=%s",
            self.node_id, self.num_layers, self.shape, self.dtype, self.device)
        self._coord_setup()
        self._maybe_enable_kv_scatter()

        if dist_utils.get_kv_warmup_enabled():
            try:
                self._warmup_kv_ops()
            except Exception as e:
                logger.exception(
                    "TPUConnectorWorker %s rank%d --> warmup failed "
                    "(continuing; first real request will pay the compile "
                    "cost): %s", self.node_id, self.tp_rank, e)

    def _warmup_block_sizes(self) -> list[int]:
        """Block count to pre-compile: the worst-case max_blocks only."""
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        max_blocks = (max_model_len + block_size - 1) // block_size
        return [max_blocks]

    def _warmup_kv_ops(self) -> None:
        """Pre-compile the XLA executables for the KV gather/scatter hot
        path so the first real request doesn't block on a long compile and
        time out the PULL round-trip on the other side.

        Producer role pre-compiles the D2H path (`index_select` + copy into
        a host staging tensor). Consumer role pre-compiles the H2D +
        `index_put_` path. Both are done per-layer, matching runtime."""
        sizes = self._warmup_block_sizes()
        logger.info(
            "TPUConnectorWorker %s rank%d --> warmup starting | role=%s | "
            "num_layers=%d | block_sizes=%s", self.node_id, self.tp_rank,
            "producer" if self.is_producer else "consumer", self.num_layers,
            sizes)
        start = time.perf_counter()
        for n in sizes:
            if n <= 0:
                continue
            t0 = time.perf_counter()
            self._warmup_coord_once(n)
            logger.info(
                "TPUConnectorWorker %s rank%d --> warmup num_blocks=%d "
                "took %.2fs", self.node_id, self.tp_rank, n,
                time.perf_counter() - t0)
        logger.info("TPUConnectorWorker %s rank%d --> warmup done in %.2fs",
                    self.node_id, self.tp_rank,
                    time.perf_counter() - start)

    def _warmup_coord_once(self, num_blocks: int) -> None:
        """TP>1 coord path: warmup using slot 0 of the shm pool. No traffic
        exists yet at register_runner time, so touching slot 0 is safe.

        Mirrors _coord_stage_shard / _coord_scatter_shard exactly so the
        compiled executable cache hits on first real request."""
        from torch_tpu._internal.sync import sync as tpu_sync
        kv_caches = self.runner.kv_caches
        indices = torch.arange(num_blocks,
                               dtype=torch.int64,
                               device=self.device)
        slot_idx = 0
        if self.is_producer:
            for layer_idx, cache in enumerate(kv_caches):
                src_shard = torch.index_select(cache, 0, indices)
                cpu_shard = src_shard.detach().to("cpu")
                dest_view = self._coord_pool.layer_view(
                    slot_idx, self.tp_rank, layer_idx, num_blocks)
                dest_view.copy_(cpu_shard)
        else:
            use_kv_scatter = self._kv_scatter_enabled
            if use_kv_scatter:
                dest_blocks_dev = torch.tensor(list(range(num_blocks)),
                                               dtype=torch.int32,
                                               device=self.device)
                device_shards = []
                for layer_idx in range(len(kv_caches)):
                    src_view = self._coord_pool.layer_view(
                        slot_idx, self.tp_rank, layer_idx, num_blocks)
                    device_shards.append(src_view.to(self.device))
                new_caches = kv_scatter.multi_layer_scatter_into(
                    device_shards, list(kv_caches), dest_blocks_dev)
                for c in new_caches:
                    tpu_sync.synchronize(c)
                for i, c in enumerate(new_caches):
                    kv_caches[i] = c
            else:
                for layer_idx, cache in enumerate(kv_caches):
                    src_view = self._coord_pool.layer_view(
                        slot_idx, self.tp_rank, layer_idx, num_blocks)
                    device_shard = src_view.to(self.device)
                    cache.index_put_((indices, ), device_shard)
                    tpu_sync.synchronize(cache)

    # ---- Main dispatch driven by start_load_kv --------------------------
    def process_send_load(self, metadata: TPUConnectorMetadata):
        return self._coord_process_send_load(metadata)

    def _resolve_remote_host_port(self, req_meta: LoadMeta) -> tuple[str, int]:
        if isinstance(req_meta.remote_host, list):
            assert len(req_meta.remote_host) == len(req_meta.remote_port)
            host = req_meta.remote_host[self.node_id]
            port = req_meta.remote_port[self.node_id]
        else:
            host = req_meta.remote_host
            port = req_meta.remote_port
        return host, int(port)

    # ---- Polled by the runner each step --------------------------------
    def get_finished(self) -> tuple[set[str], set[str]]:
        return self._coord_get_finished()

    # ====================================================================
    #  Coordinator-mode internals
    # ====================================================================
    # Rank 0 owns all external ZMQ endpoints (data + side-channel) and a
    # SharedMemory slot pool; other ranks stage/scatter into/from their
    # slice of a slot. Control messages between rank 0 and the rest flow
    # over ZMQ ROUTER/DEALER on ipc:// .

    def _coord_setup(self) -> None:
        """Allocate the shm pool, bind ZMQ endpoints, spawn threads. Called
        from register_runner() once kv_caches are known."""
        kv_layer = self.runner.kv_caches[0]
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        max_blocks = (max_model_len + block_size - 1) // block_size

        # Shard shape is what one rank holds for one layer at max block count.
        # kv_layer is already this rank's shard, so kv_layer.shape[0] is
        # total-blocks-in-cache (>> max_blocks per request); we only need
        # the per-layer spatial dims.
        layer_shard_shape = (max_blocks, ) + tuple(kv_layer.shape[1:])

        # Derive slot count from the user's GB budget. Each slot holds one
        # request's worth of KV across all ranks, so the budget scales
        # with max_model_len * num_layers * kv_heads * head_dim * dtype;
        # 128 GB default comfortably covers hundreds of slots for a typical
        # 70B-class model at TP=8 / max_model_len=8192.
        dtype_bytes = torch.tensor([], dtype=kv_layer.dtype).element_size()
        per_layer_bytes = dtype_bytes
        for d in layer_shard_shape:
            per_layer_bytes *= d
        per_slot_bytes = self.tp_size * self.num_layers * per_layer_bytes
        budget_bytes = int(dist_utils.get_kv_shm_pool_gb() * (1024**3))
        num_slots = max(1, budget_bytes // per_slot_bytes)
        logger.info(
            "TPUConnectorWorker %s rank%d --> shm pool budget=%.2fGB "
            "per_slot=%.2fMB -> num_slots=%d", self.node_id, self.tp_rank,
            budget_bytes / (1024**3), per_slot_bytes / (1024**2), num_slots)

        self._coord_pool_spec = PoolSpec(
            num_slots=num_slots,
            tp_size=self.tp_size,
            num_layers=self.num_layers,
            max_blocks=max_blocks,
            layer_shard_shape=layer_shard_shape,
            dtype=kv_layer.dtype,
        )

        shm_name = dist_utils.get_shm_name(self.node_id)
        ipc_path = dist_utils.get_ipc_socket_path(self.node_id)

        if self.tp_rank == 0:
            self._coord_pool = HostKVShmPool.create(self._coord_pool_spec,
                                                    shm_name)
            self._coord_setup_rank0(ipc_path)
        else:
            self._coord_pool = self._coord_attach_shm_with_retry(
                self._coord_pool_spec, shm_name)
            self._coord_setup_worker(ipc_path)

    def _coord_attach_shm_with_retry(self, spec: PoolSpec,
                                     name: str) -> HostKVShmPool:
        """Non-rank-0 ranks may race ahead of rank 0. Poll the shm name a
        few seconds before giving up."""
        deadline = time.perf_counter() + 30.0
        last_err: Optional[Exception] = None
        while time.perf_counter() < deadline:
            try:
                return HostKVShmPool.attach(spec, name)
            except FileNotFoundError as e:
                last_err = e
                time.sleep(0.1)
        raise RuntimeError(
            f"rank {self.tp_rank}: could not attach to shm {name}: {last_err}")

    # ---- Rank-0 bring-up -----------------------------------------------
    def _coord_setup_rank0(self, ipc_path: str) -> None:
        # IPC ROUTER: workers DEAL in.
        ipc = self.zmq_cxt.socket(zmq.ROUTER)
        ipc.setsockopt(zmq.LINGER, 0)
        # Remove any stale endpoint from a prior crashed run.
        _try_remove_ipc_endpoint(ipc_path)
        ipc.bind(ipc_path)
        self._coord_ipc_sock = ipc
        logger.info("TPUConnectorWorker %s rank0 --> IPC listening on %s",
                    self.node_id, ipc_path)

        # IPC listener thread.
        t_ipc = threading.Thread(target=self._coord_rank0_ipc_loop,
                                 name="tpu_conn_ipc",
                                 daemon=True)
        t_ipc.start()
        self._coord_threads.append(t_ipc)

        # Expiration sweeper for pending sends.
        t_exp = threading.Thread(target=self._coord_rank0_expire_loop,
                                 name="tpu_conn_expire",
                                 daemon=True)
        t_exp.start()
        self._coord_threads.append(t_exp)

        # External endpoints are producer-only.
        if self.is_producer:
            for ch in range(self._n_channels):
                t_data = threading.Thread(
                    target=self._coord_rank0_external_data_loop,
                    args=(ch, ),
                    name=f"tpu_conn_data_ch{ch}",
                    daemon=True)
                t_data.start()
                self._coord_threads.append(t_data)

            t_notif = threading.Thread(
                target=self._coord_rank0_external_notif_loop,
                name="tpu_conn_notif",
                daemon=True)
            t_notif.start()
            self._coord_threads.append(t_notif)
        else:
            # Consumer-side: outbound notify-done DEALERs are shared across
            # requests; guard the dict (and each send) behind a lock since
            # it's touched by the main-thread drain path, the IPC listener
            # thread, and async pull threads. Pull sockets are opened per-
            # request, so no caching needed for them.
            self._coord_notif_sockets: dict[str, zmq.Socket] = {}
            self._coord_sockets_lock = threading.Lock()

    # ---- Non-zero rank bring-up ----------------------------------------
    def _coord_setup_worker(self, ipc_path: str) -> None:
        sock = self.zmq_cxt.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        # Identity encodes rank so the coordinator can address us explicitly.
        ident = f"tpu_rank_{self.tp_rank}".encode("utf-8")
        sock.setsockopt(zmq.IDENTITY, ident)
        sock.connect(ipc_path)
        self._coord_ipc_sock = sock
        # Announce ourselves; messages buffer on DEALER until coord ROUTER
        # reads them, so this is safe even if coord isn't up yet.
        self._coord_send_ipc(_IPC_HELLO, (self.tp_rank, ))
        logger.info("TPUConnectorWorker %s rank%d --> IPC connected to %s",
                    self.node_id, self.tp_rank, ipc_path)

        t = threading.Thread(target=self._coord_worker_ipc_loop,
                             name=f"tpu_conn_worker_ipc_{self.tp_rank}",
                             daemon=True)
        t.start()
        self._coord_threads.append(t)

    # ---- IPC envelope helpers ------------------------------------------
    # All IPC send paths enqueue; the IPC loop thread is the only one that
    # actually calls send_multipart. This keeps the single ZMQ socket off
    # of multiple threads.

    def _coord_send_ipc(self, tag: bytes, payload) -> None:
        """Worker (non-zero rank) -> coordinator. Enqueues a DEALER frame."""
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        # DEALER frames are [tag, data]; ROUTER side reassembles identity
        # from the socket itself.
        self._coord_ipc_out.put((None, tag, data))

    def _coord_broadcast(self, tag: bytes, payload) -> None:
        """Coordinator -> every known non-zero rank. Enqueues one frame per
        rank, addressed to the rank's DEALER identity."""
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        with self._coord_lock:
            identities = list(self._coord_rank_to_identity.values())
        for ident in identities:
            self._coord_ipc_out.put((ident, tag, data))

    def _coord_ipc_drain_outbound(self) -> None:
        """Pop any pending outbound messages and send them. Called from
        inside the owning IPC loop; must not be called from other threads."""
        sock = self._coord_ipc_sock
        try:
            while True:
                ident, tag, data = self._coord_ipc_out.get_nowait()
                try:
                    if ident is None:
                        # DEALER: [tag, data]
                        sock.send_multipart([tag, data])
                    else:
                        # ROUTER: [identity, tag, data]
                        sock.send_multipart([ident, tag, data])
                except zmq.ZMQError as e:
                    logger.warning(
                        "TPUConnectorWorker %s --> IPC send tag=%s failed: %s",
                        self.node_id, tag, e)
        except queue.Empty:
            return

    # ---- Coord teardown -------------------------------------------------
    def _coord_teardown(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_coord_executor"):
            self._coord_executor.shutdown(wait=False)
        for t in getattr(self, "_coord_threads", []):
            t.join(timeout=2)
        if getattr(self, "_coord_pool", None) is not None:
            self._coord_pool.close()

    # =========================================================
    # Process send/load dispatcher (coord mode)
    # =========================================================
    def _coord_process_send_load(self, metadata: TPUConnectorMetadata) -> None:
        # Rank 0 is the only one that knows about remote endpoints and owns
        # the free list; it kicks off both staging (producer) and pulling
        # (consumer) here. Every rank, including rank 0, is also responsible
        # for its own H2D scatter when a pulled req is in the "drain" phase
        # (remote_block_ids is None).
        #
        # Staging runs inline on the main thread on every rank (rank 0 while
        # broadcasting, others after waiting on STAGE_NOTIFY). Running stage
        # on a background executor is unsafe under TP>1: on non-zero ranks,
        # by the time the executor got to `.to("cpu")`, the main thread had
        # already queued the next model_forward (first-time decode compile
        # under multi-rank collectives), so the D2H sync blocked behind it
        # and the PULL timed out. Inline stage drains each rank's XLA queue
        # before model_forward adds anything to it.

        if self.tp_rank == 0:
            if metadata.reqs_to_send or metadata.reqs_to_load:
                # Only gate on worker registration when we actually need to
                # broadcast. If all HELLO's are still in flight, wait briefly.
                if not self._coord_workers_ready.wait(timeout=5.0):
                    logger.warning(
                        "TPUConnectorWorker rank0 --> still waiting on "
                        "HELLOs: %d/%d registered",
                        len(self._coord_rank_to_identity), self.tp_size - 1)
                for req_id, req_meta in metadata.reqs_to_send.items():
                    self._coord_rank0_handle_new_send(req_id, req_meta)
                for req_id, req_meta in metadata.reqs_to_load.items():
                    self._coord_rank0_handle_new_load(req_id, req_meta)
        else:
            # Non-zero ranks: stage inline for each reqs_to_send in this
            # metadata. Rank 0 broadcasts STAGE_NOTIFY (with slot_idx); we
            # block here until it arrives (via IPC thread), run stage in
            # the main thread, and return before model_forward executes.
            for req_id, req_meta in metadata.reqs_to_send.items():
                self._coord_worker_stage_inline(req_meta.uuid, req_id)

        # Drain step (all ranks): scheduler is asking us to scatter reqs for
        # which the pull has already finished.
        for req_id, req_meta in metadata.reqs_to_load.items():
            if req_meta.remote_block_ids is None:
                self._coord_drain_scatter(req_id, req_meta)

    # =========================================================
    # Rank 0 producer path: stage + serve PULLs
    # =========================================================
    def _coord_rank0_handle_new_send(self, req_id: ReqId,
                                     req_meta: SendMeta) -> None:
        """Allocate a slot, broadcast the assignment to non-zero ranks, and
        stage rank 0's own shard inline on the main thread.

        The inline stage is deliberate: see the note in
        _coord_process_send_load. Broadcasting first (via the IPC-out queue,
        drained by the IPC thread every ~50ms) lets non-zero ranks begin
        their own inline stage in parallel with us."""
        assert self.is_producer
        num_blocks = len(req_meta.local_block_ids)
        logger.info(
            "TPUConnectorWorker rank0 --> handle_new_send req_id=%s uuid=%s "
            "blocks=%d", req_id, req_meta.uuid, num_blocks)
        try:
            slot_idx = self._coord_pool.acquire_slot(timeout=0.0)
        except Exception:
            # Pool full. Drop request; scheduler will surface a failure.
            with self._coord_lock:
                send_holders = [
                    f"uuid={u} req={e.req_id} "
                    f"acked={e.pull_acked} "
                    f"exp_in={max(0.0, e.expiration_time - time.perf_counter()):.1f}s"
                    for u, e in self._coord_send.items()
                ]
                recv_holders = [
                    f"uuid={u} req={e.req_id}"
                    for u, e in self._coord_recv.items()
                ]
                self._coord_done_sending.add(req_id)
            logger.error(
                "TPUConnectorWorker rank0 --> shm pool exhausted for "
                "req_id=%s (producer). send_holders=%s recv_holders=%s",
                req_id, send_holders, recv_holders)
            return

        entry = _CoordSendEntry(
            req_id=req_id,
            slot_idx=slot_idx,
            num_blocks=num_blocks,
            expiration_time=req_meta.expiration_time,
        )
        with self._coord_lock:
            self._coord_send[req_meta.uuid] = entry

        # Enqueue broadcast first so other ranks can start their own stage
        # while we run ours.
        self._coord_broadcast(
            _IPC_STAGE_NOTIFY,
            (req_meta.uuid, slot_idx, num_blocks, req_meta.local_block_ids))
        logger.info(
            "TPUConnectorWorker rank0 --> handle_new_send broadcast uuid=%s "
            "slot=%d to %d non-zero ranks", req_meta.uuid, slot_idx,
            len(self._coord_rank_to_identity))

        # Rank 0 stages its own shard inline on the main thread. The XLA
        # queue is empty at this point (scheduler just handed us work; no
        # model_forward yet), so .to("cpu") materializes quickly.
        try:
            self._coord_stage_shard(slot_idx, num_blocks,
                                    req_meta.local_block_ids)
            with self._coord_lock:
                e2 = self._coord_send.get(req_meta.uuid)
                if e2 is None:
                    return
                e2.staged.add(0)
                if len(e2.staged) == self.tp_size:
                    e2.stage_complete.set()
        except Exception as ex:
            logger.exception(
                "TPUConnectorWorker rank0 --> stage_self uuid=%s failed: %s",
                req_meta.uuid, ex)

    def _coord_worker_stage_inline(self,
                                   uuid: int,
                                   req_id: ReqId,
                                   timeout: float = 10.0) -> None:
        """Non-zero rank, main-thread path: wait for STAGE_NOTIFY to arrive
        in _worker_pending_stage (populated by the IPC listener thread),
        then run stage_shard synchronously. Must happen before the main
        thread enters model_forward, otherwise .to("cpu") will block behind
        the first-time decode compile."""
        logger.info(
            "TPUConnectorWorker rank%d --> stage_inline waiting for "
            "STAGE_NOTIFY req_id=%s uuid=%s", self.tp_rank, req_id, uuid)
        deadline = time.perf_counter() + timeout
        info: Optional[tuple[int, int, list[int]]] = None
        with self._worker_pending_stage_cv:
            while uuid not in self._worker_pending_stage:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    logger.warning(
                        "TPUConnectorWorker rank%d --> stage_inline timeout "
                        "uuid=%s (STAGE_NOTIFY never arrived)", self.tp_rank,
                        uuid)
                    return
                self._worker_pending_stage_cv.wait(timeout=rem)
            info = self._worker_pending_stage.pop(uuid)
        slot_idx, num_blocks, block_ids = info
        try:
            self._coord_stage_shard(slot_idx, num_blocks, block_ids)
        except Exception as e:
            logger.exception(
                "TPUConnectorWorker rank%d --> stage_inline uuid=%s failed: %s",
                self.tp_rank, uuid, e)
        # ACK regardless so rank 0's stage_complete can fire.
        self._coord_send_ipc(_IPC_STAGE_DONE, (uuid, self.tp_rank))

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

    def _coord_stage_shard(self, slot_idx: int, num_blocks: int,
                           block_ids: list[int]) -> None:
        """D2H this rank's shard of the selected blocks into the shm slot.

        We use `.to("cpu")` per layer rather than a raw
        `dest_view.copy_(lazy_src)` onto shm: the latter has been observed
        to accumulate all N layers into one monolithic XLA graph
        (synchronize on a CPU tensor is a no-op), which either takes
        >30s to compile or never flushes -- producing a PULL timeout.
        `.to("cpu")` is the canonical torch_tpu materialization path and
        flushes each layer's lazy gather+copy independently."""
        logger.info(
            "TPUConnectorWorker %s rank%d --> stage_shard entering "
            "slot=%d blocks=%d layers=%d", self.node_id, self.tp_rank,
            slot_idx, num_blocks, len(self.runner.kv_caches))
        indices = torch.tensor(block_ids,
                               dtype=torch.int64,
                               device=self.device)
        kv_caches = self.runner.kv_caches
        stage_t0 = time.perf_counter()
        first_layer_ms = 0.0
        for layer_idx, cache in enumerate(kv_caches):
            layer_t0 = time.perf_counter()
            src_shard = torch.index_select(cache, 0, indices)
            # Force materialization to CPU (blocking in torch_tpu),
            # then memcpy into shm via plain CPU->CPU copy_.
            cpu_shard = src_shard.detach().to("cpu")
            dest_view = self._coord_pool.layer_view(slot_idx, self.tp_rank,
                                                    layer_idx, num_blocks)
            dest_view.copy_(cpu_shard)
            if layer_idx == 0:
                first_layer_ms = (time.perf_counter() - layer_t0) * 1000.0
                logger.info(
                    "TPUConnectorWorker %s rank%d --> stage_shard layer 0 "
                    "done in %.2fms (slot=%d blocks=%d)", self.node_id,
                    self.tp_rank, first_layer_ms, slot_idx, num_blocks)
        total_ms = (time.perf_counter() - stage_t0) * 1000.0
        self._lat.record(_LAT_D2H, total_ms)
        self._lat.record(_LAT_STAGE, total_ms)
        logger.info(
            "TPUConnectorWorker %s rank%d --> stage slot=%d blocks=%d "
            "layers=%d d2h=%.2fms (first-layer=%.2fms)", self.node_id,
            self.tp_rank, slot_idx, num_blocks, len(kv_caches), total_ms,
            first_layer_ms)

    # =========================================================
    # Rank 0 consumer path: pull, write shm, broadcast LOAD_NOTIFY
    # =========================================================
    def _coord_rank0_handle_new_load(self, req_id: ReqId,
                                     req_meta: LoadMeta) -> None:
        assert not self.is_producer
        if req_meta.remote_block_ids is None:
            # Either a full cache hit or a drain-for-completed-pull. Only
            # the former requires telling other ranks to skip; the drain
            # case means _worker_pending_load already has the real entry
            # from an earlier LOAD_NOTIFY broadcast.
            with self._coord_lock:
                is_cache_hit = req_meta.uuid not in self._coord_recv
            if is_cache_hit:
                self._coord_broadcast(_IPC_LOAD_SKIP, (req_meta.uuid, ))
            return

        num_blocks = len(req_meta.local_block_ids)
        assert num_blocks == len(req_meta.remote_block_ids)
        # Preemption dedup: vLLM's scheduler re-emits update_state_after_alloc
        # for requests that were preempted and rescheduled. The LoadMeta
        # carries the same uuid (assigned by the producer) but new
        # local_block_ids. If we treat it as a fresh load we acquire a
        # second shm slot, submit a duplicate _coord_rank0_pull thread,
        # and orphan the first slot when we overwrite _coord_recv[uuid]
        # -- eventually exhausting the pool and hanging the scheduler.
        # Detect the re-emit by uuid and just update local_blocks in
        # place so the in-flight pull targets the new block destinations.
        with self._coord_lock:
            existing = self._coord_recv.get(req_meta.uuid)
            if existing is not None:
                pull_in_flight = not existing.load_complete.is_set()
                existing.local_blocks = req_meta.local_block_ids
                existing.num_blocks = num_blocks
                existing.req_id = req_id
                existing.remote_blocks = req_meta.remote_block_ids
        if existing is not None:
            if pull_in_flight:
                # LOAD_NOTIFY hasn't been broadcast yet; the pull thread
                # will read the updated local_blocks when it publishes.
                logger.warning(
                    "TPUConnectorWorker rank0 --> dedup preempted load "
                    "req_id=%s uuid=%s (pull in flight, new blocks=%d)",
                    req_id, req_meta.uuid, num_blocks)
            else:
                # Pull already completed and LOAD_NOTIFY fired with the
                # old blocks. Re-broadcast so workers' cached
                # _worker_pending_load reflects the new destinations
                # before they drain. There's a small race if a worker
                # already popped the old entry, but scatter under the
                # correct blocks is still the right intent.
                logger.warning(
                    "TPUConnectorWorker rank0 --> dedup preempted load "
                    "req_id=%s uuid=%s (pull already complete, "
                    "rebroadcasting LOAD_NOTIFY with new blocks=%d)", req_id,
                    req_meta.uuid, num_blocks)
                self._coord_broadcast(
                    _IPC_LOAD_NOTIFY,
                    (existing.uuid, existing.slot_idx, existing.num_blocks,
                     existing.local_blocks))
            return

        try:
            slot_idx = self._coord_pool.acquire_slot(timeout=0.0)
        except Exception:
            with self._coord_lock:
                recv_holders = [
                    f"uuid={u} req={e.req_id} "
                    f"pull_ok={e.pull_ok} "
                    f"load={e.load_complete.is_set()} "
                    f"reported={e.reported_done} "
                    f"copied={sorted(e.copied)}"
                    for u, e in self._coord_recv.items()
                ]
                send_holders = [
                    f"uuid={u} req={e.req_id} "
                    f"acked={e.pull_acked}"
                    for u, e in self._coord_send.items()
                ]
                self._coord_done_recving.add(req_id)
            logger.error(
                "TPUConnectorWorker rank0 --> shm pool exhausted for "
                "req_id=%s; surfacing done_recving so scheduler advances "
                "(expect wrong output for this req). recv_holders=%s "
                "send_holders=%s", req_id, recv_holders, send_holders)
            return

        remote_host, remote_port = self._resolve_remote_host_port(req_meta)
        entry = _CoordRecvEntry(
            req_id=req_id,
            uuid=req_meta.uuid,
            slot_idx=slot_idx,
            num_blocks=num_blocks,
            local_blocks=req_meta.local_block_ids,
            remote_blocks=req_meta.remote_block_ids,
            remote_host=remote_host,
            remote_port=remote_port,
        )
        with self._coord_lock:
            self._coord_recv[req_meta.uuid] = entry

        self._coord_executor.submit(self._coord_rank0_pull, entry)

    def _coord_rank0_pull(self, entry: _CoordRecvEntry) -> None:
        """Pull the full KV blob (all ranks' shards) from remote coord and
        scatter it into shm. Then broadcast LOAD_NOTIFY and set the
        load_complete event so get_finished() returns done_recving.

        Uses ``self._n_channels`` parallel DEALER sockets (one per thread)
        to N producer ROUTER ports ``base_port + ch``. Each channel carries
        its subset of rank payloads; we reassemble them into a single
        rank->payload map then unpack all shards into shm."""
        n_channels = self._n_channels
        base_port = int(entry.remote_port)
        uuid_bytes = str(entry.uuid).encode("utf-8")
        timeout_s = dist_utils.get_p2p_wait_pull_timeout()
        req_blocks_pickle = pickle.dumps(entry.remote_blocks,
                                         protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "TPUConnectorWorker rank0 --> PULL req_id=%s uuid=%s "
            "from=%s:%s+0..%d blocks=%d n_channels=%d", entry.req_id,
            entry.uuid, entry.remote_host, base_port, n_channels - 1,
            entry.num_blocks, n_channels)

        num_layers = self.num_layers

        def _pull_channel(ch: int):
            """Pull one channel's subset of ranks AND unpack their layer
            frames into shm inline. Returns per-channel (wire_ms,
            unpack_ms, ranks_handled) so the orchestrator can log the
            breakdown but does not need to reassemble anything."""
            port = base_port + ch
            sock_path = make_zmq_path("tcp", entry.remote_host, port)
            sock = make_zmq_socket(ctx=self.zmq_cxt,
                                   path=sock_path,
                                   socket_type=zmq.DEALER,
                                   bind=False)
            try:
                t0 = time.perf_counter()
                sock.send_multipart([_MSG_PULL, uuid_bytes, req_blocks_pickle])
                deadline = t0 + timeout_s
                frames = None
                while time.perf_counter() < deadline:
                    remaining_ms = max(
                        1, int((deadline - time.perf_counter()) * 1000))
                    if sock.poll(timeout=min(remaining_ms, 1000)) != 0:
                        # copy=False returns zmq.Frame objects whose
                        # .buffer is a readonly memoryview into the ZMQ
                        # recv buffer -- one less 256MB copy per layer.
                        frames = sock.recv_multipart(copy=False)
                        break
                if frames is None:
                    raise TimeoutError(
                        f"PULL ch={ch} timed out after {timeout_s}s "
                        f"for req_id={entry.req_id}")
                if bytes(frames[0].buffer) != _MSG_OK:
                    tag = bytes(frames[0].buffer)
                    payload_hint = (bytes(frames[1].buffer)
                                    if len(frames) > 1 else b"")
                    raise RuntimeError(
                        f"PULL ch={ch} failed for req_id={entry.req_id}: "
                        f"[{tag!r}, {payload_hint!r}]")
                t_wire_done = time.perf_counter()

                # Wire format after OK:
                #   frames[1]=uuid echo, frames[2]=n_ranks ascii
                #   ch 0: frames[3]=header, then per-rank blocks
                #   ch >0: per-rank blocks start at frames[3]
                # Per-rank block = (rank_idx, layer0, layer1, ..., layer_{L-1})
                n_ranks = int(bytes(frames[2].buffer).decode("utf-8"))
                idx = 3
                header_frame: Optional[zmq.Frame] = None
                if ch == 0:
                    header_frame = frames[idx]
                    idx += 1

                ranks_handled: list[int] = []
                for _ in range(n_ranks):
                    rank_idx = int(bytes(frames[idx].buffer).decode("utf-8"))
                    idx += 1
                    layer_buffers = [
                        frames[idx + li].buffer for li in range(num_layers)
                    ]
                    idx += num_layers
                    self._coord_pool.unpack_rank_layers(
                        entry.slot_idx, rank_idx, entry.num_blocks,
                        layer_buffers)
                    ranks_handled.append(rank_idx)
                t_unpack_done = time.perf_counter()
                wire_ms = (t_wire_done - t0) * 1000.0
                unpack_ms = (t_unpack_done - t_wire_done) * 1000.0
                return ch, wire_ms, unpack_ms, ranks_handled, header_frame
            finally:
                sock.close(linger=0)

        t_total0 = time.perf_counter()
        try:
            futures = [
                self._coord_executor.submit(_pull_channel, ch)
                for ch in range(n_channels)
            ]
            results: list[tuple[int, float, float, list[int],
                                Optional[zmq.Frame]]] = []
            for fut in futures:
                results.append(fut.result())
            t_done = time.perf_counter()

            # Sanity-check all ranks were delivered exactly once.
            seen: set[int] = set()
            header_frame: Optional[zmq.Frame] = None
            for _ch, _w, _u, ranks, hdr in results:
                if hdr is not None:
                    header_frame = hdr
                for r in ranks:
                    if r in seen:
                        raise RuntimeError(
                            f"PULL req_id={entry.req_id}: rank {r} "
                            f"delivered by multiple channels")
                    seen.add(r)
            if header_frame is None:
                raise RuntimeError(
                    f"PULL req_id={entry.req_id}: channel 0 omitted header")
            if len(seen) != self.tp_size:
                raise RuntimeError(
                    f"PULL req_id={entry.req_id}: got {len(seen)} ranks, "
                    f"expected {self.tp_size}")

            total_ms = (t_done - t_total0) * 1000.0
            # With inline unpack per channel, wire+unpack are already
            # overlapped; the outer total is max over channels of
            # (wire+unpack). Report them separately for continuity
            # with the existing latency tracker.
            max_wire = max(w for _c, w, _u, _r, _h in results)
            max_unpack = max(u for _c, _w, u, _r, _h in results)
            self._lat.record(_LAT_WIRE, max_wire)
            self._lat.record(_LAT_UNPACK, max_unpack)
            per_ch_ms = ",".join(f"{w + u:.1f}"
                                 for _c, w, u, _r, _h in results)
            logger.info(
                "TPUConnectorWorker rank0 --> done PULL req_id=%s uuid=%s "
                "total=%.2fms max_wire=%.2fms max_unpack=%.2fms "
                "per_ch_ms=[%s]", entry.req_id, entry.uuid, total_ms, max_wire,
                max_unpack, per_ch_ms)
            entry.pull_ok = True
        except Exception as e:
            logger.exception(
                "TPUConnectorWorker rank0 --> PULL req_id=%s failed: %s",
                entry.req_id, e)
            entry.pull_ok = False
        finally:
            # Tell workers the slot is ready (or drop it on failure).
            if entry.pull_ok:
                self._coord_broadcast(_IPC_LOAD_NOTIFY,
                                      (entry.uuid, entry.slot_idx,
                                       entry.num_blocks, entry.local_blocks))
            else:
                self._coord_broadcast(_IPC_DROP, (entry.uuid, ))
                # Clean up: failed pulls will never reach COPY_DONE, so
                # free the slot and drop bookkeeping now. Also surface
                # the req_id as done_recving -- otherwise the vLLM
                # scheduler waits forever for completion of a req whose
                # entry we just silently removed.
                with self._coord_lock:
                    self._coord_recv.pop(entry.uuid, None)
                    self._coord_done_recving.add(entry.req_id)
                self._coord_pool.release_slot(entry.slot_idx)
            entry.load_complete.set()

    # =========================================================
    # Drain pass: every rank (including 0) scatters its shard
    # =========================================================
    def _coord_drain_scatter(self, req_id: ReqId, req_meta: LoadMeta) -> None:
        uuid = req_meta.uuid
        if self.tp_rank == 0:
            # Rank 0's state is in _coord_recv; grab slot info from there.
            with self._coord_lock:
                entry = self._coord_recv.get(uuid)
            if entry is None or not entry.pull_ok:
                # Unknown or failed; scheduler will treat this as a cache hit
                # and we notify upstream so P can free its buffer.
                self._coord_rank0_send_notify(req_meta, uuid)
                return
            slot_idx = entry.slot_idx
            num_blocks = entry.num_blocks
            local_blocks = entry.local_blocks
        else:
            info = self._coord_worker_wait_load(uuid)
            if info is None:
                # DROP arrived or timeout. Nothing to do.
                return
            slot_idx, num_blocks, local_blocks = info

        # Scatter our own shard.
        try:
            self._coord_scatter_shard(slot_idx, num_blocks, local_blocks)
        except Exception as e:
            logger.exception(
                "TPUConnectorWorker %s rank%d --> scatter req_id=%s failed: %s",
                self.node_id, self.tp_rank, req_id, e)
            if self.tp_rank != 0:
                # Still ack so coord can free the slot.
                self._coord_send_ipc(_IPC_COPY_DONE, (uuid, self.tp_rank))
            else:
                # Rank 0 must still ack its own scatter, otherwise the
                # all-ranks-copied check in register_copy never fires
                # and the slot leaks even after every other rank succeeds.
                self._coord_rank0_register_copy(uuid, req_meta)
            return

        if self.tp_rank == 0:
            self._coord_rank0_register_copy(uuid, req_meta)
        else:
            self._coord_send_ipc(_IPC_COPY_DONE, (uuid, self.tp_rank))

    def _coord_scatter_shard(self, slot_idx: int, num_blocks: int,
                             local_blocks: list[int]) -> None:
        """H2D + scatter this rank's shard from shm into the kv cache.

        Two paths:
          * Multi-layer kv_scatter (default when its startup smoke test
            passes): ``src_view.to(device)`` per layer (queued lazily),
            then one bridge call that scatters all N layers in a single
            fused kernel with each destination donated+aliased to its
            output.
          * Naive fallback (kv_scatter smoke test failed):
            ``src_view.to(device)`` + ``index_put_`` per layer."""
        if not local_blocks:
            return
        kv_caches = self.runner.kv_caches
        from torch_tpu._internal.sync import sync as tpu_sync
        use_kv_scatter = self._kv_scatter_enabled
        h2d_ms_total = 0.0
        insert_ms_total = 0.0
        if use_kv_scatter:
            # One bridge crossing for all N layers. We still issue N
            # .to(device) calls sequentially (the only way to get shm
            # bytes to HBM under the current torch_tpu bridge), but let
            # torch_tpu queue them up lazily before firing the one
            # kernel call that does all N scatters.
            dest_blocks_dev = torch.tensor(local_blocks,
                                           dtype=torch.int32,
                                           device=self.device)
            prebuilt = kv_scatter.prepare_scatter_args(dest_blocks_dev,
                                                       self.device)
            h2d_t0 = time.perf_counter()
            device_shards = []
            for layer_idx in range(len(kv_caches)):
                src_view = self._coord_pool.layer_view(slot_idx, self.tp_rank,
                                                       layer_idx, num_blocks)
                device_shards.append(src_view.to(self.device))
            h2d_t1 = time.perf_counter()
            new_caches = kv_scatter.multi_layer_scatter_into(
                device_shards,
                list(kv_caches),
                dest_blocks_dev,
                prebuilt_args=prebuilt)
            # Sync every returned cache -- they're independent outputs
            # of the fused kernel and XLA may schedule them in parallel,
            # so syncing only the last would race.
            for c in new_caches:
                tpu_sync.synchronize(c)
            k_wait_t = time.perf_counter()
            for i, c in enumerate(new_caches):
                kv_caches[i] = c
            h2d_ms_total = (h2d_t1 - h2d_t0) * 1000.0
            insert_ms_total = (k_wait_t - h2d_t1) * 1000.0
            path = "kv_scatter"
        else:
            indices = torch.tensor(local_blocks,
                                   dtype=torch.int64,
                                   device=self.device)
            for layer_idx, cache in enumerate(kv_caches):
                src_view = self._coord_pool.layer_view(slot_idx, self.tp_rank,
                                                       layer_idx, num_blocks)
                h_t0 = time.perf_counter()
                device_shard = src_view.to(self.device)
                h_t1 = time.perf_counter()
                cache.index_put_((indices, ), device_shard)
                tpu_sync.synchronize(cache)
                h_t2 = time.perf_counter()
                h2d_ms_total += (h_t1 - h_t0) * 1000.0
                insert_ms_total += (h_t2 - h_t1) * 1000.0
            path = "naive"
        self._lat.record(_LAT_H2D, h2d_ms_total)
        self._lat.record(_LAT_INSERT, insert_ms_total)
        self._lat.record(_LAT_SCATTER, h2d_ms_total + insert_ms_total)
        logger.info(
            "TPUConnectorWorker %s rank%d --> scatter slot=%d blocks=%d "
            "layers=%d h2d=%.2fms insert=%.2fms path=%s", self.node_id,
            self.tp_rank, slot_idx, num_blocks, len(kv_caches), h2d_ms_total,
            insert_ms_total, path)

    def _coord_rank0_register_copy(self, uuid: int,
                                   req_meta: LoadMeta) -> None:
        all_copied = False
        with self._coord_lock:
            entry = self._coord_recv.get(uuid)
            if entry is None:
                return
            entry.copied.add(0)
            if len(entry.copied) == self.tp_size:
                all_copied = True
        if all_copied:
            # Notify remote P and release slot.
            self._coord_rank0_send_notify(req_meta, uuid)
            with self._coord_lock:
                entry = self._coord_recv.pop(uuid, None)
            if entry is not None:
                self._coord_pool.release_slot(entry.slot_idx)

    def _coord_rank0_send_notify(self, req_meta: LoadMeta, uuid: int) -> None:
        remote_host = req_meta.remote_host
        if isinstance(remote_host, list):
            remote_host = remote_host[self.node_id]
        sock_path = make_zmq_path("tcp", remote_host, self.side_channel_port)
        # DEALER is not thread-safe; serialize create+send under the lock.
        with self._coord_sockets_lock:
            sock = self._coord_notif_sockets.get(sock_path)
            if sock is None:
                sock = make_zmq_socket(ctx=self.zmq_cxt,
                                       path=sock_path,
                                       socket_type=zmq.DEALER,
                                       bind=False)
                self._coord_notif_sockets[sock_path] = sock
                logger.info(
                    "TPUConnectorWorker rank0 --> notify channel to %s",
                    sock_path)
            logger.info(
                "TPUConnectorWorker rank0 --> notify pull-done uuid=%s", uuid)
            sock.send_string(str(uuid))

    # =========================================================
    # Non-zero rank: wait for coord messages then act
    # =========================================================
    def _coord_worker_wait_load(
            self,
            uuid: int,
            timeout: float = 1.0) -> Optional[tuple[int, int, list[int]]]:
        """Block until LOAD_NOTIFY or LOAD_SKIP for uuid arrives, or timeout
        expires. Returns (slot_idx, num_blocks, local_blocks) for a scatter,
        or None for skip/timeout. The dict stores None as a sentinel set
        by LOAD_SKIP, so we can tell "skip" apart from "not there yet" by
        presence."""
        deadline = time.perf_counter() + timeout
        with self._worker_pending_load_cv:
            while uuid not in self._worker_pending_load:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    return None
                self._worker_pending_load_cv.wait(timeout=rem)
            info = self._worker_pending_load.pop(uuid)
        return info

    # =========================================================
    # IPC listeners
    # =========================================================
    def _coord_rank0_ipc_loop(self) -> None:
        sock = self._coord_ipc_sock
        while not self._stop_event.is_set():
            # Drain outbound sends from the queue first so notifications
            # don't starve behind inbound polling.
            self._coord_ipc_drain_outbound()
            try:
                # Short poll so outbound drains stay responsive. ZMQ will
                # buffer in-flight messages while we're away.
                if sock.poll(timeout=50) == 0:
                    continue
                frames = sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker rank0 --> IPC recv error: %s", e)
                continue
            # ROUTER gives us [identity, tag, payload].
            if len(frames) != 3:
                logger.warning(
                    "TPUConnectorWorker rank0 --> IPC malformed frames n=%d",
                    len(frames))
                continue
            identity, tag, payload = frames
            try:
                obj = pickle.loads(payload)
            except Exception as e:
                logger.warning(
                    "TPUConnectorWorker rank0 --> bad IPC "
                    "payload tag=%s: %s", tag, e)
                continue
            if tag == _IPC_HELLO:
                (rank, ) = obj
                with self._coord_lock:
                    self._coord_rank_to_identity[rank] = identity
                    # All non-zero ranks registered? Unblock new-req
                    # processing on rank 0.
                    if len(self._coord_rank_to_identity) == self.tp_size - 1:
                        self._coord_workers_ready.set()
                logger.info(
                    "TPUConnectorWorker rank0 --> HELLO from rank=%d "
                    "(%d/%d registered)", rank,
                    len(self._coord_rank_to_identity), self.tp_size - 1)
            elif tag == _IPC_STAGE_DONE:
                uuid, rank = obj
                fired_complete = False
                staged_count = 0
                with self._coord_lock:
                    entry = self._coord_send.get(uuid)
                    if entry is None:
                        continue
                    entry.staged.add(rank)
                    staged_count = len(entry.staged)
                    if staged_count == self.tp_size:
                        entry.stage_complete.set()
                        fired_complete = True
                if fired_complete:
                    logger.info(
                        "TPUConnectorWorker rank0 --> stage_complete uuid=%s "
                        "(all %d ranks staged)", uuid, self.tp_size)
                else:
                    logger.info(
                        "TPUConnectorWorker rank0 --> STAGE_DONE uuid=%s "
                        "rank=%d (%d/%d)", uuid, rank, staged_count,
                        self.tp_size)
            elif tag == _IPC_COPY_DONE:
                uuid, rank = obj
                all_copied = False
                entry_for_notify = None
                with self._coord_lock:
                    entry = self._coord_recv.get(uuid)
                    if entry is None:
                        continue
                    entry.copied.add(rank)
                    if len(entry.copied) == self.tp_size:
                        all_copied = True
                        entry_for_notify = entry
                if all_copied and entry_for_notify is not None:
                    # Synthesize a LoadMeta-like object for the notify path.
                    dummy = LoadMeta(
                        uuid=entry_for_notify.uuid,
                        local_block_ids=entry_for_notify.local_blocks,
                        remote_block_ids=entry_for_notify.remote_blocks,
                        remote_host=entry_for_notify.remote_host,
                        remote_port=entry_for_notify.remote_port,
                    )
                    self._coord_rank0_send_notify(dummy, entry_for_notify.uuid)
                    with self._coord_lock:
                        self._coord_recv.pop(uuid, None)
                    self._coord_pool.release_slot(entry_for_notify.slot_idx)
            else:
                logger.warning(
                    "TPUConnectorWorker rank0 --> unknown IPC tag=%s", tag)

    def _coord_worker_ipc_loop(self) -> None:
        sock = self._coord_ipc_sock
        while not self._stop_event.is_set():
            self._coord_ipc_drain_outbound()
            try:
                if sock.poll(timeout=50) == 0:
                    continue
                frames = sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker rank%d --> IPC recv error: %s",
                    self.tp_rank, e)
                continue
            # DEALER gives us [tag, payload].
            if len(frames) != 2:
                logger.warning(
                    "TPUConnectorWorker rank%d --> IPC malformed frames n=%d",
                    self.tp_rank, len(frames))
                continue
            tag, payload = frames
            try:
                obj = pickle.loads(payload)
            except Exception as e:
                logger.warning(
                    "TPUConnectorWorker rank%d --> bad IPC payload tag=%s: %s",
                    self.tp_rank, tag, e)
                continue
            if tag == _IPC_STAGE_NOTIFY:
                uuid, slot_idx, num_blocks, block_ids = obj
                logger.info(
                    "TPUConnectorWorker rank%d --> STAGE_NOTIFY received "
                    "uuid=%s slot=%d blocks=%d", self.tp_rank, uuid, slot_idx,
                    num_blocks)
                # Hand off to the main thread (waiting in
                # _coord_worker_stage_inline); don't stage in an executor
                # here, otherwise .to("cpu") races with model_forward.
                with self._worker_pending_stage_cv:
                    self._worker_pending_stage[uuid] = (slot_idx, num_blocks,
                                                        block_ids)
                    self._worker_pending_stage_cv.notify_all()
            elif tag == _IPC_LOAD_NOTIFY:
                uuid, slot_idx, num_blocks, local_blocks = obj
                with self._worker_pending_load_cv:
                    self._worker_pending_load[uuid] = (slot_idx, num_blocks,
                                                       local_blocks)
                    self._worker_pending_load_cv.notify_all()
            elif tag == _IPC_LOAD_SKIP:
                (uuid, ) = obj
                # None marks "coord said no scatter needed for this uuid"
                # (cache hit). _coord_worker_wait_load surfaces None
                # immediately without the 200ms-ish timeout wait.
                with self._worker_pending_load_cv:
                    self._worker_pending_load[uuid] = None
                    self._worker_pending_load_cv.notify_all()
            elif tag == _IPC_DROP:
                (uuid, ) = obj
                with self._worker_pending_load_cv:
                    self._worker_pending_load[uuid] = None
                    self._worker_pending_load_cv.notify_all()
                with self._worker_pending_stage_cv:
                    self._worker_pending_stage.pop(uuid, None)
                    self._worker_pending_stage_cv.notify_all()
            else:
                logger.warning(
                    "TPUConnectorWorker rank%d --> unknown IPC tag=%s",
                    self.tp_rank, tag)

    # =========================================================
    # Rank 0: external data-server (producer serves PULL)
    # =========================================================
    def _coord_rank0_external_data_loop(self, channel_idx: int) -> None:
        """One data server per TCP channel. channel_idx in [0, n_channels).

        A rank's payload rides channel ``rank % n_channels``. Channel 0 also
        prepends the pickled header so the consumer can derive tp_size /
        shapes before reassembling. Each channel binds its own port
        ``base + channel_idx`` and serves its subset of ranks only, giving
        N parallel TCP streams end-to-end."""
        port = self._kv_transfer_ports[channel_idx]
        sock_path = make_zmq_path("tcp", "*", port)
        sock = make_zmq_socket(ctx=self.zmq_cxt,
                               path=sock_path,
                               socket_type=zmq.ROUTER,
                               bind=True)
        sock.setsockopt(zmq.ROUTER_MANDATORY, 0)
        ranks_on_channel = [
            r for r in range(self.tp_size)
            if r % self._n_channels == channel_idx
        ]
        logger.info(
            "TPUConnectorWorker rank0 --> data server ch=%d listening on "
            "tcp://%s:%s serving ranks=%s", channel_idx, self.host_ip, port,
            ranks_on_channel)
        while not self._stop_event.is_set():
            try:
                if sock.poll(timeout=500) == 0:
                    continue
                frames = sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker rank0 --> data recv "
                    "ch=%d error: %s", channel_idx, e)
                continue
            # Expected: [client_id, PULL, uuid_bytes, remote_blocks_pickle]
            if len(frames) < 3 or frames[1] != _MSG_PULL:
                logger.warning(
                    "TPUConnectorWorker rank0 --> data malformed "
                    "request ch=%d n=%d", channel_idx, len(frames))
                continue
            client_id = frames[0]
            uuid_bytes = frames[2]
            try:
                uuid = int(uuid_bytes.decode("utf-8"))
            except ValueError:
                sock.send_multipart([client_id, _MSG_ERR, b"bad-uuid"])
                continue
            logger.info(
                "TPUConnectorWorker rank0 --> PULL received ch=%d uuid=%s",
                channel_idx, uuid)
            response = self._coord_rank0_build_pull_response(
                uuid, uuid_bytes, channel_idx, ranks_on_channel)
            if response is None:
                sock.send_multipart([client_id, _MSG_ERR, uuid_bytes])
                continue
            # copy=False lets ZMQ send the big per-layer memoryviews
            # directly from shm via scatter-gather sendmsg, no userland
            # copy. Small framing frames pay a tiny per-frame overhead;
            # net win is ~256MB of memcpy avoided per rank.
            sock.send_multipart([client_id, *response], copy=False)

    def _coord_rank0_build_pull_response(
            self, uuid: int, uuid_bytes: bytes, channel_idx: int,
            ranks_on_channel: list[int]) -> Optional[list[bytes]]:
        # A PULL can arrive before the scheduler has routed reqs_to_send
        # through process_send_load (the D-side orchestrator may dispatch
        # the pull immediately after handing D the uuid, racing P's own
        # metadata tick). Wait up to the p2p timeout for the entry to
        # appear, then wait up to the remaining budget for staging. With
        # n_channels parallel PULLs the race window is n_channels-fold more
        # likely to fire, so this backoff is required for correctness.
        timeout = dist_utils.get_p2p_wait_pull_timeout()
        deadline = time.perf_counter() + timeout
        entry = None
        while time.perf_counter() < deadline:
            with self._coord_lock:
                entry = self._coord_send.get(uuid)
            if entry is not None:
                break
            time.sleep(0.05)
        if entry is None:
            logger.warning(
                "TPUConnectorWorker rank0 --> PULL ch=%d unknown uuid=%s "
                "(timed out waiting for register-send)", channel_idx, uuid)
            return None
        remaining = max(0.0, deadline - time.perf_counter())
        if not entry.stage_complete.wait(timeout=remaining):
            logger.warning(
                "TPUConnectorWorker rank0 --> PULL ch=%d uuid=%s timed out "
                "waiting for staging (have %d of %d)", channel_idx, uuid,
                len(entry.staged), self.tp_size)
            return None
        # Wire format (per channel, after the ROUTER envelope):
        #   ch 0:  [OK, uuid, n_ranks, header,
        #            (rank_idx, layer_mv_0, ..., layer_mv_{L-1})*]
        #   ch >0: [OK, uuid, n_ranks,
        #            (rank_idx, layer_mv_0, ..., layer_mv_{L-1})*]
        # Each layer_mv is a memoryview straight into the shm slot for
        # that (rank, layer); there is no bytearray/bytes staging on
        # the producer. Consumer knows num_layers from self.num_layers
        # so it doesn't need to be inline; the header on ch 0 carries
        # the sanity-check metadata.
        frames: list = [
            _MSG_OK, uuid_bytes,
            str(len(ranks_on_channel)).encode("utf-8")
        ]
        if channel_idx == 0:
            header = {
                "tp_size":
                self.tp_size,
                "num_layers":
                self.num_layers,
                "num_blocks":
                entry.num_blocks,
                "dtype":
                str(self._coord_pool_spec.dtype),
                "layer_shard_shape":
                list(self._coord_pool_spec.layer_shard_shape[1:]),
            }
            frames.append(
                pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL))
        total_bytes = 0
        for r in ranks_on_channel:
            frames.append(str(r).encode("utf-8"))
            layer_views = self._coord_pool.rank_layer_views(
                entry.slot_idx, r, entry.num_blocks)
            frames.extend(layer_views)
            total_bytes += sum(len(v) for v in layer_views)
        logger.info(
            "TPUConnectorWorker rank0 --> serving PULL ch=%d uuid=%s "
            "ranks=%s size=%.2fMB", channel_idx, uuid, ranks_on_channel,
            total_bytes / (1024 * 1024))
        return frames

    # =========================================================
    # Rank 0: external side-channel (producer receives NOTIFY)
    # =========================================================
    def _coord_rank0_external_notif_loop(self) -> None:
        sock_path = make_zmq_path("tcp", "*", self.side_channel_port)
        sock = make_zmq_socket(ctx=self.zmq_cxt,
                               path=sock_path,
                               socket_type=zmq.ROUTER,
                               bind=True)
        logger.info("TPUConnectorWorker rank0 --> side channel on %s",
                    sock_path)
        while not self._stop_event.is_set():
            try:
                if sock.poll(timeout=500) == 0:
                    continue
                _client_id, uuid_bytes = sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker rank0 --> notif recv error: %s", e)
                continue
            try:
                uuid = int(uuid_bytes.decode("utf-8"))
            except ValueError:
                continue
            # Mark the send entry as acked; get_finished() sweeps it.
            with self._coord_lock:
                entry = self._coord_send.get(uuid)
                if entry is None:
                    logger.warning(
                        "TPUConnectorWorker rank0 --> stray pull-done "
                        "uuid=%s", uuid)
                    continue
                entry.pull_acked = True
            logger.info("TPUConnectorWorker rank0 --> pull-done uuid=%s", uuid)

    # =========================================================
    # Rank 0: periodic expiration sweep
    # =========================================================
    def _coord_rank0_expire_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=1.0):
                return
            now = time.perf_counter()
            expired: list[int] = []
            with self._coord_lock:
                for uuid, entry in list(self._coord_send.items()):
                    # Acked entries are handled in get_finished() so they
                    # surface to the scheduler in the correct tick.
                    if not entry.pull_acked and now > entry.expiration_time:
                        expired.append(uuid)
            for uuid in expired:
                logger.warning(
                    "TPUConnectorWorker rank0 --> expire send uuid=%s", uuid)
                with self._coord_lock:
                    entry = self._coord_send.pop(uuid, None)
                    if entry is None:
                        continue
                    self._coord_done_sending.add(entry.req_id)
                self._coord_pool.release_slot(entry.slot_idx)

    # =========================================================
    # get_finished (coord mode)
    # =========================================================
    def _coord_get_finished(self) -> tuple[set[str], set[str]]:
        # Only rank 0 returns non-empty; scheduler unions across workers.
        if self.tp_rank != 0:
            return set(), set()
        done_sending: set[str] = set()
        done_recving: set[str] = set()
        released_slots: list[int] = []
        with self._coord_lock:
            # Producer: pull_acked → done_sending; scheduler unioning means
            # we only need to report once.
            for uuid in list(self._coord_send.keys()):
                entry = self._coord_send[uuid]
                if entry.pull_acked:
                    done_sending.add(entry.req_id)
                    del self._coord_send[uuid]
                    released_slots.append(entry.slot_idx)
            # Merge in any expiration-sweeper results (already removed from
            # _coord_send by that thread, slot already released).
            if self._coord_done_sending:
                done_sending |= self._coord_done_sending
                self._coord_done_sending.clear()
            # Consumer: pull_ok + load_complete → done_recving. Mark
            # `reported_done` so repeated get_finished() calls don't emit
            # the same req id twice; the entry stays in _coord_recv until
            # all ranks have acknowledged their scatter via COPY_DONE.
            for uuid, entry in list(self._coord_recv.items()):
                if (entry.load_complete.is_set() and entry.pull_ok
                        and not entry.reported_done):
                    done_recving.add(entry.req_id)
                    entry.reported_done = True
            # Merge in failure-surfacing reqs (pull failed, shm pool
            # exhausted on admit). The scheduler advances these as if
            # the KV were ready; the output will be wrong for that req,
            # but the alternative is an engine-wide hang.
            if self._coord_done_recving:
                done_recving |= self._coord_done_recving
                self._coord_done_recving.clear()
        # Free slots outside the lock to avoid nesting lock waits.
        for sidx in released_slots:
            self._coord_pool.release_slot(sidx)
        if done_sending:
            logger.info("TPUConnectorWorker rank0 --> done_sending=%s",
                        done_sending)
        if done_recving:
            logger.info("TPUConnectorWorker rank0 --> done_recving=%s",
                        done_recving)
        return done_sending, done_recving


def get_uuid() -> int:
    int128 = uuid4().int
    # Stay under 64-bit so JSON-encoded responses through the proxy are safe.
    return int128 >> 78


def _try_remove_ipc_endpoint(ipc_path: str) -> None:
    """ZMQ's ipc:// backs onto a filesystem path; a stale file from a
    previous crashed rank-0 makes bind() fail. Remove it best-effort."""
    import os as _os
    if not ipc_path.startswith("ipc://"):
        return
    path = ipc_path[len("ipc://"):]
    try:
        _os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not clean up stale IPC endpoint %s: %s", path, e)
