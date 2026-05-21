# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TPUConnector, TPUConnectorScheduler, and TPUConnectorWorker.

Structure mirrors tests/distributed/test_tpu_connector.py in the upstream
tpu-inference repo, adapted for the torch-based coordinator-mode connector
used here.  Heavy dependencies (ZMQ sockets, shared-memory pool, TPU device
queries) are mocked at construction time so that all tests run without a real
TPU.
"""
import threading
from unittest.mock import MagicMock, patch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.request import RequestStatus

from tpu_inference.distributed.kv_transfer.tpu_connector import (  # isort: skip
    LoadMeta, TPUConnector, TPUConnectorMetadata, TPUConnectorScheduler,
    TPUConnectorWorker, _CoordRecvEntry, _CoordSendEntry)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_MOD = "tpu_inference.distributed.kv_transfer.tpu_connector"
_BASE = "tpu_inference.distributed.kv_transfer.zmq_shm_base"


def _make_vllm_config(*, is_producer: bool = True, block_size: int = 16):
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.cache_config.block_size = block_size
    return cfg


def _make_scheduler(*, is_producer: bool = False):
    """Construct a TPUConnectorScheduler with network calls patched out."""
    cfg = _make_vllm_config(is_producer=is_producer)
    with patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"), \
         patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100):
        return TPUConnectorScheduler(cfg)


def _make_worker(*,
                 tp_rank: int = 0,
                 tp_size: int = 1,
                 is_producer: bool = True) -> TPUConnectorWorker:
    """Construct a TPUConnectorWorker with all I/O and device calls mocked.

    Patches are applied only during __init__; the returned object has real
    Python state (locks, queues, dicts) but no live ZMQ or TPU handles.
    register_runner() is intentionally NOT called — tests that need it
    should mock _coord_setup and call it separately.
    """
    cfg = _make_vllm_config(is_producer=is_producer)
    with patch(f"{_BASE}.get_tensor_model_parallel_rank", return_value=tp_rank), \
         patch(f"{_BASE}.get_tensor_model_parallel_world_size", return_value=tp_size), \
         patch(f"{_BASE}.dist_utils.get_node_id", return_value=0), \
         patch(f"{_BASE}.dist_utils.get_host_ip", return_value="127.0.0.1"), \
         patch(f"{_BASE}.dist_utils.get_kv_transfer_port", return_value="9100"), \
         patch(f"{_BASE}.dist_utils.get_side_channel_port", return_value="9600"), \
         patch(f"{_BASE}.dist_utils.get_transfer_channel_number", return_value=0), \
         patch(f"{_BASE}.dist_utils.get_kv_latency_log_interval", return_value=0.0), \
         patch(f"{_BASE}.zmq.Context"):
        return TPUConnectorWorker(cfg)


# ---------------------------------------------------------------------------
# TestTPUConnector — role initialisation and method delegation
# ---------------------------------------------------------------------------


class TestTPUConnector:

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_init_scheduler_role(self, mock_sched_cls, mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.SCHEDULER)
        mock_sched_cls.assert_called_once_with(cfg)
        mock_worker_cls.assert_not_called()
        assert connector.connector_scheduler is not None
        assert connector.connector_worker is None

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_init_worker_role(self, mock_sched_cls, mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.WORKER)
        mock_worker_cls.assert_called_once_with(cfg)
        mock_sched_cls.assert_not_called()
        assert connector.connector_scheduler is None
        assert connector.connector_worker is not None

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_scheduler_method_delegation(self, mock_sched_cls,
                                         mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.SCHEDULER)
        sched = mock_sched_cls.return_value
        req, blocks, sched_out = MagicMock(), MagicMock(), MagicMock()

        connector.get_num_new_matched_tokens(req, 16)
        sched.get_num_new_matched_tokens.assert_called_once_with(req, 16)

        connector.update_state_after_alloc(req, blocks, 16)
        sched.update_state_after_alloc.assert_called_once_with(req, blocks, 16)

        # build_connector_meta receives scheduler_output but strips it before
        # delegating — verify the downstream call has no arguments.
        connector.build_connector_meta(sched_out)
        sched.build_connector_meta.assert_called_once_with()

        connector.request_finished(req, [1, 2])
        sched.request_finished.assert_called_once_with(req, [1, 2])

        connector.get_finished_count()
        sched.get_finished_count.assert_called_once_with()

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_worker_method_delegation(self, mock_sched_cls, mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.WORKER)
        worker = mock_worker_cls.return_value
        runner, meta = MagicMock(), TPUConnectorMetadata()

        connector.register_runner(runner)
        worker.register_runner.assert_called_once_with(runner)

        connector._connector_metadata = meta
        connector.start_load_kv(None)
        worker.process_send_load.assert_called_once_with(meta)

        connector.get_finished(set())
        worker.get_finished.assert_called_once_with()


# ---------------------------------------------------------------------------
# TestTPUConnectorScheduler — arithmetic and state-transition logic
# ---------------------------------------------------------------------------


class TestTPUConnectorScheduler:

    def setup_method(self):
        self.consumer = _make_scheduler(is_producer=False)
        self.producer = _make_scheduler(is_producer=True)

    # ---- get_num_new_matched_tokens ----------------------------------------

    def test_producer_always_returns_zero(self):
        req = MagicMock()
        req.kv_transfer_params = {"uuid": 1}
        n, is_async = self.producer.get_num_new_matched_tokens(req, 0)
        assert n == 0
        assert not is_async

    def test_no_kv_transfer_params_returns_zero(self):
        req = MagicMock()
        req.kv_transfer_params = None
        n, is_async = self.consumer.get_num_new_matched_tokens(req, 0)
        assert n == 0
        assert not is_async

    def test_consumer_partial_cache_hit(self):
        # prompt=35 tokens, block_size=16 → round_down=32
        # num_computed=16  → 32-16=16 tokens to load
        req = MagicMock()
        req.prompt_token_ids = [0] * 35
        req.kv_transfer_params = {"uuid": 1}
        n, is_async = self.consumer.get_num_new_matched_tokens(req, 16)
        assert n == 16
        assert is_async

    def test_consumer_full_cache_hit_returns_zero(self):
        # prompt=31 → round_down=16; computed=32 → max(16-32, 0)=0
        req = MagicMock()
        req.prompt_token_ids = [0] * 31
        req.kv_transfer_params = {"uuid": 1}
        n, is_async = self.consumer.get_num_new_matched_tokens(req, 32)
        assert n == 0
        assert not is_async

    # ---- update_state_after_alloc ------------------------------------------

    def test_update_producer_is_noop(self):
        self.producer.update_state_after_alloc(MagicMock(), MagicMock(), 32)
        assert len(self.producer.reqs_to_load) == 0

    def test_update_no_kv_params_is_noop(self):
        req = MagicMock()
        req.kv_transfer_params = None
        self.consumer.update_state_after_alloc(req, MagicMock(), 32)
        assert len(self.consumer.reqs_to_load) == 0

    def test_update_consumer_external_tokens_populates_full_meta(self):
        req = MagicMock()
        req.request_id = "req-1"
        req.kv_transfer_params = {
            "uuid": 42,
            "remote_block_ids": [10, 11],
            "remote_host": "2.2.2.2",
            "remote_port": 9200,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[1, 2]]

        self.consumer.update_state_after_alloc(req, blocks, 32)

        assert "req-1" in self.consumer.reqs_to_load
        meta = self.consumer.reqs_to_load["req-1"]
        assert meta.uuid == 42
        assert meta.local_block_ids == [1, 2]
        assert meta.remote_block_ids == [10, 11]
        assert meta.remote_host == "2.2.2.2"
        assert meta.remote_port == 9200

    def test_update_consumer_zero_external_tokens_sets_nones(self):
        """Cache hit / async drain: both local and remote block ids are None."""
        req = MagicMock()
        req.request_id = "req-2"
        req.kv_transfer_params = {
            "uuid": 99,
            "remote_block_ids": [5, 6],
            "remote_host": "3.3.3.3",
            "remote_port": 9300,
        }

        self.consumer.update_state_after_alloc(req, MagicMock(), 0)

        assert "req-2" in self.consumer.reqs_to_load
        meta = self.consumer.reqs_to_load["req-2"]
        assert meta.uuid == 99
        assert meta.local_block_ids is None
        assert meta.remote_block_ids is None

    # ---- build_connector_meta ----------------------------------------------

    def test_build_meta_drains_reqs_to_send(self):
        self.producer.reqs_to_send = {"r1": "m1"}
        meta = self.producer.build_connector_meta()
        assert meta.reqs_to_send == {"r1": "m1"}
        assert len(self.producer.reqs_to_send) == 0

    def test_build_meta_drains_reqs_to_load(self):
        self.consumer.reqs_to_load = {"r2": "m2"}
        meta = self.consumer.build_connector_meta()
        assert meta.reqs_to_load == {"r2": "m2"}
        assert len(self.consumer.reqs_to_load) == 0

    # ---- request_finished --------------------------------------------------

    def test_consumer_request_finished_is_noop(self):
        delay, params = self.consumer.request_finished(MagicMock(), [])
        assert not delay
        assert params is None

    def test_producer_not_length_capped_returns_no_delay(self):
        req = MagicMock()
        req.status = RequestStatus.RUNNING
        delay, params = self.producer.request_finished(req, [1, 2])
        assert not delay
        assert params is None

    @patch(f"{_MOD}.get_uuid", return_value=777)
    @patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30)
    def test_producer_finished_all_full_blocks(self, _timeout, _uuid):
        req = MagicMock()
        req.request_id = "p-full"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 32  # 32 % 16 == 0 → all_full=True

        delay, params = self.producer.request_finished(req, [3, 4])

        assert delay
        send = self.producer.reqs_to_send["p-full"]
        assert send.uuid == 777
        assert send.local_block_ids == [3, 4]
        assert params["uuid"] == 777
        assert params["remote_block_ids"] == [3, 4]
        assert params["remote_host"] == "127.0.0.1"
        assert params["remote_port"] == 9100

    @patch(f"{_MOD}.get_uuid", return_value=888)
    @patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30)
    def test_producer_finished_trailing_partial_block_is_dropped(
            self, _timeout, _uuid):
        """Trailing partial block must not be transferred: last id excluded."""
        req = MagicMock()
        req.request_id = "p-partial"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 33  # 33 % 16 != 0 → last block is partial

        delay, params = self.producer.request_finished(req, [5, 6, 7])

        assert delay
        send = self.producer.reqs_to_send["p-partial"]
        assert send.local_block_ids == [5, 6]  # [7] dropped
        assert params["remote_block_ids"] == [5, 6]

    def test_producer_finished_only_partial_block_returns_no_delay(self):
        """When the sole block is partial, no transfer is triggered."""
        req = MagicMock()
        req.request_id = "p-tiny"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 5  # 5 % 16 != 0

        delay, params = self.producer.request_finished(req, [8])

        assert not delay
        assert params == {}
        assert "p-tiny" not in self.producer.reqs_to_send

    # ---- get_finished_count ------------------------------------------------

    def test_get_finished_count_single_host(self):
        self.consumer.kv_ip = "1.1.1.1"
        assert self.consumer.get_finished_count() == 1

    def test_get_finished_count_multi_host(self):
        self.consumer.kv_ip = ["1.1.1.1", "2.2.2.2", "3.3.3.3"]
        assert self.consumer.get_finished_count() == 3


# ---------------------------------------------------------------------------
# TestTPUConnectorWorkerInit — construction-time state
# ---------------------------------------------------------------------------


class TestTPUConnectorWorkerInit:
    """Verify the coordinator state set up by __init__ / _init_coord_state.

    register_runner() is not called; all shm/ZMQ/thread bringup is skipped.
    """

    def test_tp1_coord_workers_ready_is_set_eagerly(self):
        # With a single rank there are no peers to wait for.
        worker = _make_worker(tp_rank=0, tp_size=1)
        assert worker._coord_workers_ready.is_set()

    def test_tp2_coord_workers_ready_not_yet_set(self):
        worker = _make_worker(tp_rank=0, tp_size=2)
        assert not worker._coord_workers_ready.is_set()

    def test_n_channels_auto_equals_tp_size(self):
        # get_transfer_channel_number returns 0 → auto → n_channels == tp_size
        worker = _make_worker(tp_rank=0, tp_size=4)
        assert worker._n_channels == 4

    def test_n_channels_override_clamped_to_tp_size(self):
        # Requesting 16 channels on a TP=4 worker is clamped to 4.
        with patch(f"{_BASE}.get_tensor_model_parallel_rank", return_value=0), \
             patch(f"{_BASE}.get_tensor_model_parallel_world_size", return_value=4), \
             patch(f"{_BASE}.dist_utils.get_node_id", return_value=0), \
             patch(f"{_BASE}.dist_utils.get_host_ip", return_value="127.0.0.1"), \
             patch(f"{_BASE}.dist_utils.get_kv_transfer_port", return_value="9100"), \
             patch(f"{_BASE}.dist_utils.get_side_channel_port", return_value="9600"), \
             patch(f"{_BASE}.dist_utils.get_transfer_channel_number", return_value=16), \
             patch(f"{_BASE}.dist_utils.get_kv_latency_log_interval", return_value=0.0), \
             patch(f"{_BASE}.zmq.Context"):
            worker = TPUConnectorWorker(_make_vllm_config())
        assert worker._n_channels == 4

    def test_kv_scatter_disabled_until_smoke_test(self):
        # Scatter is only enabled after register_runner runs the smoke test.
        worker = _make_worker()
        assert not worker._kv_scatter_enabled

    def test_kv_transfer_ports_span_n_channels(self):
        worker = _make_worker(tp_rank=0, tp_size=3)
        base = int(worker.kv_transfer_port)
        assert worker._kv_transfer_ports == [base, base + 1, base + 2]


# ---------------------------------------------------------------------------
# TestResolveRemoteHostPort — multi-host routing helper
# ---------------------------------------------------------------------------


class TestResolveRemoteHostPort:
    """_resolve_remote_host_port picks the right host/port for this node_id."""

    def setup_method(self):
        self.worker = _make_worker(tp_rank=0, tp_size=1)

    def test_single_host_passthrough(self):
        meta = LoadMeta(uuid=1,
                        local_block_ids=[0],
                        remote_block_ids=[0],
                        remote_host="1.2.3.4",
                        remote_port=9100)
        host, port = self.worker._resolve_remote_host_port(meta)
        assert host == "1.2.3.4"
        assert port == 9100

    def test_multi_host_selects_node_id_entry(self):
        # node_id=0 (set in _make_worker) → picks index 0
        meta = LoadMeta(uuid=2,
                        local_block_ids=[0],
                        remote_block_ids=[0],
                        remote_host=["10.0.0.1", "10.0.0.2"],
                        remote_port=[9100, 9101])
        host, port = self.worker._resolve_remote_host_port(meta)
        assert host == "10.0.0.1"
        assert port == 9100


# ---------------------------------------------------------------------------
# TestCoordGetFinished — _coord_get_finished state-machine tests
# ---------------------------------------------------------------------------


class TestCoordGetFinished:
    """Directly inject coordinator state and verify the returned finished sets.

    These tests cover the six state transitions in _coord_get_finished without
    needing a running ZMQ socket, shm pool, or real TPU.
    """

    def setup_method(self):
        self.worker = _make_worker(tp_rank=0, tp_size=1, is_producer=True)
        # _coord_pool is None until register_runner; mock it so release_slot
        # calls don't crash.
        self.worker._coord_pool = MagicMock()

    # ---- non-zero rank short-circuit ---------------------------------------

    def test_non_zero_rank_returns_empty_sets(self):
        worker = _make_worker(tp_rank=1, tp_size=2)
        done_sending, done_recving = worker._coord_get_finished()
        assert done_sending == set()
        assert done_recving == set()

    # ---- producer / done_sending -------------------------------------------

    def test_pull_acked_entry_moves_to_done_sending(self):
        entry = _CoordSendEntry(req_id="req-acked",
                                slot_idx=0,
                                num_blocks=4,
                                expiration_time=1e9,
                                pull_acked=True)
        self.worker._coord_send[10] = entry

        done_sending, done_recving = self.worker._coord_get_finished()

        assert done_sending == {"req-acked"}
        assert done_recving == set()
        assert 10 not in self.worker._coord_send
        self.worker._coord_pool.release_slot.assert_called_once_with(0)

    def test_unacked_entry_stays_in_coord_send(self):
        entry = _CoordSendEntry(req_id="req-pending",
                                slot_idx=1,
                                num_blocks=4,
                                expiration_time=1e9,
                                pull_acked=False)
        self.worker._coord_send[20] = entry

        done_sending, _ = self.worker._coord_get_finished()

        assert done_sending == set()
        assert 20 in self.worker._coord_send
        self.worker._coord_pool.release_slot.assert_not_called()

    def test_expiry_sweep_results_drained_once(self):
        """Entries written by the expiration-sweeper thread are merged in and
        cleared so they aren't double-reported on the next tick."""
        self.worker._coord_done_sending = {"swept-req"}

        done_sending, _ = self.worker._coord_get_finished()

        assert done_sending == {"swept-req"}
        assert len(self.worker._coord_done_sending) == 0

    # ---- consumer / done_recving -------------------------------------------

    def test_load_complete_pull_ok_moves_to_done_recving(self):
        ev = threading.Event()
        ev.set()
        entry = _CoordRecvEntry(req_id="req-pulled",
                                uuid=55,
                                slot_idx=2,
                                num_blocks=4,
                                local_blocks=[0, 1, 2, 3],
                                remote_blocks=[10, 11, 12, 13],
                                remote_host="1.2.3.4",
                                remote_port=9100,
                                load_complete=ev,
                                pull_ok=True,
                                reported_done=False)
        self.worker._coord_recv[55] = entry

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == {"req-pulled"}
        assert entry.reported_done is True

    def test_incomplete_pull_not_reported(self):
        ev = threading.Event()  # not set
        entry = _CoordRecvEntry(req_id="req-in-flight",
                                uuid=56,
                                slot_idx=3,
                                num_blocks=4,
                                local_blocks=[],
                                remote_blocks=[],
                                remote_host="1.2.3.4",
                                remote_port=9100,
                                load_complete=ev,
                                pull_ok=False,
                                reported_done=False)
        self.worker._coord_recv[56] = entry

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == set()
        assert not entry.reported_done

    def test_reported_done_prevents_duplicate_reporting(self):
        ev = threading.Event()
        ev.set()
        entry = _CoordRecvEntry(
            req_id="req-already-reported",
            uuid=57,
            slot_idx=4,
            num_blocks=4,
            local_blocks=[],
            remote_blocks=[],
            remote_host="1.2.3.4",
            remote_port=9100,
            load_complete=ev,
            pull_ok=True,
            reported_done=True)  # already reported on a prior tick
        self.worker._coord_recv[57] = entry

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == set()

    def test_failure_recving_surfaced_and_cleared(self):
        """Requests in _coord_done_recving (pool exhausted / pull failed) are
        surfaced exactly once so the scheduler doesn't hang waiting for them."""
        self.worker._coord_done_recving = {"failed-req"}

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == {"failed-req"}
        assert len(self.worker._coord_done_recving) == 0

    # ---- combined tick -----------------------------------------------------

    def test_send_and_recv_reported_in_same_tick(self):
        entry_send = _CoordSendEntry(req_id="sending-req",
                                     slot_idx=5,
                                     num_blocks=2,
                                     expiration_time=1e9,
                                     pull_acked=True)
        self.worker._coord_send[100] = entry_send

        ev = threading.Event()
        ev.set()
        entry_recv = _CoordRecvEntry(req_id="recving-req",
                                     uuid=101,
                                     slot_idx=6,
                                     num_blocks=2,
                                     local_blocks=[0, 1],
                                     remote_blocks=[10, 11],
                                     remote_host="1.2.3.4",
                                     remote_port=9100,
                                     load_complete=ev,
                                     pull_ok=True,
                                     reported_done=False)
        self.worker._coord_recv[101] = entry_recv

        done_sending, done_recving = self.worker._coord_get_finished()

        assert done_sending == {"sending-req"}
        assert done_recving == {"recving-req"}
