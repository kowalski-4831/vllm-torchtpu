# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Raiden-only KV transfer adapter."""

import pytest
import torch

from tpu_inference.distributed.kv_transfer.raiden_transfer_engine import (
    RaidenTransferEngine, _import_backend_module)


class _FakeBackendModule:

    class TransferEngine:
        instances = []

        def __init__(self,
                     kv_caches,
                     tp_rank,
                     local_control_port,
                     max_blocks,
                     num_slots,
                     timeout_s,
                     unsafe_skip_buffer_lock=True):
            self.kv_caches = kv_caches
            self.tp_rank = tp_rank
            self.local_control_port = local_control_port
            self.max_blocks = max_blocks
            self.num_slots = num_slots
            self.timeout_s = timeout_s
            self.unsafe_skip_buffer_lock = unsafe_skip_buffer_lock
            self.calls = []
            _FakeBackendModule.TransferEngine.instances.append(self)

        def notify_for_read(self, req_id, uuid, block_ids):
            self.calls.append(("notify_for_read", req_id, uuid, block_ids))
            return 7

        def start_read(self, req_id, uuid, remote_endpoint, remote_block_ids,
                       local_block_ids):
            self.calls.append(("start_read", req_id, uuid, remote_endpoint,
                               remote_block_ids, local_block_ids))
            return 8

        def complete_read(self):
            self.calls.append(("complete_read", ))
            return ["sent"], ["recv"], []


class _LegacyFakeBackendModule:

    class TransferEngine(_FakeBackendModule.TransferEngine):
        instances = []
        notify_for_read = None
        start_read = None
        complete_read = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _LegacyFakeBackendModule.TransferEngine.instances.append(self)

        def register_send(self, req_id, uuid, block_ids):
            self.calls.append(("register_send", req_id, uuid, block_ids))
            return 7

        def submit_load(self, req_id, uuid, remote_endpoint, remote_block_ids,
                        local_block_ids):
            self.calls.append(("submit_load", req_id, uuid, remote_endpoint,
                               remote_block_ids, local_block_ids))
            return 8

        def poll_finished(self):
            self.calls.append(("poll_finished", ))
            return ["sent"], ["recv"], []


class _LegacyNamedBackendModule:

    class RaidenTransferEngine(_FakeBackendModule.TransferEngine):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _LegacyNamedBackendModule.RaidenTransferEngine.instances.append(
                self)


def test_adapter_delegates_to_cpp_backend():
    _FakeBackendModule.TransferEngine.instances = []
    kv_caches = [torch.empty(1)]
    engine = RaidenTransferEngine(kv_caches=kv_caches,
                                  tp_rank=2,
                                  local_control_port=9104,
                                  max_blocks=3,
                                  num_slots=4,
                                  timeout_s=12.0,
                                  backend_module=_FakeBackendModule,
                                  unsafe_skip_buffer_lock=False)
    backend = _FakeBackendModule.TransferEngine.instances[-1]

    assert backend.kv_caches == kv_caches
    assert backend.tp_rank == 2
    assert backend.local_control_port == 9104
    assert backend.max_blocks == 3
    assert backend.num_slots == 4
    assert backend.timeout_s == 12.0
    assert backend.unsafe_skip_buffer_lock is False

    assert engine.register_send("req-p", 123, [1, 2]) == 7
    assert engine.submit_load("req-d", 123, "host:1", [1], [2]) == 8
    assert engine.poll_finished() == (["sent"], ["recv"], [])

    assert [call[0] for call in backend.calls] == [
        "notify_for_read",
        "start_read",
        "complete_read",
    ]


def test_adapter_delegates_to_legacy_cpp_backend_names():
    _LegacyFakeBackendModule.TransferEngine.instances = []
    engine = RaidenTransferEngine(kv_caches=[],
                                  tp_rank=0,
                                  local_control_port=9100,
                                  max_blocks=1,
                                  num_slots=1,
                                  timeout_s=1.0,
                                  backend_module=_LegacyFakeBackendModule)
    backend = _LegacyFakeBackendModule.TransferEngine.instances[-1]

    assert engine.register_send("req-p", 123, [1, 2]) == 7
    assert engine.submit_load("req-d", 123, "host:1", [1], [2]) == 8
    assert engine.poll_finished() == (["sent"], ["recv"], [])

    assert [call[0] for call in backend.calls] == [
        "register_send",
        "submit_load",
        "poll_finished",
    ]


def test_adapter_accepts_legacy_backend_class_name():
    _LegacyNamedBackendModule.RaidenTransferEngine.instances = []
    engine = RaidenTransferEngine(kv_caches=[],
                                  tp_rank=0,
                                  local_control_port=9100,
                                  max_blocks=1,
                                  num_slots=1,
                                  timeout_s=1.0,
                                  backend_module=_LegacyNamedBackendModule)
    backend = _LegacyNamedBackendModule.RaidenTransferEngine.instances[-1]

    assert engine.register_send("req-p", 123, [1, 2]) == 7
    assert backend.calls == [("notify_for_read", "req-p", 123, [1, 2])]


def test_missing_cpp_backend_export_fails_fast():
    with pytest.raises(ImportError, match="TransferEngine"):
        RaidenTransferEngine(kv_caches=[],
                             tp_rank=0,
                             local_control_port=9100,
                             max_blocks=1,
                             num_slots=1,
                             timeout_s=1.0,
                             backend_module=object())


def test_import_backend_prefers_current_tpu_raiden_api(monkeypatch):
    backend = object()
    calls = []

    def fake_import(module_name):
        calls.append(module_name)
        return backend

    monkeypatch.setattr(
        "tpu_inference.distributed.kv_transfer.raiden_transfer_engine.importlib.import_module",
        fake_import)

    assert _import_backend_module() is backend
    assert calls == ["api.torch.transfer_engine"]


def test_import_backend_falls_back_to_legacy_api(monkeypatch):
    backend = object()
    calls = []

    def fake_import(module_name):
        calls.append(module_name)
        if module_name == "api.torch.transfer_engine":
            raise ModuleNotFoundError(
                "No module named 'api.torch.transfer_engine'",
                name="api.torch.transfer_engine")
        return backend

    monkeypatch.setattr(
        "tpu_inference.distributed.kv_transfer.raiden_transfer_engine.importlib.import_module",
        fake_import)

    assert _import_backend_module() is backend
    assert calls == [
        "api.torch.transfer_engine",
        "api.torch.raiden_transfer_engine",
    ]


def test_import_backend_preserves_internal_import_failures(monkeypatch):

    def fake_import(module_name):
        raise ModuleNotFoundError("No module named 'torch_tpu'",
                                  name="torch_tpu")

    monkeypatch.setattr(
        "tpu_inference.distributed.kv_transfer.raiden_transfer_engine.importlib.import_module",
        fake_import)

    with pytest.raises(ModuleNotFoundError, match="torch_tpu"):
        _import_backend_module()
