# SPDX-License-Identifier: Apache-2.0
"""Raiden-only KV transfer engine adapter for TPU disaggregation."""

from __future__ import annotations

import importlib
from typing import Any

import torch

_BACKEND_MODULES = (
    "api.torch.transfer_engine",
    "api.torch.raiden_transfer_engine",
)
_BACKEND_CLASS_NAMES = ("TransferEngine", "RaidenTransferEngine")


def _is_missing_requested_module(exc: ImportError, module_name: str) -> bool:
    missing_name = getattr(exc, "name", None)
    if not missing_name:
        return False
    return module_name == missing_name or module_name.startswith(
        f"{missing_name}.")


def _import_backend_module() -> Any:
    last_error: ImportError | None = None
    for module_name in _BACKEND_MODULES:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if not _is_missing_requested_module(exc, module_name):
                raise
            last_error = exc
    module_list = ", ".join(_BACKEND_MODULES)
    raise ImportError(
        f"Could not import Raiden backend from {module_list}") from last_error


def _get_backend_class(backend_module: Any) -> type:
    for class_name in _BACKEND_CLASS_NAMES:
        backend_cls = getattr(backend_module, class_name, None)
        if backend_cls is not None:
            return backend_cls
    raise ImportError("Raiden backend does not export TransferEngine "
                      "(or legacy RaidenTransferEngine)")


class RaidenTransferEngine:
    """Thin adapter around the standalone C++ Raiden transfer backend."""

    def __init__(
        self,
        *,
        kv_caches: list[torch.Tensor],
        tp_rank: int,
        local_control_port: int,
        max_blocks: int,
        num_slots: int,
        timeout_s: float,
        backend_module: Any | None = None,
        unsafe_skip_buffer_lock: bool = True,
    ) -> None:
        backend_module = backend_module or _import_backend_module()
        backend_cls = _get_backend_class(backend_module)
        self._backend = backend_cls(list(kv_caches), tp_rank,
                                    int(local_control_port), int(max_blocks),
                                    int(num_slots), float(timeout_s),
                                    unsafe_skip_buffer_lock)

    def _call_backend(self, current_name: str, legacy_name: str, *args:
                      Any) -> Any:
        method = getattr(self._backend, current_name, None)
        if method is None:
            method = getattr(self._backend, legacy_name, None)
        if method is None:
            raise AttributeError(
                f"Raiden backend does not export {current_name} or "
                f"{legacy_name}")
        return method(*args)

    def register_send(self, req_id: str, uuid: int,
                      block_ids: list[int]) -> int:
        return int(
            self._call_backend("notify_for_read", "register_send", req_id,
                               int(uuid), list(block_ids)))

    def submit_load(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint: str,
        remote_block_ids: list[int],
        local_block_ids: list[int],
    ) -> int:
        return int(
            self._call_backend("start_read", "submit_load", req_id, int(uuid),
                               remote_endpoint, list(remote_block_ids),
                               list(local_block_ids)))

    def poll_finished(self) -> tuple[list[str], list[str], list[str]]:
        return self._call_backend("complete_read", "poll_finished")
