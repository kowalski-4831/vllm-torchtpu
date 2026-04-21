from types import SimpleNamespace

import pytest

from tpu_inference.platforms import tpu_platform


class _FakeExecutionModeModule:

    def __init__(self, member_names: list[str], default_name: str):
        members = {name: SimpleNamespace(name=name) for name in member_names}
        self.EagerMode = SimpleNamespace(__members__=members)
        self._current_mode = self.EagerMode.__members__[default_name]

    def get_eager_mode(self):
        return self._current_mode

    def set_eager_mode(self, mode):
        self._current_mode = mode


def test_resolve_torchtpu_eager_mode():
    execution_mode = _FakeExecutionModeModule(
        ["DEFER_AND_FUSE", "DEFER_NEVER"],
        "DEFER_NEVER",
    )

    mode = tpu_platform._resolve_torchtpu_eager_mode("defer_and_fuse",
                                                     execution_mode)

    assert mode.name == "DEFER_AND_FUSE"


def test_resolve_torchtpu_eager_mode_rejects_unsupported_mode():
    execution_mode = _FakeExecutionModeModule(
        ["DEFER_AND_FUSE", "DEFER_NEVER"],
        "DEFER_NEVER",
    )

    with pytest.raises(
            RuntimeError,
            match="TPU_TORCH_TPU_EAGER_MODE=.*defer_never_and_launch_blocking"
    ):
        tpu_platform._resolve_torchtpu_eager_mode(
            "defer_never_and_launch_blocking", execution_mode)


def test_configure_torchtpu_eager_mode_logs_effective_mode(monkeypatch):
    execution_mode = _FakeExecutionModeModule(
        ["DEFER_AND_FUSE", "DEFER_NEVER"],
        "DEFER_NEVER",
    )
    log_messages: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(tpu_platform, "_torchtpu_eager_mode_configured", False)
    monkeypatch.setattr(tpu_platform, "_get_torchtpu_execution_mode_module",
                        lambda: execution_mode)
    monkeypatch.setattr(tpu_platform.envs,
                        "TPU_TORCH_TPU_EAGER_MODE",
                        "defer_and_fuse",
                        raising=False)
    monkeypatch.setattr(tpu_platform.logger, "info",
                        lambda msg, *args: log_messages.append((msg, args)))

    tpu_platform._configure_torchtpu_eager_mode()

    assert execution_mode.get_eager_mode().name == "DEFER_AND_FUSE"
    assert log_messages == [(
        "TorchTPU eager mode configured: %s (policy=%s, previous=%s)",
        ("DEFER_AND_FUSE", "defer_and_fuse", "DEFER_NEVER"),
    )]
