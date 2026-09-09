# SPDX-License-Identifier: Apache-2.0
"""The pipeline wave's launch schedule, driven on CPU with a fake ring.

Every stage runs its own event stream in a thread. The fake ring behaves like
the runtime: a launch completes only when every stage has issued the same
launch number, and it hands stage r-1's payload of that launch to stage r. A
schedule that leaves the stages with unequal counts therefore deadlocks and
fails the join timeout instead of passing by accident.
"""
import random
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import pp_wave

HIDDEN = 8
MAX_ROWS = 64
BUCKETS = (8, 16, 32, 64)


class _Ring:

    def __init__(self, size):
        self.size = size
        self.count = [0] * size
        self.slots = {}
        self.cv = threading.Condition()

    def launch(self, rank, x):
        with self.cv:
            pos = self.count[rank]
            self.slots[(rank, pos)] = x.clone()
            self.count[rank] += 1
            self.cv.notify_all()
            if not self.cv.wait_for(lambda: all(c > pos for c in self.count),
                                    timeout=20):
                raise TimeoutError(f"rank {rank} launch {pos} never paired")
            # closed cycle: the first stage receives the last stage's block
            return self.slots[((rank - 1) % self.size, pos)].clone()


@pytest.fixture
def stages(monkeypatch):
    """Eight PPWave instances on CPU sharing one fake ring."""
    size = 8
    ring = _Ring(size)
    current = {}
    monkeypatch.setattr(pp_wave, "get_or_create_pp_mesh", lambda: None)
    monkeypatch.setattr(
        pp_wave, "pp_permute_op", lambda mesh:
        (lambda x, rank=current["rank"]: ring.launch(rank, x)))
    monkeypatch.setattr(pp_wave, "_launch_now", lambda op, x: op(x))
    from vllm.distributed import parallel_state
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group", lambda: SimpleNamespace(rank_in_group=current["rank"],
                                                world_size=size))
    template = {
        "hidden_states": ((HIDDEN, ), torch.float32),
        "residual": ((HIDDEN, ), torch.float32),
    }
    waves = []
    for rank in range(size):
        current["rank"] = rank
        waves.append(pp_wave.PPWave(torch.device("cpu"), MAX_ROWS, template))
    return ring, waves


def _run_stage(wave, bursts, results, errors):
    """One stage's event stream: a forward adds 1 to the hidden states it
    received and passes the residual on; the first stage seeds each forward
    with its number as hidden states and a residual of 0.5, which every
    later stage must see as part of the sum."""
    try:
        wave.warmup()
        g = 0
        for burst in bursts:
            for rows in burst:
                received = wave.start_forward(rows)
                if wave.rank == 0:
                    h = torch.full((rows, HIDDEN), float(g))
                    r = torch.full((rows, HIDDEN), 0.5)
                else:
                    assert torch.equal(received["residual"],
                                       torch.zeros(rows, HIDDEN))
                    h = received["hidden_states"] + 1
                    r = received["residual"]
                    assert h.shape == (rows, HIDDEN)
                if wave.rank == wave.last:
                    results.append((g, rows, h, r))
                    wave.end_forward(None, rows)
                else:
                    wave.end_forward({"hidden_states": h, "residual": r}, rows)
                g += 1
            wave.settle()
    except BaseException as exc:  # reported by the test thread
        errors.append((wave.rank, exc))


def test_random_bursts_deliver_every_forward_and_keep_counts_equal(stages):
    ring, waves = stages
    rng = random.Random(7)
    bursts = [[
        rng.choice(BUCKETS) for _ in range(rng.choice((1, 2, 3, 5, 9)))
    ] for _ in range(6)]
    results, errors = [], []
    threads = [
        threading.Thread(target=_run_stage,
                         args=(w, bursts, results, errors),
                         daemon=True) for w in waves
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a stage never finished"
    assert not errors, errors
    total = sum(len(b) for b in bursts)
    assert len(results) == total
    last = waves[-1].rank
    for g, rows, h, r in results:
        assert torch.equal(h, torch.full((rows, HIDDEN), g + last + 0.5))
        assert torch.equal(r, torch.zeros(rows, HIDDEN))
    assert len({w.launches for w in waves}) == 1
    assert waves[0].launches == 1 + total + len(bursts) * (len(waves) - 1)


def test_settle_without_an_open_burst_is_a_noop(stages):
    _, waves = stages
    waves[3].settle()
    assert waves[3].launches == 0


def test_a_single_forward_burst_launches_stages_plus_forwards_minus_one(
        stages):
    ring, waves = stages
    bursts = [[16]]
    results, errors = [], []
    threads = [
        threading.Thread(target=_run_stage,
                         args=(w, bursts, results, errors),
                         daemon=True) for w in waves
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert [w.launches for w in waves] == [1 + 1 + 7] * 8


def test_the_hand_off_carries_hidden_states_and_residual_only(monkeypatch):
    from vllm.distributed import parallel_state
    monkeypatch.setattr(pp_wave, "get_or_create_pp_mesh", lambda: None)
    monkeypatch.setattr(pp_wave, "pp_permute_op", lambda mesh: None)
    monkeypatch.setattr(parallel_state, "get_pp_group",
                        lambda: SimpleNamespace(rank_in_group=0, world_size=2))
    with pytest.raises(ValueError, match="hidden_states and residual"):
        pp_wave.PPWave(
            torch.device("cpu"), MAX_ROWS, {
                "hidden_states": ((HIDDEN, ), torch.float32),
                "other": ((HIDDEN, ), torch.float32),
            })
    with pytest.raises(ValueError, match="one dtype"):
        pp_wave.PPWave(
            torch.device("cpu"), MAX_ROWS, {
                "hidden_states": ((HIDDEN, ), torch.float32),
                "residual": ((HIDDEN, ), torch.bfloat16),
            })


def test_summing_before_the_hand_off_keeps_the_residual_and_rounds_the_norm_input_once(
):
    """The next stage's first norm adds hidden states and residual in
    float32, normalizes the sum and keeps it rounded as the residual. The
    hand-off sends the sum rounded to the model dtype with a zero residual,
    so the residual is identical and the norm input differs by at most one
    bfloat16 rounding."""
    # a local generator: the global seed would initialize every accelerator
    # backend in this process, and a process that opened the TPU keeps it
    generator = torch.Generator().manual_seed(0)
    hidden = (torch.randn(256, 512, generator=generator) * 4).to(
        torch.bfloat16)
    residual = (torch.randn(256, 512, generator=generator) * 4).to(
        torch.bfloat16)

    # the norm's input and new residual in a single-stage model
    norm_input = hidden.float() + residual.float()
    new_residual = norm_input.to(torch.bfloat16)

    # the same after the hand-off, as PPWave.end_forward and start_forward
    # shape it
    sent = hidden + residual
    received_norm_input = sent.float() + torch.zeros_like(sent).float()
    received_residual = received_norm_input.to(torch.bfloat16)

    assert torch.equal(received_residual, new_residual)
    assert torch.allclose(received_norm_input, norm_input, rtol=2**-8, atol=0)
