# SPDX-License-Identifier: Apache-2.0
"""Real-TPU round-trip test for the cpu_tpu offload handlers.

Exercises the actual data path of
``SingleDirectionOffloadingHandler`` -- the D2H gather -> DMA -> host-pool
scatter (store) and the H2D load -> device scatter (reload) -- on real TPU
tensors, WITHOUT bringing up an LLM. It is the lightweight guardrail for
the offload store/load correctness: a block stored to the host pool and
reloaded into a different GPU slot must come back bit-identical.

Unlike an end-to-end ``LLM`` test this uses only eager TPU ops (index
gather/scatter + host<->device ``copy_``); there is no ``torch.compile`` /
EngineCore bring-up, so it runs in well under a second once the device is
up.
"""
import time

import pytest
import torch

try:
    import torch_tpu  # noqa: F401
    _HAS_TPU = True
except Exception:  # pragma: no cover - import guard
    _HAS_TPU = False

pytestmark = pytest.mark.skipif(not _HAS_TPU,
                                reason="requires torch_tpu / a TPU device")

NUM_BLOCKS = 16
BLOCK_ELEMS = 64  # per-block trailing size (flattened)
DTYPE = torch.float32


def _make_kv(device):
    """KV cache as one [num_blocks, block_elems] tensor; block i is filled
    with the constant value (i + 1) so each block is trivially identifiable."""
    vals = torch.arange(1, NUM_BLOCKS + 1, dtype=DTYPE).reshape(NUM_BLOCKS, 1)
    return vals.expand(NUM_BLOCKS, BLOCK_ELEMS).contiguous().to(device)


def test_offload_store_reload_roundtrip_preserves_block_data():
    from vllm.v1.kv_offload.base import GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    from vllm_torchtpu.offload.cpu_tpu import SingleDirectionOffloadingHandler

    dev = torch.device("tpu")

    kv = _make_kv(dev)
    kv_orig = kv.detach().clone()
    cpu_pool = torch.zeros(NUM_BLOCKS, BLOCK_ELEMS, dtype=DTYPE)  # host pool

    store_src_blocks = [3, 7]  # GPU blocks to offload
    host_slots = [0, 1]  # destination host-pool slots
    reload_dst_blocks = [10, 12]  # fresh GPU slots to reload into

    d2h = SingleDirectionOffloadingHandler([kv], [cpu_pool], 1, 1)
    h2d = SingleDirectionOffloadingHandler([cpu_pool], [kv], 1, 1)
    try:
        # --- STORE: gather GPU blocks -> host pool (D2H) ---
        assert d2h.transfer_async(1, (GPULoadStoreSpec(
            store_src_blocks, group_sizes=[2],
            block_indices=[0]), CPULoadStoreSpec(host_slots)))
        d2h.wait({1})  # blocks until gather + DMA + host scatter land
        d2h.get_finished()  # pop + release buffers

        assert torch.equal(cpu_pool[host_slots[0]], kv_orig[3].cpu())
        assert torch.equal(cpu_pool[host_slots[1]], kv_orig[7].cpu())

        # Overwrite the source GPU blocks to prove the reload comes from the
        # host pool, not the (now-clobbered) original slots.
        kv[torch.tensor(store_src_blocks, device=dev)] = -999.0

        # --- RELOAD: host pool -> fresh GPU slots (H2D) ---
        h2d._stash_req_id_for_job(2, "roundtrip-req")
        assert h2d.transfer_async(
            2, (CPULoadStoreSpec(host_slots),
                GPULoadStoreSpec(
                    reload_dst_blocks, group_sizes=[2], block_indices=[0])))
        # H2D defers its scatter to flush_pending_scatters; drive both until
        # the destination slots reflect the reloaded data.
        deadline = time.time() + 30.0
        while time.time() < deadline:
            h2d.get_finished()
            h2d.flush_pending_scatters({"roundtrip-req"})
            if torch.equal(kv[reload_dst_blocks[0]].cpu(), kv_orig[3].cpu()):
                break
            time.sleep(0.02)

        assert torch.equal(kv[reload_dst_blocks[0]].cpu(), kv_orig[3].cpu()), \
            "reloaded block 0 does not match the originally-stored data"
        assert torch.equal(kv[reload_dst_blocks[1]].cpu(), kv_orig[7].cpu()), \
            "reloaded block 1 does not match the originally-stored data"
    finally:
        d2h.shutdown()
        h2d.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
