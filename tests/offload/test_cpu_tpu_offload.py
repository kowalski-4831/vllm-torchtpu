# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the TPU↔CPU KV cache offloading stack.

Covers the pieces that don't require a real TPU:
- `_resolve_kv_cache_dtype` strictness (raise on "auto") and
  `get_kv_cache_shape` probe-path placeholder semantics in
  `tpu_inference.layers.vllm.attention`.
- `TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes` size formula scales
  proportionally with `KV_H2D_POOL_MAX_BLOCKS`.
- `TPUCPUOffloadingSpec.prewarm_shapes` / `prewarm_shape` delegate to the
  H2D handler when `_tpu_handlers` is populated; no-op before.
- `Transfer` dataclass defaults and the error-capture wrapping pattern
  used by `_h2d_task` / `_d2h_task`.
- `transfer_async` H2D chunking math: `chunks_total = ceil(len(src_ids)
  / chunk_size)` at the boundary cases.
- `tpu_worker._estimate_kv_connector_hbm_reserve` dynamic spec resolution
  (no kv_connector / missing metadata / unimportable module → 0).

Heavy dependencies (real Pallas kernels, torch_tpu DMA, vllm config
plumbing) are mocked at construction time so the tests run on a host
without a TPU.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestResolveKvCacheDtype(unittest.TestCase):
    """attention._resolve_kv_cache_dtype contract."""

    def test_passes_through_torch_dtype(self):
        from tpu_inference.layers.vllm.attention import _resolve_kv_cache_dtype
        self.assertEqual(_resolve_kv_cache_dtype(torch.bfloat16),
                         torch.bfloat16)

    def test_resolves_known_strings(self):
        from tpu_inference.layers.vllm.attention import _resolve_kv_cache_dtype
        self.assertEqual(_resolve_kv_cache_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(_resolve_kv_cache_dtype("fp8_e4m3"),
                         torch.float8_e4m3fn)
        self.assertEqual(_resolve_kv_cache_dtype("half"), torch.half)

    def test_auto_raises(self):
        from tpu_inference.layers.vllm.attention import _resolve_kv_cache_dtype
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            _resolve_kv_cache_dtype("auto")

    def test_unknown_raises(self):
        from tpu_inference.layers.vllm.attention import _resolve_kv_cache_dtype
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _resolve_kv_cache_dtype("not-a-dtype")


class TestGetKvCacheShapeProbe(unittest.TestCase):
    """OffloadingConnectorWorker probes get_kv_cache_shape without a dtype
    to discover the num_blocks dimension. The placeholder return must
    preserve num_blocks at position 0 — see
    vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py
    register_kv_caches."""

    def test_auto_returns_placeholder_with_num_blocks_at_dim_0(self):
        from tpu_inference.layers.vllm.attention import PallasAttentionBackend
        shape = PallasAttentionBackend.get_kv_cache_shape(num_blocks=1234,
                                                          block_size=16,
                                                          num_kv_heads=1,
                                                          head_size=256)
        self.assertEqual(shape[0], 1234)
        self.assertEqual(shape.index(1234), 0)

    def test_concrete_dtype_returns_real_shape(self):
        from tpu_inference.layers.vllm.attention import PallasAttentionBackend
        shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks=2048,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            cache_dtype_str="bfloat16")
        self.assertEqual(shape[0], 2048)
        self.assertEqual(shape[1], 16)  # block_size


class TestEstimateHbmReserveBytes(unittest.TestCase):
    """TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes scales linearly
    with both num_layers and KV_H2D_POOL_MAX_BLOCKS."""

    @staticmethod
    def _make_vllm_config(num_layers=32,
                          num_kv_heads=8,
                          head_size=128,
                          block_size=16):
        cfg = MagicMock()
        cfg.model_config.get_num_layers.return_value = num_layers
        cfg.model_config.get_num_kv_heads.return_value = num_kv_heads
        cfg.model_config.get_head_size.return_value = head_size
        cfg.model_config.dtype = "bfloat16"
        cfg.cache_config.block_size = block_size
        cfg.cache_config.cache_dtype = "auto"  # resolved via model_dtype
        return cfg

    def test_doubling_pool_doubles_reserve(self):
        from tpu_inference.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "1024"}):
            half = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            full = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        self.assertEqual(full, 2 * half)

    def test_doubling_layers_doubles_reserve(self):
        from tpu_inference.offload.cpu_tpu import TPUCPUOffloadingSpec
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            small = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=32))
            large = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=64))
        self.assertEqual(large, 2 * small)

    def test_pool_blocks_rounds_to_next_power_of_2(self):
        from tpu_inference.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "1500"}):
            reserve_1500 = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            reserve_2048 = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        # 1500 rounds up to 2048
        self.assertEqual(reserve_1500, reserve_2048)


class TestPrewarmDelegation(unittest.TestCase):
    """TPUCPUOffloadingSpec.prewarm_shapes / prewarm_shape forward to the
    H2D handler. Before get_handlers() runs (_tpu_handlers is None) they
    are no-ops so the runner's collective_rpc doesn't crash."""

    def test_prewarm_shapes_none_when_handlers_not_built(self):
        from tpu_inference.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = None
        self.assertEqual(spec.prewarm_shapes, [])
        spec.prewarm_shape(2048)  # must not raise

    def test_prewarm_shapes_delegates_to_h2d_handler(self):
        from tpu_inference.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = MagicMock()
        spec._tpu_handlers.cpu_to_gpu_handler.prewarm_shapes = [1, 2, 4, 8]
        self.assertEqual(spec.prewarm_shapes, [1, 2, 4, 8])

    def test_prewarm_shape_forwards_argument(self):
        from tpu_inference.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = MagicMock()
        spec.prewarm_shape(512)
        spec._tpu_handlers.cpu_to_gpu_handler.prewarm_shape.\
            assert_called_once_with(512)


class TestTransferDataclass(unittest.TestCase):
    """Transfer's chunk-state fields and error-capture pattern."""

    def test_defaults(self):
        from tpu_inference.offload.cpu_tpu import Transfer
        t = Transfer(job_id=1, num_bytes=100, n=10, dma_done=None)
        self.assertIsNone(t.host_buffer)
        self.assertIsNone(t.device_buffer)
        self.assertIsNone(t.dst_ids)
        self.assertIsNone(t.src_ids)
        self.assertEqual(t.chunks_total, 0)
        self.assertEqual(t.chunks_done, 0)
        self.assertIsNone(t.dst_ids_i32)
        self.assertIsNone(t.error)

    def test_error_capture_pattern(self):
        """The wrapper pattern in _h2d_task / _d2h_task sets t.error
        before re-raising so get_finished can report success=False."""
        from tpu_inference.offload.cpu_tpu import Transfer
        t = Transfer(job_id=1, num_bytes=100, n=10, dma_done=None)
        try:
            try:
                raise RuntimeError("simulated DMA failure")
            except BaseException as e:
                t.error = e
                raise
        except RuntimeError:
            pass
        self.assertIsNotNone(t.error)
        self.assertIsInstance(t.error, RuntimeError)
        self.assertEqual(str(t.error), "simulated DMA failure")


class TestChunkingMath(unittest.TestCase):
    """transfer_async H2D branch computes
    `chunks_total = max(1, ceil(len(src_ids) / chunk_size))`. Exercise
    boundary cases at the arithmetic level without spinning up a real
    handler (which would need a TPU device for the staging buffer)."""

    @staticmethod
    def _chunks(n_ids: int, chunk_size: int) -> int:
        return max(1, (n_ids + chunk_size - 1) // chunk_size)

    def test_single_chunk_when_fits(self):
        for n in (0, 1, 100, 2047, 2048):
            self.assertEqual(self._chunks(n, 2048), 1, f"n={n}")

    def test_multi_chunk_when_oversized(self):
        self.assertEqual(self._chunks(2049, 2048), 2)
        self.assertEqual(self._chunks(4096, 2048), 2)
        self.assertEqual(self._chunks(4097, 2048), 3)
        self.assertEqual(self._chunks(5120, 2048), 3)

    def test_smaller_pool_more_chunks(self):
        self.assertEqual(self._chunks(2048, 512), 4)
        self.assertEqual(self._chunks(2049, 512), 5)


class TestEstimateKvConnectorHbmReserve(unittest.TestCase):
    """tpu_worker._estimate_kv_connector_hbm_reserve dynamically resolves
    the configured spec via kv_connector_extra_config without hardcoding
    connector or spec class names."""

    @staticmethod
    def _make_worker(kv_transfer_config):
        from tpu_inference.worker.tpu_worker import TPUWorker
        worker = TPUWorker.__new__(TPUWorker)
        worker.vllm_config = MagicMock()
        worker.vllm_config.kv_transfer_config = kv_transfer_config
        return worker

    def test_no_kv_connector_returns_zero(self):
        worker = self._make_worker(kv_transfer_config=None)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)

    def test_missing_spec_metadata_returns_zero(self):
        """A kv_connector without spec_name / spec_module_path keys
        (e.g., a P/D disagg connector that doesn't use OffloadingConnector's
        factory mechanism) gets a zero reserve — its budget must not be
        shrunk."""
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {}
        worker = self._make_worker(kv_transfer_config=kv_tc)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)

    def test_unimportable_spec_module_returns_zero(self):
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {
            "spec_name": "Nonexistent",
            "spec_module_path": "does.not.exist.module",
        }
        worker = self._make_worker(kv_transfer_config=kv_tc)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)

    def test_spec_without_estimate_method_returns_zero(self):
        """Even if the spec class resolves, returning 0 when it doesn't
        advertise an HBM reserve preserves the duck-typing contract."""
        kv_tc = MagicMock()
        # Point at this very test module — has no estimate_hbm_reserve_bytes
        # classmethod, so duck-typing falls through to 0.
        kv_tc.kv_connector_extra_config = {
            "spec_name": "TestEstimateKvConnectorHbmReserve",
            "spec_module_path": "tests.offload.test_cpu_tpu_offload",
        }
        worker = self._make_worker(kv_transfer_config=kv_tc)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)


if __name__ == "__main__":
    unittest.main()
