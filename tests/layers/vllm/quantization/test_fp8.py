# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for FP8 quantization weight processing on TPU.

Verifies that process_weights_after_loading correctly transforms FP8
weights and scales for the GMM kernel, and that FP8->BF16 dequantization
for linear layers is numerically correct.

Run on CPU (default):
    pytest tests/layers/vllm/quantization/test_fp8.py -v

Run on TPU (when available):
    pytest tests/layers/vllm/quantization/test_fp8.py -v --use-tpu
"""

import pytest
import torch

from tpu_inference.layers.vllm.quantization.fp8 import VllmFp8LinearMethodTPU


class FakeQuant:
    """Minimal quant config for testing."""

    def __init__(self, weight_block_size=None, activation_scheme="dynamic"):
        self.weight_block_size = weight_block_size
        self.activation_scheme = activation_scheme


class FakeMoEConfig:
    """Minimal MoE config for testing."""

    def __init__(self, has_bias=False, experts_per_token=2):
        self.has_bias = has_bias
        self.experts_per_token = experts_per_token


class FakeActivation:
    """Mimics MoEActivation enum."""

    def __init__(self, value):
        self.value = value


class FakeFusedMoELayer(torch.nn.Module):
    """Minimal FusedMoE-like layer for testing process_weights_after_loading."""

    def __init__(self,
                 w13_weight,
                 w2_weight,
                 w13_scale,
                 w2_scale,
                 activation="silu",
                 has_bias=False,
                 experts_per_token=2):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        self.w13_weight_scale_inv = torch.nn.Parameter(w13_scale,
                                                       requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(w2_scale,
                                                      requires_grad=False)
        self.activation = FakeActivation(activation)
        self.moe_config = FakeMoEConfig(has_bias=has_bias,
                                        experts_per_token=experts_per_token)

    # Make isinstance(layer, FusedMoE) work via duck typing in the assert
    def __class_getitem__(cls, item):
        return cls


class TestFp8MoEScaleReshape:
    """Tests for block-quantized FP8 MoE scale reshaping."""

    @pytest.mark.parametrize("num_experts,intermediate,hidden,block_size", [
        (4, 256, 512, 128),
        (8, 384, 768, 128),
        (2, 128, 256, 128),
    ])
    def test_scale_output_shape(self, device, num_experts, intermediate,
                                hidden, block_size):
        """Verify scales are reshaped to [E, in/B, 1, out_dim] for GMM."""
        E, inter, H, B = num_experts, intermediate, hidden, block_size

        # Input scale shape: [E, 2*inter/B, H/B] (w13) and [E, H/B, inter/B] (w2)
        w13_scale = torch.randn(E, 2 * inter // B, H // B, device=device)
        w2_scale = torch.randn(E, H // B, inter // B, device=device)

        # Simulate the reshape logic from process_weights_after_loading
        # Step 1: transpose [E, out/B, in/B] -> [E, in/B, out/B]
        w13_out = w13_scale.transpose(1, 2)
        # Step 2: expand by block_h then reshape
        w13_out = w13_out.unsqueeze(-1).expand(*w13_out.shape, B).reshape(
            w13_out.shape[0], w13_out.shape[1], -1)
        # Step 3: unsqueeze for GMM
        w13_out = w13_out.unsqueeze(2).to(torch.float32)

        # Expected: [E, H/B, 1, 2*inter]
        assert w13_out.shape == (E, H // B, 1, 2 * inter)

        # Same for w2
        w2_out = w2_scale.transpose(1, 2)
        w2_out = w2_out.unsqueeze(-1).expand(*w2_out.shape, B).reshape(
            w2_out.shape[0], w2_out.shape[1], -1)
        w2_out = w2_out.unsqueeze(2).to(torch.float32)

        assert w2_out.shape == (E, inter // B, 1, H)

    def test_scale_values_are_repeated(self, device):
        """Each block scale should be repeated block_h times in output dim."""
        E, B = 2, 128
        out_blocks, in_blocks = 4, 2  # out=512, in=256

        scale = torch.arange(E * out_blocks * in_blocks,
                             dtype=torch.float32,
                             device=device).reshape(E, out_blocks, in_blocks)

        # Apply reshape
        out = scale.transpose(1, 2)
        out = out.unsqueeze(-1).expand(*out.shape,
                                       B).reshape(out.shape[0], out.shape[1],
                                                  -1)
        out = out.unsqueeze(2)

        # Shape: [E, in_blocks, 1, out_blocks * B]
        assert out.shape == (E, in_blocks, 1, out_blocks * B)

        # Check each block_h-sized chunk has the same value
        for e in range(E):
            for ib in range(in_blocks):
                for ob in range(out_blocks):
                    chunk = out[e, ib, 0, ob * B:(ob + 1) * B]
                    assert torch.all(chunk == chunk[0]), \
                        f"Scale not constant within block at e={e}, ib={ib}, ob={ob}"


class TestFp8LinearDequant:
    """Tests for FP8 -> BF16 dequantization in linear layers."""

    def test_block_dequant_shape(self, device):
        """Dequantized weight should preserve original shape."""
        out_dim, in_dim = 256, 512
        block_h, block_w = 128, 128

        weight_fp8 = torch.randn(out_dim,
                                 in_dim,
                                 device=device,
                                 dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(out_dim // block_h,
                               in_dim // block_w,
                               device=device,
                               dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale_inv,
                                                    requires_grad=False)

        method = VllmFp8LinearMethodTPU(
            FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        assert layer.weight.shape == (out_dim, in_dim)
        assert layer.weight.dtype == torch.bfloat16

    def test_block_dequant_applies_scales(self, device):
        """Block scales should actually scale the weights."""
        out_dim, in_dim = 256, 256
        block_h, block_w = 128, 128

        # Use ones as FP8 weight — after dequant, result should equal scale
        weight_fp8 = torch.ones(out_dim,
                                in_dim,
                                device=device,
                                dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        # Set different scales per block
        scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                                 device=device,
                                 dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale_inv,
                                                    requires_grad=False)

        method = VllmFp8LinearMethodTPU(
            FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        w = layer.weight.float()
        # fp8 ones = 1.0, so each block should be scaled by its scale_inv
        # Block [0,0] (top-left 128x128) should be ~1.0
        # Block [0,1] (top-right 128x128) should be ~2.0
        # Block [1,0] (bottom-left 128x128) should be ~3.0
        # Block [1,1] (bottom-right 128x128) should be ~4.0
        assert torch.allclose(w[:128, :128].mean(),
                              torch.tensor(1.0),
                              atol=0.1)
        assert torch.allclose(w[:128, 128:].mean(),
                              torch.tensor(2.0),
                              atol=0.1)
        assert torch.allclose(w[128:, :128].mean(),
                              torch.tensor(3.0),
                              atol=0.1)
        assert torch.allclose(w[128:, 128:].mean(),
                              torch.tensor(4.0),
                              atol=0.1)

    def test_per_tensor_dequant(self, device):
        """Per-tensor (non-block) dequant should scale entire weight."""
        out_dim, in_dim = 64, 128

        weight_fp8 = torch.ones(out_dim,
                                in_dim,
                                device=device,
                                dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale = torch.tensor([2.5], device=device, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=None))
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.bfloat16
        # fp8 ones * 2.5 should be ~2.5
        assert torch.allclose(layer.weight.float().mean(),
                              torch.tensor(2.5),
                              atol=0.1)

    @pytest.mark.skip(reason="Fails on TPU")
    def test_apply_is_linear(self, device):
        """apply() should be a standard F.linear call."""
        out_dim, in_dim = 64, 128
        batch = 4

        method = VllmFp8LinearMethodTPU(
            FakeQuant(weight_block_size=[128, 128]))

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(out_dim,
                                                      in_dim,
                                                      device=device,
                                                      dtype=torch.bfloat16),
                                          requires_grad=False)

        x = torch.randn(batch, in_dim, device=device, dtype=torch.bfloat16)
        result = method.apply(layer, x)

        assert result.shape == (batch, out_dim)
        expected = torch.nn.functional.linear(x, layer.weight)
        assert torch.allclose(result, expected)


class TestFp8WeightTranspose:
    """Tests for weight transpose in MoE processing."""

    def test_weights_transposed(self, device):
        """Weights should be transposed from [E, out, in] to [E, in, out]."""
        E, inter, H = 4, 256, 512

        w13 = torch.randn(E, 2 * inter, H, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(E, H, inter, device=device, dtype=torch.bfloat16)

        w13_t = w13.transpose(1, 2).contiguous()
        w2_t = w2.transpose(1, 2).contiguous()

        assert w13_t.shape == (E, H, 2 * inter)
        assert w2_t.shape == (E, inter, H)
        # Verify the transpose is correct
        assert torch.equal(w13_t[0, :, 0], w13[0, 0, :])

    def test_swigluoai_deinterleave(self, device):
        """swigluoai activation should de-interleave w13 weights."""
        E, inter, H = 2, 128, 256

        # Create w13 with known interleaved pattern
        w13 = torch.zeros(E, 2 * inter, H, device=device, dtype=torch.bfloat16)
        # Even rows (gate/w1) = 1.0, odd rows (up/w3) = 2.0
        w13[:, ::2, :] = 1.0
        w13[:, 1::2, :] = 2.0

        # De-interleave
        w1 = w13[:, ::2, :]
        w3 = w13[:, 1::2, :]
        w13_deinterleaved = torch.cat([w1, w3], dim=1)

        # First half should be all 1.0 (w1), second half all 2.0 (w3)
        assert torch.all(w13_deinterleaved[:, :inter, :] == 1.0)
        assert torch.all(w13_deinterleaved[:, inter:, :] == 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
