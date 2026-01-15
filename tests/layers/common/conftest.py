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
Pytest configuration for MXFP4 requantization tests.

Provides fixtures for TPU device testing.
"""

import pytest
import torch


def pytest_addoption(parser):
    """Add --use-tpu command line option."""
    parser.addoption(
        "--use-tpu",
        action="store_true",
        default=False,
        help="Run tests on TPU device instead of CPU",
    )


@pytest.fixture
def device(request):
    """Get the device to run tests on (CPU or TPU).
    
    For JAX: Already uses TPU automatically when available.
    For PyTorch: Uses torch_tpu.api.tpu_device() when --use-tpu flag is set.
    
    Usage:
        pytest tests/layers/common/ -v              # Run on CPU
        pytest tests/layers/common/ -v --use-tpu   # Run on TPU
    """
    use_tpu = request.config.getoption("--use-tpu")
    if use_tpu:
        try:
            from torch_tpu import api
            return api.tpu_device()
        except ImportError:
            pytest.skip("torch_tpu not available")
        except Exception as e:
            pytest.skip(f"TPU not available: {e}")
    return torch.device("cpu")
