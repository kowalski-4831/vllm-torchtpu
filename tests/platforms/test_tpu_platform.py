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

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig

from tpu_inference.platforms.tpu_platform import TpuPlatform


class TestTpuPlatform:

    @pytest.fixture
    def vllm_config(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = MagicMock(spec=ModelConfig)
        vllm_config.model_config.dtype = torch.bfloat16
        vllm_config.model_config.is_hybrid = False
        vllm_config.cache_config = MagicMock(spec=CacheConfig)
        vllm_config.cache_config.block_size = None
        vllm_config.compilation_config = MagicMock()
        vllm_config.compilation_config.mode = MagicMock()
        vllm_config.compilation_config.compile_sizes = [16, 32]
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_batched_tokens = 2048
        vllm_config.scheduler_config.is_multimodal_model = False
        vllm_config.speculative_config = None
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {}
        return vllm_config

    @patch("tpu_inference.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "tpu_inference.platforms.tpu_platform.TpuPlatform._initialize_sharding_config"
    )
    @patch(
        "tpu_inference.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    @patch("tpu_inference.platforms.tpu_platform.vllm_envs")
    def test_check_and_update_config_hybrid_block_size(
            self, mock_vllm_envs, mock_update_dp, mock_prepare_env,
            mock_sharding, mock_apply_patches, vllm_config):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.model_config.is_hybrid = True
        vllm_config.cache_config.block_size = 123  # already set

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 999
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules', {
                    'tpu_inference.layers.vllm.attention':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }):
            TpuPlatform.check_and_update_config(vllm_config)

        # Verify block_size wasn't overridden by get_page_size
        assert vllm_config.cache_config.block_size == 123
        # And get_page_size shouldn't even be called because is_hybrid is True
        mock_pallas.get_page_size.assert_not_called()
