# Copyright 2026 Google LLC
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

import torch
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        MambaSpec)

from tpu_inference.runner.tpu_runner import TPUModelRunner


class DummyMamba(MambaBase):

    def __init__(self):
        super().__init__()

    def get_kv_cache_spec(self, vllm_config):
        return MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
        )

    def get_state_dtype(self):
        return (torch.bfloat16, torch.float32)

    def get_state_shape(self):
        return ((4, 128), (8, 64, 32))

    @property
    def mamba_type(self):
        return "dummy"


class TestTPURunner:

    def setup_method(self):
        self.mock_device = torch.device('cpu')

        model_config = ModelConfig(
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype=torch.bfloat16,
        )
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            max_model_len=1024,
            is_encoder_decoder=False,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            observability_config={},
            additional_config={},
        )

        self.runner = MagicMock(spec=TPUModelRunner)
        self.runner.device = self.mock_device
        self.runner.block_size = 16
        self.runner.kv_cache_dtype = torch.bfloat16
        self.runner.use_spmd = False
        self.runner.vllm_config = vllm_config
        self.runner.cache_config = vllm_config.cache_config
        self.runner._hybrid_uniform_page_size_bytes = None

        # Bind the actual methods to our mock
        self.runner._update_mamba_page_size_padded = TPUModelRunner._update_mamba_page_size_padded.__get__(
            self.runner)
        self.runner._maybe_set_num_blocks_override = TPUModelRunner._maybe_set_num_blocks_override.__get__(
            self.runner)
        self.runner.get_kv_cache_spec = TPUModelRunner.get_kv_cache_spec.__get__(
            self.runner)
        self.runner.initialize_kv_cache = TPUModelRunner.initialize_kv_cache.__get__(
            self.runner)

    @patch(
        'tpu_inference.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('tpu_inference.runner.tpu_runner.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_update_mamba_page_size_padded(self, mock_mem_info,
                                           mock_get_page_size):
        layers = {}
        for i in range(1):
            mock_attn = MagicMock(spec=Attention)
            mock_attn.num_kv_heads = 2
            mock_attn.head_size = 128
            layers[f'attn_{i}'] = mock_attn

        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        self.runner._update_mamba_page_size_padded(layers)

        # Mamba unpadded size:
        # (4 * 128 * 2) + (8 * 64 * 32 * 4) = 1024 + 65536 = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 4096 + 199680 = 203776
        assert self.runner._hybrid_uniform_page_size_bytes == 203776
        assert self.runner.cache_config.mamba_page_size_padded == 203776

        # Check that num blocks override calculation was called
        mock_mem_info.assert_called_once()
        # avail (10GB * 0.9 = 9,663,676,416) // 4096 * 4096 // 203776 = 47423
        assert self.runner.cache_config.num_gpu_blocks_override == 47423

    @patch(
        'tpu_inference.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
        return_value=(100, 16, 2, 1, 128))
    @patch(
        'tpu_inference.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('tpu_inference.runner.tpu_runner.bind_kv_cache')
    @patch('tpu_inference.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_hybrid_duplication(self, mock_input_batch,
                                                    mock_has_kv_transfer,
                                                    mock_bind_kv_cache,
                                                    mock_get_page_size,
                                                    mock_get_shape):
        num_blocks = 100
        uniform_size = 203776
        tensor_size = uniform_size * num_blocks

        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=uniform_size)
        mamba_spec = MambaSpec(block_size=16,
                               shapes=[(4, 128), (8, 64, 32)],
                               dtypes=[torch.bfloat16, torch.float32],
                               page_size_padded=uniform_size)

        layer_names = ["attn.0", "mamba.0", "mamba.1", "mamba.2"]

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.1"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.2"],
                             kv_cache_spec=mamba_spec),
        ]

        kv_cache_tensors = [
            KVCacheTensor(size=tensor_size, shared_by=layer_names)
        ]
        kv_cache_config = KVCacheConfig(num_blocks=num_blocks,
                                        kv_cache_tensors=kv_cache_tensors,
                                        kv_cache_groups=kv_cache_groups)

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)

        mock_bt = MagicMock()
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1),
                                                          dtype=torch.int32)
        mock_input_batch.return_value.block_table = [mock_bt]

        self.runner.input_batch = mock_input_batch.return_value
        self.runner.kv_caches = []
        self.runner.initialize_kv_cache(kv_cache_config)

        mock_bind_kv_cache.assert_called_once()
        created_caches = mock_bind_kv_cache.call_args[0][0]

        assert len(created_caches) == 4

        for name in ["mamba.0", "mamba.1", "mamba.2"]:
            assert isinstance(created_caches[name], tuple)
            assert len(created_caches[name]) == 2
            assert created_caches[name][0].shape == (num_blocks, 4, 128)
            assert created_caches[name][1].shape == (num_blocks, 8, 64, 32)

        assert isinstance(created_caches["attn.0"], torch.Tensor)
        assert created_caches["attn.0"].shape == (100, 16, 2, 1, 128)

    @patch('tpu_inference.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'tpu_inference.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('tpu_inference.runner.tpu_runner.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_get_kv_cache_spec_hybrid_padded_size(self, mock_mem_info,
                                                  mock_get_page_size,
                                                  mock_get_layers):
        layers = {}

        mock_attn = MagicMock(spec=Attention)
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None
        layers['attn_0'] = mock_attn

        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        mock_get_layers.return_value = layers

        # Call the method
        kv_cache_specs = self.runner.get_kv_cache_spec()

        # Mamba unpadded size: (4 * 128 * 2) + (8 * 64 * 32 * 4) = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 203776
        expected_padded_size = 203776

        # Verify specs
        assert 'attn_0' in kv_cache_specs
        attn_spec = kv_cache_specs['attn_0']
        assert isinstance(attn_spec, FullAttentionSpec)
        assert attn_spec.page_size_padded == expected_padded_size

        for i in range(3):
            layer_name = f'mamba_{i}'
            assert layer_name in kv_cache_specs
            mamba_spec = kv_cache_specs[layer_name]
            assert isinstance(mamba_spec, MambaSpec)
            assert mamba_spec.page_size_padded == expected_padded_size

    @patch('tpu_inference.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'tpu_inference.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    def test_get_kv_cache_spec_pure_attention_no_cache_config_updates(
            self, mock_get_page_size, mock_get_layers):
        mock_attn = MagicMock(spec=Attention)
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None

        layers = {'layer.0': mock_attn}
        mock_get_layers.return_value = layers

        with patch.object(self.runner,
                          '_update_mamba_page_size_padded') as mock_update:
            self.runner.get_kv_cache_spec()
            mock_update.assert_not_called()
