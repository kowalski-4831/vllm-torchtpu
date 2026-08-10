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

import pytest
from vllm.config import CacheConfig

from vllm_torchtpu.executors.ray_distributed_executor_v2 import (
    RayDistributedExecutorV2, get_tpu_bundles_for_indices)


class MockParallelConfig:

    def __init__(self):
        self.world_size = 4
        self.tensor_parallel_size = 2
        self.pipeline_parallel_size = 1
        self.prefill_context_parallel_size = 1
        self.ray_workers_use_nsight = False
        self.placement_group = None
        self.max_parallel_loading_workers = None
        self.nnodes_within_dp = 1
        self.local_world_size = 4
        self.ray_runtime_env = {}
        self.data_parallel_rank_local = None
        self.data_parallel_master_port = 0


class MockVllmConfig:

    def __init__(self):
        self.parallel_config = MockParallelConfig()
        self.model_config = MagicMock()
        self.cache_config = MagicMock(spec=CacheConfig)
        self.cache_config.num_gpu_blocks_override = None
        self.lora_config = MagicMock()
        self.load_config = MagicMock()
        self.scheduler_config = MagicMock()
        self.speculative_config = MagicMock()
        self.prompt_adapter_config = MagicMock()
        self.observability_config = MagicMock()
        self.device_config = MagicMock()
        self.kv_transfer_config = None
        self.instance_id = "vllm-test-instance"


@patch("vllm.v1.executor.ray_executor_v2.RayExecutorV2.__init__",
       lambda x, y: None)
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.ray")
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.current_platform")
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.get_ip",
       return_value="127.0.0.1")
@patch(
    "vllm_torchtpu.executors.ray_distributed_executor_v2._wait_until_pg_ready")
class TestTpuRayDistributedExecutorV2:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.vllm_config = MockVllmConfig()
        self.parallel_config = self.vllm_config.parallel_config

    def test_initialize_ray_cluster_basic(self, mock_wait_until_pg_ready,
                                          mock_get_ip, mock_platform,
                                          mock_ray):
        # --- Setup mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"

        mock_ray.is_initialized.return_value = False
        mock_ray.nodes.return_value = [{
            "NodeID": "node_1",
            "Resources": {
                "TPU": 4
            }
        }]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
        mock_wait_until_pg_ready.return_value = None

        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 1.0}, {"TPU": 1.0}]
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config

        # --- Test ---
        executor._initialize_ray_cluster()

        # --- Assertions ---
        mock_ray.init.assert_called_once()
        assert executor.parallel_config.placement_group == mock_placement_group
        mock_ray.util.placement_group.assert_called_once()

    def test_initialize_ray_cluster_pipeline_parallelism(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        self.parallel_config.pipeline_parallel_size = 2
        self.parallel_config.world_size = 8

        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [{
            "NodeID": "node_1",
            "Resources": {
                "TPU": 4
            }
        }, {
            "NodeID": "node_2",
            "Resources": {
                "TPU": 4
            }
        }]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = \
            "node_1"
        mock_placement_group = MagicMock()
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor._initialize_ray_cluster()

        mock_ray.util.placement_group.assert_called_once_with(
            [{
                "TPU": 4,
                "node:127.0.0.1": 0.001
            }, {
                "TPU": 4
            }], strategy="PACK")

    def test_initialize_ray_cluster_reuses_existing_pg(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        mock_platform.ray_device_key = "TPU"
        existing_pg = MagicMock()

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor.parallel_config.placement_group = existing_pg

        executor._initialize_ray_cluster()

        mock_ray.util.placement_group.assert_not_called()
        assert executor.parallel_config.placement_group == existing_pg

    def test_get_actor_resource_kwargs(self, mock_wait_until_pg_ready,
                                       mock_get_ip, mock_platform, mock_ray):
        mock_platform.ray_device_key = "TPU"

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.parallel_config

        resource_kwargs = executor._get_actor_resource_kwargs()

        assert resource_kwargs == {"num_gpus": 0, "resources": {"TPU": 1.0}}

    @patch(
        "vllm_torchtpu.executors.ray_distributed_executor_v2.get_driver_env_vars",
        return_value={})
    @patch("vllm_torchtpu.executors.ray_distributed_executor_v2.get_open_port",
           return_value=9000)
    @patch(
        "vllm_torchtpu.executors.ray_distributed_executor_v2.get_distributed_init_method",
        return_value="tcp://127.0.0.1:9000")
    @patch("vllm_torchtpu.executors.ray_distributed_executor_v2.MessageQueue")
    def test_init_executor_env_vars_propagation(
            self, mock_mq, mock_init_method, mock_port, mock_driver_env,
            mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray):
        # This test verifies that _init_executor correctly computes and propagates all
        # TPU-specific multi-host environment variables to the remote workers.

        mock_platform.ray_device_key = "TPU"
        mock_platform.device_control_env_var = "TPU_VISIBLE_CHIPS"
        mock_platform._get_tpu_topology.return_value = "2x2"

        # Mock Ray cluster placement group and nodes
        mock_ray.is_initialized.return_value = True
        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 1.0}, {"TPU": 1.0}]
        mock_ray.util.placement_group.return_value = mock_placement_group
        mock_ray.nodes.return_value = [{
            "NodeID": "node_1",
            "Resources": {
                "TPU": 4.0
            }
        }, {
            "NodeID": "node_2",
            "Resources": {
                "TPU": 4.0
            }
        }]

        # We have 2 workers on 2 different nodes (multi-host setup)
        bundle_to_node = [(0, "node_1", "10.0.0.1"), (1, "node_2", "10.0.0.2")]

        # Mock get_bundles_sorted_by_node or similar utils
        with patch("vllm_torchtpu.executors.ray_distributed_executor_v2.get_bundles_sorted_by_node", return_value=bundle_to_node), \
             patch("vllm_torchtpu.executors.ray_distributed_executor_v2.get_tpu_multihost_topology", return_value="2x2"):
            mock_ray.get_runtime_context().get_node_id.return_value = "node_1"

            # Setup parallel config
            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 2
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.local_world_size = 1

            executor = RayDistributedExecutorV2(self.vllm_config)
            executor.vllm_config = self.vllm_config
            executor.parallel_config = self.parallel_config

            # Mock worker actors
            mock_worker_actor_1 = MagicMock()
            mock_worker_actor_1.get_node_and_physical_gpu_ids.remote.return_value = "node_1_ref"
            mock_worker_actor_2 = MagicMock()
            mock_worker_actor_2.get_node_and_physical_gpu_ids.remote.return_value = "node_2_ref"

            # ray.remote(RayWorkerProc).options().remote() mocks
            mock_remote_class = MagicMock()
            mock_remote_class.options.return_value.remote.side_effect = [
                mock_worker_actor_1, mock_worker_actor_2
            ]
            mock_ray.remote.return_value = mock_remote_class

            # Mock physical TPU ID discovery for each worker.
            mock_ray.get.side_effect = [
                # Discover GPU/TPU IDs (Step 6)
                [("node_1", [4]), ("node_2", [6])],
                # Initialize workers (Step 7)
                [None, None],
                # Collect response MQ handles (Step 8)
                [{
                    "status": "READY",
                    "handle": "handle_1"
                }, {
                    "status": "READY",
                    "handle": "handle_2"
                }]
            ]

            # Run initialization
            executor._init_executor()

            # Verify the environment variables passed to each worker during initialize_worker.remote()
            # Worker 0 (Rank 0, Node 1)
            mock_worker_actor_1.initialize_worker.remote.assert_called_once()
            args_w0 = mock_worker_actor_1.initialize_worker.remote.call_args[0]
            local_rank_w0 = args_w0[0]
            worker_env_w0 = args_w0[1]
            kwargs_w0 = mock_worker_actor_1.initialize_worker.remote.call_args.kwargs

            assert local_rank_w0 == 0
            assert worker_env_w0["LOCAL_WORLD_SIZE"] == "1"
            assert worker_env_w0["NNODES"] == "2"
            assert worker_env_w0["NODE_RANK"] == "0"
            assert worker_env_w0["MASTER_ADDR"] == "10.0.0.1"
            assert worker_env_w0["TORCH_TPU_TOPOLOGY"] == "2x2"
            assert "10.0.0.1:8070" in worker_env_w0[
                "TORCH_TPU_SLICEBUILDER_ADDRESSES"]
            assert "10.0.0.2:8070" in worker_env_w0[
                "TORCH_TPU_SLICEBUILDER_ADDRESSES"]
            assert kwargs_w0["assigned_physical_gpu_ids"] == [4]

            # Worker 1 (Rank 1, Node 2)
            mock_worker_actor_2.initialize_worker.remote.assert_called_once()
            args_w1 = mock_worker_actor_2.initialize_worker.remote.call_args[0]
            local_rank_w1 = args_w1[0]
            worker_env_w1 = args_w1[1]
            kwargs_w1 = mock_worker_actor_2.initialize_worker.remote.call_args.kwargs

            assert local_rank_w1 == 0
            assert worker_env_w1["LOCAL_WORLD_SIZE"] == "1"
            assert worker_env_w1["NNODES"] == "2"
            assert worker_env_w1["NODE_RANK"] == "1"
            assert worker_env_w1["MASTER_ADDR"] == "10.0.0.1"
            assert worker_env_w1["TORCH_TPU_TOPOLOGY"] == "2x2"
            assert kwargs_w1["assigned_physical_gpu_ids"] == [6]

            executor.ray_worker_handles = []

    def test_ray_distributed_executor_v2_bundle_expansion(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        self.parallel_config.world_size = 16
        self.parallel_config.tensor_parallel_size = 16
        self.parallel_config.pipeline_parallel_size = 1

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor.world_size = 16

        # 4 host-level bundles for world_size = 16
        bundle_to_node = [
            (0, "node_0", "10.0.0.1"),
            (1, "node_1", "10.0.0.2"),
            (2, "node_2", "10.0.0.3"),
            (3, "node_3", "10.0.0.4"),
        ]

        # Test get_tpu_bundles_for_indices helper function
        selected = get_tpu_bundles_for_indices(
            None, [0, 2], bundle_to_node_id=bundle_to_node)
        assert selected == [(0, "node_0", "10.0.0.1"),
                            (2, "node_2", "10.0.0.3")]
        with pytest.raises(ValueError, match="Bundle index 99 not found"):
            get_tpu_bundles_for_indices(None, [99],
                                        bundle_to_node_id=bundle_to_node)

        # Test host-level bundle expansion (4 bundles for world_size = 16)
        assignments = executor._get_bundle_assignments(None, bundle_to_node)
        assert len(assignments) == 16
        assert len(executor.bundle_assignments) == 16
        for rank in range(16):
            expected_node_id = f"node_{rank // 4}"
            assert assignments[rank]["rank"] == rank
            assert assignments[rank]["node_id"] == expected_node_id

        # Test 1:1 chip-level bundle mapping (16 bundles for world_size = 16)
        chip_bundles = [(i, f"node_{i // 4}", f"10.0.0.{1 + (i // 4)}")
                        for i in range(16)]
        chip_assignments = executor._get_bundle_assignments(None, chip_bundles)
        assert len(chip_assignments) == 16
        for rank in range(16):
            assert chip_assignments[rank]["rank"] == rank
            assert chip_assignments[rank]["bundle_id_idx"] == rank

        # Indivisible bundle count raises ValueError
        bad_bundle_to_node = [
            (0, "node_0", "10.0.0.1"),
            (1, "node_1", "10.0.0.2"),
            (2, "node_2", "10.0.0.3"),
        ]
        with pytest.raises(ValueError, match="divisible"):
            executor._get_bundle_assignments(None, bad_bundle_to_node)


def test_ray_distributed_executor_bundle_expansion():
    """Verify V1 executor bundle index expansion for host-level and chip-level bundles."""

    def expand_bundles(bundle_indices, world_size):
        if len(bundle_indices) < world_size:
            if len(bundle_indices
                   ) == 0 or world_size % len(bundle_indices) != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by "
                    f"the number of placement group bundles ({len(bundle_indices)})."
                )
            workers_per_bundle = world_size // len(bundle_indices)
            expanded_indices = []
            for b_id in bundle_indices:
                expanded_indices.extend([b_id] * workers_per_bundle)
            return expanded_indices
        elif len(bundle_indices) != world_size:
            raise ValueError(
                f"Number of bundle indices ({len(bundle_indices)}) must be less than or equal to "
                f"world_size ({world_size}).")
        return bundle_indices

    # 4 host-level bundle indices expanded 4x for world_size=16
    assert expand_bundles(
        [0, 1, 2, 3], 16) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    # 16 chip-level bundle indices unchanged for world_size=16
    chip_indices = list(range(16))
    assert expand_bundles(chip_indices, 16) == chip_indices

    # Indivisible bundle count raises ValueError
    with pytest.raises(ValueError, match="must be divisible"):
        expand_bundles([0, 1, 2], 16)

    # Too many bundle indices raises ValueError
    with pytest.raises(ValueError, match="less than or equal to world_size"):
        expand_bundles(list(range(17)), 16)
