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

from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata, SpeculativeDecodingManager)


def test_speculative_decoding_manager_metadata_indices(device):
    # Mock TPUModelRunner
    mock_runner = MagicMock()
    mock_runner.device = device
    mock_runner.arange_np = np.arange(100, dtype=np.int32)
    mock_runner.num_tokens_paddings = [8, 16, 32, 64]

    # Mock CPU input IDs tensor
    # Total length is logits indices size
    # E.g., 10 scheduled tokens
    input_ids_cpu = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 dtype=torch.int32)
    mock_runner.input_ids_cpu = input_ids_cpu

    # Instantiate manager
    manager = SpeculativeDecodingManager(mock_runner)

    # Mock input batch:
    # num_draft_tokens = [3, 2] (batch_size = 2)
    # cu_num_scheduled_tokens = [4, 7] (first request has 4 scheduled tokens, second has 3)
    num_draft_tokens = np.array([3, 2], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([4, 7], dtype=np.int32)
    padded_num_reqs = 4

    metadata = manager.get_spec_decode_metadata(
        num_draft_tokens=num_draft_tokens,
        cu_num_scheduled_tokens=cu_num_scheduled_tokens,
        padded_num_reqs=padded_num_reqs)

    # Verify shapes and properties of the returned SpecDecodeMetadata
    assert isinstance(metadata, SpecDecodeMetadata)

    # Expected draft lengths cpu array
    assert np.array_equal(metadata.draft_lengths_cpu, num_draft_tokens)

    # Output tensors are copied to target device
    assert metadata.draft_token_ids.device.type == device.type
    assert metadata.draft_lengths.device.type == device.type
    assert metadata.target_logits_indices.device.type == device.type
    assert metadata.bonus_logits_indices.device.type == device.type
    assert metadata.final_logits_indices.device.type == device.type
    assert metadata.segment_ids.device.type == device.type
    assert metadata.group_indices.device.type == device.type

    # 1. Draft Lengths: num_draft_tokens padded to padded_num_reqs (4) -> [3, 2, 0, 0]
    expected_draft_lengths = torch.tensor([3, 2, 0, 0],
                                          dtype=torch.int32,
                                          device=device)
    assert torch.equal(metadata.draft_lengths, expected_draft_lengths)

    # 2. Bonus Logits Indices: [3, 6] (cu_num_sampled_tokens - 1) padded to padded_num_reqs (4) -> [3, 6, 0, 0]
    expected_bonus_indices = torch.tensor([3, 6, 0, 0],
                                          dtype=torch.int32,
                                          device=device)
    assert torch.equal(metadata.bonus_logits_indices, expected_bonus_indices)

    # 3. Segment IDs: repeat request indices [0, 0, 0, 1, 1] padded to nearest static token bucket (8)
    # Unpadded size: 3 + 2 = 5 tokens. Nearest static padding is 8.
    # Padded with total request count (2) -> [0, 0, 0, 1, 1, 2, 2, 2]
    expected_segment_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2],
                                        dtype=torch.int64,
                                        device=device)
    assert torch.equal(metadata.segment_ids, expected_segment_ids)

    # 4. Group Indices: range per segment [0, 1, 2, 0, 1] padded to static bucket (8) with 0 -> [0, 1, 2, 0, 1, 0, 0, 0]
    expected_group_indices = torch.tensor([0, 1, 2, 0, 1, 0, 0, 0],
                                          dtype=torch.int32,
                                          device=device)
    assert torch.equal(metadata.group_indices, expected_group_indices)
