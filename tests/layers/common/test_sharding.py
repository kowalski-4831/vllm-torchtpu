from vllm_torchtpu.layers.common.sharding import ShardingAxisName


def test_attention_axes_match_single_worker_mesh():
    assert ShardingAxisName.ATTN_DATA is None
    assert ShardingAxisName.ATTN_HEAD == "model"
