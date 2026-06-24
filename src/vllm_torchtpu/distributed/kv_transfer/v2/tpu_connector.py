from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_hma import (
    TPUConnectorHMA, TPUConnectorHMAScheduler, TPUConnectorHMAWorker)

from .common import LayerType, TensorLayout
from .layout import HeadSegment, KVCacheRegion, TokenFirstLayoutSpec
from .metadata import (ConnectorMetadataV2, HeadMapping, KVParallelLayout,
                       LocalDecodeAllocation, PullMeta, RankTransferPlan,
                       SourceBlockRef, StridedSegmentOp, TpKVTopology)
from .pcp_policy import PcpReshardingPolicy, PcpTokenTransfer
from .planner import ContiguousHeadTPTransferPlanner, TPTransferPlanner


class TPUConnectorV2Scheduler(TPUConnectorHMAScheduler):
    """Scheduler wrapper that reuses the current HMA scheduler semantics."""


class TPUConnectorV2Worker(TPUConnectorHMAWorker):
    """Worker wrapper that reuses the current HMA worker data path."""

    def build_pull_meta(
        self,
        metadata: ConnectorMetadataV2,
        topology: TpKVTopology,
        planner: TPTransferPlanner,
    ) -> PullMeta:
        return planner.build_pull_meta(metadata, topology)

    def lower_transfer_plan(
        self,
        metadata: ConnectorMetadataV2,
        topology: TpKVTopology,
        pull_meta: PullMeta,
        destination: LocalDecodeAllocation,
        planner: TPTransferPlanner,
    ) -> dict[int, RankTransferPlan]:
        return planner.lower(metadata, topology, destination, pull_meta)


class TPUConnectorV2(TPUConnectorHMA):
    """V2 shell with HMA-compatible lifecycle behavior.

    This class intentionally wires V2 scheduler/worker subclasses while all
    lifecycle methods are inherited from TPUConnectorHMA. That makes the file
    usable as a drop-in experiment without changing the existing connector.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        **kwargs: Any,
    ):
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUConnectorV2Scheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = TPUConnectorV2Worker(vllm_config)
        else:
            raise ValueError(f"Invalid role: {role}")


TPUConnector = TPUConnectorV2
TPUConnectorScheduler = TPUConnectorV2Scheduler
TPUConnectorWorker = TPUConnectorV2Worker

__all__ = [
    "ConnectorMetadataV2",
    "ContiguousHeadTPTransferPlanner",
    "HeadSegment",
    "HeadMapping",
    "KVCacheRegion",
    "KVParallelLayout",
    "LayerType",
    "LocalDecodeAllocation",
    "PullMeta",
    "PcpReshardingPolicy",
    "PcpTokenTransfer",
    "RankTransferPlan",
    "SourceBlockRef",
    "StridedSegmentOp",
    "TPTransferPlanner",
    "TPUConnector",
    "TPUConnectorScheduler",
    "TPUConnectorV2",
    "TPUConnectorV2Scheduler",
    "TPUConnectorV2Worker",
    "TPUConnectorWorker",
    "TensorLayout",
    "TokenFirstLayoutSpec",
    "TpKVTopology",
]
