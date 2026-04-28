import os

from vllm.utils.network_utils import get_ip

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# For multi-host usage only, to collect IP and port for all nodes.
_NODES_KV_IP_PORT = dict()


def set_node_kv_ip_port(ip_port: tuple[int, str, int]):
    global _NODES_KV_IP_PORT
    node_id, ip, port = ip_port
    _NODES_KV_IP_PORT[node_id] = (ip, port)


def get_kv_ips() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        num_nodes = len(_NODES_KV_IP_PORT)
        ips = []
        for node_id in range(num_nodes):
            ips.append(_NODES_KV_IP_PORT[node_id][0])
        return ips
    else:
        return get_host_ip()


def get_kv_ports() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        num_nodes = len(_NODES_KV_IP_PORT)
        ports = []
        for node_id in range(num_nodes):
            ports.append(_NODES_KV_IP_PORT[node_id][1])
        return ports
    else:
        return get_kv_transfer_port()


def get_host_ip() -> str:
    """Use `VLLM_HOST_IP` if set, otherwise use default network interface IP."""
    return get_ip()


def get_kv_transfer_port() -> str:
    port = os.getenv("TPU_KV_TRANSFER_PORT", "9100")
    return port


def get_side_channel_port() -> str:
    port = os.getenv("TPU_SIDE_CHANNEL_PORT", "9600")
    return port


def get_node_id() -> int:
    # TODO(xiang): Is it possible to get this from a pre-defiend env?
    id = os.getenv("TPU_NODE_ID", 0)
    return int(id)


def get_transfer_channel_number() -> int:
    """Parallel TCP stream count for the coord-mode KV wire transfer.

    0 (default) means auto: coord path uses tp_size channels, one per
    producer rank shard, bound on consecutive ports base..base+tp_size-1.
    A non-zero override is clamped to tp_size; ranks are round-robined
    onto channels by ``rank % n_channels``."""
    n = os.getenv("TPU_KV_TRANSFER_CHANNEL_NUMBER", "0")
    return int(n)


def get_p2p_wait_pull_timeout() -> int:
    """KV-cache transfer timeout in seconds."""
    timeout_str = os.getenv("TPU_P2P_WAIT_PULL_TIMEOUT", "30")
    return int(timeout_str)


def get_kv_shm_pool_gb() -> float:
    """Total GB budget for the per-host shared-memory KV staging pool.

    The slot count is derived at runtime: ``num_slots = max(1, floor(
    budget_bytes / per_slot_bytes))``. Each slot holds one request's
    worth of KV data across all TP ranks (~1GB for typical 70B-class
    models at max_model_len=8192, TP=8)."""
    gb_str = os.getenv("TPU_KV_SHM_POOL_GB", "128")
    return float(gb_str)


def get_ipc_socket_path(node_id: int) -> str:
    """Per-host intra-process IPC endpoint used between rank-0 coordinator
    and the other TP ranks."""
    # One socket per host id; workers on the same host share it.
    prefix = os.getenv("TPU_IPC_SOCKET_DIR", "/tmp")
    return f"ipc://{prefix}/tpu_conn_node{node_id}.sock"


def get_shm_name(node_id: int) -> str:
    """Name for the per-host SharedMemory block. Scoped by host id so
    a single host can run multiple disjoint nodes if ever needed."""
    return f"tpu_conn_kv_node{node_id}"


def get_kv_warmup_enabled() -> bool:
    """Whether to pre-compile KV gather/scatter XLA executables at startup.

    Without warmup, the first PULL on the consumer side can time out because
    the producer is blocked compiling index_select / index_put_ for shapes
    it has never seen before."""
    enable_str = os.getenv("TPU_KV_WARMUP_ENABLED", "true").lower()
    return enable_str in ("true", "1", "yes")


def get_kv_latency_log_interval() -> float:
    """How often (seconds) to log aggregated KV transfer latency stats.
    Set to 0 to disable periodic summaries (per-request lines still print)."""
    val_str = os.getenv("TPU_KV_LATENCY_LOG_INTERVAL", "30")
    try:
        return float(val_str)
    except ValueError:
        return 30.0
