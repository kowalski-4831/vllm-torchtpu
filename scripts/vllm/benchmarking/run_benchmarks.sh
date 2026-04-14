#!/bin/bash
# Self-contained benchmark runner for torchtpu-vllm.
# Starts a vLLM server, runs benchmark_serving.py for each ISL/OSL x concurrency
# combo, saves results locally.
#
# Usage:
#   ./scripts/vllm/benchmarking/run_benchmarks.sh [--config CONFIG_NAME] [--dry-run]
#
# Examples:
#   ./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-0.6b-smoke
#   ./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep
#
# Available configs (in scripts/vllm/benchmarking/configs/):
#   qwen3-coder-480b-fp8-tp8-ep  - Primary target (TP=8, EP, FP8)
#   qwen3-30b-fp8-tp4            - Smaller MoE model (TP=4, FP8)
#   qwen3-0.6b-smoke             - Quick smoke test (TP=1, no quant)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# =============================================================================
# Parse arguments
# =============================================================================
CONFIG_NAME="qwen3-coder-480b-fp8-tp8-ep"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            echo "*** DRY RUN MODE: commands will be printed but not executed ***"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config CONFIG_NAME] [--dry-run]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Load config
# =============================================================================
CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    find "$SCRIPT_DIR/configs/" -name '*.sh' -print0 2>/dev/null | xargs -0 -I{} basename {} .sh
    exit 1
fi

# Defaults (config can override)
MODEL=""
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="512:512"
CONCURRENCY_OPTIONS="1"
MAX_NUM_BATCHED_TOKENS=""
RANDOM_RANGE_RATIO=""

# shellcheck source=/dev/null
source "$CONFIG_FILE"

if [ -z "$MODEL" ]; then
    echo "ERROR: Config must set MODEL"
    exit 1
fi
if [ -z "$RANDOM_RANGE_RATIO" ]; then
    echo "ERROR: Config must set RANDOM_RANGE_RATIO"
    exit 1
fi

PORT="${PORT:-8000}"

# Verify vllm bench serve is available
if ! python3 -m vllm.entrypoints.cli.main bench serve --help &>/dev/null; then
    echo "ERROR: 'vllm bench serve' not available. Is vLLM installed?"
    exit 1
fi
echo "Using benchmark tool: vllm bench serve"

# =============================================================================
# Setup results directory
# =============================================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
safe_model=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
RESULTS_DIR="$REPO_DIR/benchmark_runs/${safe_model}_tp${TENSOR_PARALLELISM}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
log_file="$RESULTS_DIR/benchmark.log"

# =============================================================================
# Helper functions
# =============================================================================
ceil_scale_by_ratio() {
    local val=$1
    local ratio=$2
    python3 -c "import math; print(math.ceil(${val} * (1.0 + ${ratio})))"
}

start_vllm_server() {
    local max_model_len=$1
    local max_num_batched_tokens=$2
    local max_num_seqs=$3
    local gpu_mem_util=0.95

    # Qwen models use lower memory utilization
    local model_lower
    model_lower=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    if echo "$model_lower" | grep -q "qwen"; then
        gpu_mem_util=0.8
    fi

    # Build extra args
    local extra_args=""
    if [ "$ENABLE_EP" = "true" ]; then
        extra_args="--enable-expert-parallel"
    fi
    if [ -n "$QUANTIZATION" ]; then
        extra_args="$extra_args --quantization $QUANTIZATION"
    fi

    export PYTHONUNBUFFERED=1
    export MODEL_IMPL_TYPE=vllm

    local server_cmd="vllm serve --model=${MODEL} --tensor-parallel-size=$TENSOR_PARALLELISM --data-parallel-size=$DATA_PARALLELISM --max-model-len=$max_model_len --max-num-batched-tokens=$max_num_batched_tokens --max-num-seqs=$max_num_seqs --port $PORT --no-async-scheduling --no-enable-prefix-caching --gpu-memory-utilization=$gpu_mem_util $extra_args"

    echo ""
    echo "================================================"
    echo "Starting vLLM server"
    echo "  Model: $MODEL"
    echo "  TP=$TENSOR_PARALLELISM DP=$DATA_PARALLELISM EP=$ENABLE_EP"
    echo "  max_model_len=$max_model_len"
    echo "  max_num_batched_tokens=$max_num_batched_tokens"
    echo "  max_num_seqs=$max_num_seqs"
    echo "  gpu_mem_util=$gpu_mem_util"
    echo "================================================"
    echo "[cmd] $server_cmd"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] Skipping server launch"
        return
    fi

    $server_cmd >> "$RESULTS_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    echo "Server started (pid=$SERVER_PID)"
    tail -f "$RESULTS_DIR/server.log" &
    TAIL_PID=$!

    # Wait for server to be ready (up to 90 minutes for large model compilation)
    local max_wait=$((90 * 60))
    local waited=0
    while ! curl -s -o /dev/null --connect-timeout 1 "http://localhost:$PORT/health" 2>/dev/null; do
        if [ "$waited" -ge "$max_wait" ]; then
            echo "ERROR: Server did not become ready within 90 minutes."
            stop_vllm_server
            exit 1
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: Server process died during startup."
            echo "========== Last 200 lines of server.log =========="
            tail -200 "$RESULTS_DIR/server.log"
            echo "==================================================="
            exit 1
        fi
        if [ "$waited" -gt 0 ] && [ $((waited % 60)) -eq 0 ]; then
            echo "Waiting for server... (${waited}s elapsed)"
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "Server ready on port $PORT (${waited}s)"
}

stop_vllm_server() {
    if [ "$DRY_RUN" = "1" ]; then
        return
    fi
    echo ""
    echo "Stopping vLLM server..."
    pkill -TERM -f "vllm serve" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
    sleep 2
    rm -f /tmp/libtpu_lockfile*
    kill "$TAIL_PID" 2>/dev/null || true
    echo "Server stopped."
}

# =============================================================================
# Compute server parameters from config
# =============================================================================
max_seq_len=0
for config in $ISL_OSL_CONFIGS; do
    IFS=':' read -r input_len output_len <<< "$config"
    worst_input_len=$(ceil_scale_by_ratio "$input_len" "$RANDOM_RANGE_RATIO")
    worst_output_len=$(ceil_scale_by_ratio "$output_len" "$RANDOM_RANGE_RATIO")
    total=$((worst_input_len + worst_output_len))
    if [ "$total" -gt "$max_seq_len" ]; then
        max_seq_len=$total
    fi
done
max_model_len=$max_seq_len
max_batched_tokens=$max_model_len

# Cap max_batched_tokens if configured
if [ -n "$MAX_NUM_BATCHED_TOKENS" ] && [ "$MAX_NUM_BATCHED_TOKENS" -gt 0 ] 2>/dev/null; then
    if [ "$max_batched_tokens" -gt "$MAX_NUM_BATCHED_TOKENS" ]; then
        echo "Capping max_batched_tokens: $max_batched_tokens -> $MAX_NUM_BATCHED_TOKENS"
        max_batched_tokens=$MAX_NUM_BATCHED_TOKENS
    fi
fi

max_concurrency=0
for c in $CONCURRENCY_OPTIONS; do
    if [ "$c" -gt "$max_concurrency" ]; then max_concurrency=$c; fi
done

# Save config metadata
cat > "$RESULTS_DIR/config.json" << EOF
{
    "model": "$MODEL",
    "tensor_parallelism": $TENSOR_PARALLELISM,
    "data_parallelism": $DATA_PARALLELISM,
    "enable_ep": $ENABLE_EP,
    "quantization": "${QUANTIZATION:-none}",
    "isl_osl_configs": "$ISL_OSL_CONFIGS",
    "concurrency_options": "$CONCURRENCY_OPTIONS",
    "max_model_len": $max_model_len,
    "max_num_batched_tokens": $max_batched_tokens,
    "timestamp": "$TIMESTAMP"
}
EOF

# =============================================================================
# Run benchmarks
# =============================================================================
echo ""
echo "================================================"
echo "Benchmark Configuration"
echo "  Config: $CONFIG_NAME"
echo "  Model: $MODEL"
echo "  TP=$TENSOR_PARALLELISM DP=$DATA_PARALLELISM EP=$ENABLE_EP"
echo "  ISL/OSL: $ISL_OSL_CONFIGS"
echo "  Concurrency: $CONCURRENCY_OPTIONS"
echo "  Results: $RESULTS_DIR"
echo "================================================"

{
    echo "========================================"
    echo "START TIME: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "MODEL: $MODEL"
    echo "CONFIG: $CONFIG_NAME"
    echo "TP: $TENSOR_PARALLELISM  DP: $DATA_PARALLELISM  EP: $ENABLE_EP"
    echo "========================================"
} > "$log_file"

# Use uniform random MoE routing for consistent benchmarking
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random

trap stop_vllm_server EXIT
start_vllm_server "$max_model_len" "$max_batched_tokens" "$max_concurrency"

exit_code=0

for config in $ISL_OSL_CONFIGS; do
    IFS=':' read -r input_len output_len <<< "$config"
    total=$((input_len + output_len))
    if [ "$total" -gt "$max_model_len" ]; then
        echo "Skipping isl${input_len}_osl${output_len} (total=$total > max_model_len=$max_model_len)"
        continue
    fi

    echo ""
    echo "ISL/OSL: ${input_len}/${output_len}"
    echo "------------------------------------------------"

    for concurrency in $CONCURRENCY_OPTIONS; do
        base_name="isl${input_len}_osl${output_len}_c${concurrency}"
        result_file="${RESULTS_DIR}/${base_name}.json"
        bench_log="${RESULTS_DIR}/${base_name}.log"

        echo "  Concurrency=$concurrency"

        if [ "$DRY_RUN" = "1" ]; then
            echo "  [DRY RUN] Skipping"
            continue
        fi

        vllm bench serve \
            --backend vllm \
            --model "$MODEL" \
            --host localhost \
            --port "$PORT" \
            --dataset-name random \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --random-range-ratio "$RANDOM_RANGE_RATIO" \
            --num-prompts "$((concurrency * 2))" \
            --max-concurrency "$concurrency" \
            --request-rate inf \
            --save-result \
            --ignore-eos \
            --result-filename "$result_file" \
            --seed 42 \
            --temperature 0 2>&1 | tee -a "$bench_log"

        bench_exit=${PIPESTATUS[0]}
        if [ "$bench_exit" -eq 0 ]; then
            echo "    OK -> $result_file"
        else
            echo "    FAILED (exit $bench_exit)"
            exit_code=$bench_exit
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "ERROR: Server died during benchmark"
                tail -50 "$RESULTS_DIR/server.log"
                break 2
            fi
        fi
    done
done

# =============================================================================
# Summary
# =============================================================================
{
    echo ""
    echo "========================================"
    echo "END TIME: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "EXIT CODE: $exit_code"
    echo "========================================"
} >> "$log_file"

echo ""
echo "================================================"
if [ "$exit_code" -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "Benchmark failed (exit code: $exit_code)"
fi
echo "Results: $RESULTS_DIR"
echo "================================================"

exit "$exit_code"
