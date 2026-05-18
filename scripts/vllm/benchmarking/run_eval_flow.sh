#!/bin/bash
# Full evaluation flow script: supports manual and CI execution
set -euo pipefail

CONFIG_NAME=""
RESULTS_DIR=""
RUN_LM_EVAL=0
RUN_EVALPLUS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --run-lm-eval)
            RUN_LM_EVAL=1
            shift
            ;;
        --run-evalplus)
            RUN_EVALPLUS=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: --config is required"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="/tmp/perf_eval_$CONFIG_NAME"
fi
mkdir -p "$RESULTS_DIR"

EVAL_BASELINE="scripts/vllm/benchmarking/baselines/eval/$CONFIG_NAME.baseline.json"
PERF_BASELINE="scripts/vllm/benchmarking/baselines/perf/$CONFIG_NAME.baseline.json"
EVALPLUS_BASELINE="scripts/vllm/benchmarking/baselines/evalplus/$CONFIG_NAME.baseline.json"

EVAL_LOG="$RESULTS_DIR/eval_check.md"
PERF_LOG="$RESULTS_DIR/regression_check.md"
EVALPLUS_LOG="$RESULTS_DIR/evalplus_check.md"

run_evalplus() {
    local model="$1"
    local results_dir="$2"
    local root="$results_dir/evalplus"
    local rc=0
    rm -rf "$root"
    mkdir -p "$root"

    # Defaults if not set in env
    local datasets="${EVALPLUS_DATASETS:-humaneval mbpp}"
    local parallel="${EVALPLUS_PARALLEL:-8}"

    for dataset in $datasets; do
        echo "EvalPlus dataset: $dataset"
        evalplus.evaluate "$dataset" \
            --model "$model" \
            --backend openai \
            --base_url http://localhost:8000/v1 \
            --root "$root" \
            --greedy \
            --n_samples 1 \
            --bs 1 \
            --parallel "$parallel" \
            2>&1 | tee "$root/$dataset.log"
        dataset_rc=${PIPESTATUS[0]}
        if [ "$dataset_rc" -ne 0 ]; then
            echo "ERROR: EvalPlus failed for $dataset (exit $dataset_rc)"
            rc="$dataset_rc"
        fi
    done
    return "$rc"
}

# ========================================================
# 1. Initial Cleanup
# ========================================================
echo "=== Initial Cleanup ==="
bash "$SCRIPT_DIR/cleanup_server.sh"

# ========================================================
# 2. Run Benchmarks (starts server and keeps it alive)
# ========================================================
echo "Running benchmarks for $CONFIG_NAME..."
./scripts/vllm/benchmarking/run_benchmarks.sh \
  --config "$CONFIG_NAME" \
  --results-dir "$RESULTS_DIR" \
  --keep-alive

# Read model name from config
MODEL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")

fail=0
evalplus_rc=0

# ========================================================
# 3. Run lm_eval
# ========================================================
if [ "$RUN_LM_EVAL" = "1" ] && [ -f "$EVAL_BASELINE" ]; then
    echo "Running lm_eval..."
    lm_eval \
      --model local-chat-completions \
      --model_args model="$MODEL",base_url=http://localhost:8000/v1/chat/completions,num_concurrent=128 \
      --tasks mmlu_llama,mmlu_pro \
      --apply_chat_template \
      --fewshot_as_multiturn true \
      --limit 100 \
      --seed "0,1234,None,1234" \
      --gen_kwargs continue_final_message=True add_generation_prompt=False 'chat_template_kwargs={"enable_thinking": False}' \
      --output_path "$RESULTS_DIR"

    echo "=== Checking Eval Regression ==="
    python3 scripts/vllm/benchmarking/check_regression.py \
      --mode eval \
      --results-dir "$RESULTS_DIR" \
      --baseline "$EVAL_BASELINE" 2>&1 | tee "$EVAL_LOG" || fail=1
fi

# ========================================================
# 4. Run EvalPlus
# ========================================================
if [ "$RUN_EVALPLUS" = "1" ]; then
    echo "=== Running EvalPlus ==="
    run_evalplus "$MODEL" "$RESULTS_DIR" || evalplus_rc=$?

    if [ "$evalplus_rc" -ne 0 ]; then
        fail=1
    fi

    if [ -f "$EVALPLUS_BASELINE" ]; then
        echo "=== Checking EvalPlus Regression ==="
        python3 scripts/vllm/benchmarking/check_regression.py \
          --mode evalplus \
          --results-dir "$RESULTS_DIR" \
          --baseline "$EVALPLUS_BASELINE" 2>&1 | tee "$EVALPLUS_LOG" || fail=1
    fi
fi

# ========================================================
# 5. Cleanup after tests
# ========================================================
echo "=== Final Cleanup ==="
bash "$SCRIPT_DIR/cleanup_server.sh"

# ========================================================
# 6. Check Perf Regression
# ========================================================
if [ -f "$PERF_BASELINE" ]; then
    echo "=== Checking Perf Regression ==="
    python3 scripts/vllm/benchmarking/check_regression.py \
      --mode perf \
      --results-dir "$RESULTS_DIR" \
      --baseline "$PERF_BASELINE" 2>&1 | tee "$PERF_LOG" || fail=1
fi

exit "$fail"
