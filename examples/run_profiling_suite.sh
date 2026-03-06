#!/bin/bash
set -o pipefail

# Text Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Define Models
MODELS=(
    "Qwen/Qwen3-0.6B"
    # "openai/gpt-oss-20b"
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct"
)

# 2. Define Configurations
# Format: "batch_size input_len output_len"
CONFIGS=(
    "8 1024 1024"
    # "8 1024 8192"
    # "8 8192 1024"
)

TP_SIZE=4

# 3. Common Args
COMMON_ARGS=()

# 4. Execution Loop
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED_EXPERIMENTS=()

echo -e "${GREEN}Starting Profiling Suite - Start Time: $TIMESTAMP${NC}"
echo "--------------------------------------------------------"

for model in "${MODELS[@]}"; do
    for config in "${CONFIGS[@]}"; do
        # read splits the config string into variables
        read -r batch_size input_len output_len <<< "$config"
        max_model_len=$((input_len + output_len))

        # Sanitize model name (replace / with _) for folder naming
        # e.g. Qwen/Qwen3-0.6B -> Qwen_Qwen3-0.6B
        safe_model_name="${model//\//_}"

        # Create informative directory name
        # Structure: profiles/{MODEL_NAME}_bs{BS}_in{IN}_out{OUT}_{TIMESTAMP}
        output_dir="profiles/${safe_model_name}_bs${batch_size}_in${input_len}_out${output_len}_${TIMESTAMP}"

        echo -e "${CYAN}Running Experiment:${NC}"
        echo "  Model:       $model"
        echo "  Config:      BS=$batch_size, In=$input_len, Out=$output_len"
        echo "  MaxLen:      $max_model_len"
        echo "  Artifacts:   $output_dir"

        # Run command
        if MODEL_IMPL_TYPE="vllm" python3 examples/tpu_profiling.py \
            --model "$model" \
            --batch-size "$batch_size" \
            --input-len "$input_len" \
            --output-len "$output_len" \
            --max-model-len "$max_model_len" \
            --profile-result-dir "$output_dir" \
            --tensor-parallel-size "$TP_SIZE" \
            "${COMMON_ARGS[@]}"; then
            echo -e "${GREEN}[SUCCESS] Experiment completed.${NC}"
        else
            echo -e "${RED}[FAILURE] Experiment failed!${NC}"
            # Record failure
            FAILED_EXPERIMENTS+=("$model (BS=$batch_size, In=$input_len, Out=$output_len)")
        fi

        echo "--------------------------------------------------------"
    done
done

echo ""
if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo -e "${GREEN}All experiments finished successfully!${NC}"
else
    echo -e "${RED}Some experiments failed:${NC}"
    for fail in "${FAILED_EXPERIMENTS[@]}"; do
        echo -e "  - $fail"
    done
    exit 1
fi
