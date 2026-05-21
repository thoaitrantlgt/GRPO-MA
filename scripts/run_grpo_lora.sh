export DEBUG_MODE="${DEBUG_MODE:-false}"
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$(pwd)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_train_${TIMESTAMP}"
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Get variables from environment with default values
RUN_NAME="${RUN_NAME:-Qwen2.5-VL-3B-GRPO-lora-trajectory}"
TRAIN_CLS="${TRAIN_CLS:-GRPO-MA}"
BETA="${BETA:-0.04}"
ANSWER_NUM="${ANSWER_NUM:-4}"
THINK_NUM="${THINK_NUM:-4}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
DATASET_NAME="${DATASET_NAME:-scripts/train/grpo_trajectory.yaml}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-pretrained_weights/Qwen2.5-VL-3B-Instruct}"
TASK_TYPE="${TASK_TYPE:-think}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"

# Configure NCCL settings for distributed training
# export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export LOG_PATH="./debug_log/$RUN_NAME/${TIMESTAMP}_ans${ANSWER_NUM}.txt"

mkdir -p debug_log/$RUN_NAME/${TIMESTAMP}_thi${THINK_NUM}_ans${ANSWER_NUM}_task${TASK_TYPE}
mkdir -p output/$RUN_NAME/${TIMESTAMP}_thi${THINK_NUM}_ans${ANSWER_NUM}_task${TASK_TYPE}

# Build torchrun command arguments
TORCHRUN_ARGS=(
    --nproc_per_node="$NPROC_PER_NODE"
    --nnodes="1"
    --node_rank="0"
    --master_addr="127.0.0.1"
    --master_port="12346"
    train.py
    --deepspeed scripts/zero2.json
    --output_dir output/$RUN_NAME/${TIMESTAMP}_thi${THINK_NUM}_ans${ANSWER_NUM}_task${TASK_TYPE}
    --model_name_or_path $MODEL_NAME_OR_PATH
    --dataset_name $DATASET_NAME
    --image_root ./data
    --max_prompt_length 1024
    --max_completion_length 1024
    --train_cls $TRAIN_CLS
    --num_think_samples $THINK_NUM
    --num_answers_per_thinking $ANSWER_NUM
    --num_generations $THINK_NUM
    --per_device_train_batch_size 1
    --num_iterations $NUM_ITERATIONS
    --gradient_accumulation_steps 1
    --logging_steps 1
    --torch_dtype bfloat16
    --data_seed 42
    --report_to tensorboard
    --gradient_checkpointing true
    --attn_implementation flash_attention_2
    --num_train_epochs 1
    --run_name $RUN_NAME
    --save_steps 100
    --save_only_model true
    --learning_rate 1e-5
    --use_peft true
    --lora_r 64
    --lora_alpha 128
    --lora_dropout 0.05
    --lora_task_type CAUSAL_LM
    --freeze_vision_modules true
    --beta $BETA
    --epsilon_high 0.28
    --warmup_ratio 0.0
    --task_type $TASK_TYPE
    --max_pixels $MAX_PIXELS
    --stop_strings "</think>"
)

# Ensure log directory exists and write output to LOG_PATH while also printing to terminal
mkdir -p "$(dirname "$LOG_PATH")"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Command: torchrun ${TORCHRUN_ARGS[*]}" | tee -a "$LOG_PATH"

# Run and tee both stdout and stderr to the log, preserve original exit code
torchrun "${TORCHRUN_ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"
EXIT_CODE=${PIPESTATUS[0]}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Exit code: $EXIT_CODE" | tee -a "$LOG_PATH"
exit $EXIT_CODE