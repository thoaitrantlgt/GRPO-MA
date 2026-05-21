#!/usr/bin/env bash
# Convenience wrapper: evaluate a (LoRA) checkpoint on the ShareRobot
# trajectory test set and report DFD / HD / RMSE / EndPoint distance.

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-pretrained_weights/Qwen2.5-VL-3B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATASET="${DATASET:-metadata/grpo_sharerobot_trajectory_test.json}"
IMAGE_ROOT="${IMAGE_ROOT:-data/ShareRobot/trajectory/images}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval/$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
TEMPERATURE="${TEMPERATURE:-0.0}"

CMD=(
    python scripts/eval/eval.py
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --dataset "$DATASET"
    --image_root "$IMAGE_ROOT"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --max_pixels "$MAX_PIXELS"
    --temperature "$TEMPERATURE"
    --task_type trajectory-sharerobot
)

if [[ -n "$ADAPTER_PATH" ]]; then
    CMD+=(--adapter_path "$ADAPTER_PATH")
fi

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Running: ${CMD[*]}"
"${CMD[@]}"
