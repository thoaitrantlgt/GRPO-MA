# GRPO-MA Training Script Parameters

This document explains all hyperparameters and configuration options in the `run_grpo_lora.sh` script.

## Environment Variables

### Core Configuration

- **`RUN_NAME`** (default: `Qwen2.5-VL-3B-GRPO-lora-trajectory`)
  - Experiment name for identifying this training run
  - Used in output directories and logging

- **`TRAIN_CLS`** (default: `GRPO-MA`)
  - Training class/algorithm to use
  - Specifies the training methodology

- **`MODEL_NAME_OR_PATH`** (default: `pretrained_weights/Qwen2.5-VL-3B-Instruct`)
  - Path to the pretrained model or HuggingFace model identifier
  - Base model to fine-tune

- **`DATASET_NAME`** (default: `scripts/train/grpo_trajectory.yaml`)
  - Path to the dataset configuration file
  - Defines training data sources

### GRPO-Specific Parameters

- **`BETA`** (default: `0.04`)
  - Controls the strength of KL divergence penalty
  - Lower values = more conservative updates

- **`ANSWER_NUM`** (default: `4`)
  - Number of answer candidates to generate per thinking step
  - More answers = better exploration but higher compute cost

- **`THINK_NUM`** (default: `4`)
  - Number of thinking/reasoning samples to generate
  - Controls trajectory diversity

- **`NUM_ITERATIONS`** (default: `1`)
  - Number of GRPO iterations per training step
  - More iterations = more thorough optimization per batch

- **`TASK_TYPE`** (default: `think`)
  - Reserved CLI knob for future think/no-think ablations; currently has no
    effect on training.

### Vision Model Parameters

- **`MAX_PIXELS`** (default: `1003520`)
  - Maximum number of pixels for input images used by Qwen2.5-VL model

### Distributed Training

- **`NPROC_PER_NODE`** (auto-detected)
  - Number of GPUs per node
  - Automatically detected via `nvidia-smi`

- **`NCCL_BLOCKING_WAIT`** = `1`
  - NCCL synchronization mode
  - Ensures blocking waits for better stability

### Debugging

- **`DEBUG_MODE`** = `true`
  - Enables debug logging and verbose output

- **`LOG_PATH`**
  - Path where training logs are saved
  - Format: `./debug_log/$RUN_NAME/${TIMESTAMP}_ans${ANSWER_NUM}.txt`

## Training Arguments

### Model & Data

- **`--deepspeed`**: `scripts/zero2.json`
  - DeepSpeed ZeRO-2 configuration for memory optimization

- **`--output_dir`**: Output directory for checkpoints and logs

- **`--image_root`**: `./data`
  - Root directory containing training images

- **`--max_prompt_length`**: `1024`
  - Maximum token length for input prompts

- **`--max_completion_length`**: `1024`
  - Maximum token length for generated completions

- **`--num_generations`**: Same as `THINK_NUM`
  - Number of generation samples per iteration

### Batch & Optimization

- **`--per_device_train_batch_size`**: `1`
  - Batch size per GPU device
  - Small value due to memory constraints with large vision models

- **`--gradient_accumulation_steps`**: `1`
  - Number of steps to accumulate gradients before updating
  - Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`

- **`--num_train_epochs`**: `1`
  - Number of complete passes through the training data

- **`--learning_rate`**: `1e-5`
  - Learning rate for optimizer
  - Relatively small for stable fine-tuning

- **`--warmup_ratio`**: `0.0`
  - Fraction of training steps for learning rate warmup
  - 0.0 means no warmup
  - During warmup, M is forcibly set to 1, causing GRPO-MA to degrade into GRPO. 
  - In the paper, the warmup_ratio is set to 0, meaning this parameter is not used.

### LoRA Configuration

- **`--use_peft`**: `true`
  - Enable Parameter-Efficient Fine-Tuning (PEFT)

- **`--lora_r`**: `64`
  - LoRA rank/dimension
  - Higher rank = more expressive but more parameters

- **`--lora_alpha`**: `128`
  - LoRA scaling parameter
  - Typically set to 2× the rank

- **`--lora_dropout`**: `0.05`
  - Dropout rate for LoRA layers
  - Helps prevent overfitting

- **`--lora_task_type`**: `CAUSAL_LM`
  - Type of task for LoRA adaptation

- **`--freeze_vision_modules`**: `true`
  - Keep vision encoder frozen during training
  - Only fine-tune language components

### GRPO Algorithm

- **`--beta`**: Same as `BETA` environment variable
  - KL penalty coefficient

- **`--epsilon_high`**: `0.28`
  - Upper threshold for advantage clipping
  - Controls maximum policy update magnitude
  - This trick comes from DAPO.

### Compute & Memory

- **`--torch_dtype`**: `bfloat16`
  - Mixed precision training format
  - Saves memory while maintaining numerical stability

- **`--gradient_checkpointing`**: `true`
  - Trade computation for memory
  - Enables training larger models by recomputing activations

- **`--attn_implementation`**: `flash_attention_2`
  - Use Flash Attention 2 for efficient attention computation
  - Significantly speeds up training

### Logging & Checkpointing

- **`--logging_steps`**: `1`
  - Log metrics every N steps
  - 1 = log every step (verbose)

- **`--save_steps`**: `100`
  - Save checkpoint every N steps

- **`--save_only_model`**: `true`
  - Only save model weights, not optimizer states
  - Reduces checkpoint size

- **`--report_to`**: `tensorboard`
  - Logging backend for metrics visualization
  - Wandb is also supported.

### Miscellaneous

- **`--data_seed`**: `42`
  - Random seed for data shuffling
  - Ensures reproducibility

- **`--stop_strings`**: `"</think>"`
  - Token sequences that trigger generation stopping
  - Used to end thinking steps

## Directory Structure

```
output/$RUN_NAME/${TIMESTAMP}_thi${THINK_NUM}_ans${ANSWER_NUM}_task${TASK_TYPE}/
  ├── checkpoints/        # Model checkpoints
  └── logs/              # Training logs

debug_log/$RUN_NAME/${TIMESTAMP}_thi${THINK_NUM}_ans${ANSWER_NUM}_task${TASK_TYPE}/
  └── ${TIMESTAMP}_ans${ANSWER_NUM}.txt  # Detailed debug logs
```

## Usage Example

```bash
# Use default parameters
bash scripts/run_grpo_lora.sh

# Override specific parameters
RUN_NAME="my-experiment" BETA=0.05 ANSWER_NUM=8 bash scripts/run_grpo_lora.sh

# Train with custom model
MODEL_NAME_OR_PATH="path/to/model" bash scripts/run_grpo_lora.sh
```

## Notes

- The script automatically detects the number of available GPUs
- All paths are relative to the project root directory
- Logs are timestamped to prevent overwriting previous runs
- DeepSpeed ZeRO-2 is used for memory-efficient distributed training