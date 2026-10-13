# Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO

Official implementation of our ICML 2026 paper
**[Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO](https://arxiv.org/abs/2509.24494)**.

Hongcheng Wang*, Yinuo Huang*, Sukai Wang, Guanghui Ren&dagger;, Hao Dong&#9742;
&nbsp;&nbsp;(*equal contribution &nbsp;&dagger;project lead &nbsp;&#9742;corresponding author)

Peking University &nbsp;·&nbsp; PKU&ndash;Agibot Joint Lab &nbsp;·&nbsp; UESTC &nbsp;·&nbsp; Agibot

📄 Paper: <https://arxiv.org/abs/2509.24494> &nbsp;|&nbsp;
🐙 Repo: <https://github.com/whcpumpkin/GRPO-MA> &nbsp;|&nbsp;
✉ Contact: Hongcheng Wang (`whcpumpkin@gmail.com`), Hao Dong (`hao.dong@pku.edu.cn`)

---

## Core idea

We cast a tree-style branching point as splitting a generation into a **shared context** and one or more **continuations**, and apply the multivariate delta method to the shared-context advantage estimator under GRPO-style group normalization. This reveals a sampling-dimension asymmetry: scaling thoughts `K` alone leaves a strictly positive variance floor, whereas scaling continuations `M` per shared context drives the leading-order estimation variance to zero at rate `1/M`. Continuation-level branching is therefore *principled and potentially necessary*, not a heuristic. This bound is an algebraic invariance of the estimator, so it applies to any semantic breakpoint — not just the thought–answer split.

**GRPO-MA** is the lightweight instantiation we use to test the theory: place the breakpoint at `</think>` / `</analysis>` and sample `M` post-thought continuations per thought, with otherwise minimal changes to standard GRPO.

```
   prompt
     │
     ├── thought_1 ──► answer_{1,1}, answer_{1,2}, …, answer_{1,M}
     ├── thought_2 ──► answer_{2,1}, …
     │
     └── thought_K ──► answer_{K,1}, …, answer_{K,M}
```

## Repository scope

This is a **reference implementation** of the GRPO-MA training framework. The paper evaluates eight task families; this `v0.1.0` release ships the framework plus one canonical reference task (ShareRobot trajectory prediction). Other tasks are planned follow-ups.

| Paper task | Released in this repo | Status |
|---|:---:|---|
| Trajectory prediction (ShareRobot) | ✅ | Reference task with full data manifest, rewards, eval metrics |
| Math (AIME, GSM8K) | ❌ | Planned |
| Code (LiveBench, HumanEval) | ❌ | Planned |
| Object Detection (AgiBot World) | ❌ | Planned |
| Affordance (UMD, AGD20K) | ❌ | Planned |
| Demand Prediction | ❌ | Planned |
| OCR-based VQA (InfoVQA, STVQA, DocVQA) | ❌ | Planned |
| Simulator-based Manipulation | ❌ | Planned |

The training framework is task-agnostic; new tasks plug in via `task/TEMPLATE.py` (see [Adding a new task](#adding-a-new-task)).

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Adding a new task](#adding-a-new-task)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

```bash
git clone https://github.com/whcpumpkin/GRPO-MA.git
cd GRPO-MA
conda create -n grpo-ma python=3.10
conda activate grpo-ma
pip install -r requirements.txt
pip install -e .
# Optional: flash-attn for faster training
# pip install flash-attn==2.7.4 --no-build-isolation
```

The pinned versions in `requirements.txt` reflect the exact setup used for the paper experiments (PyTorch 2.5.1 + CUDA 12.4, `transformers==4.51.3`, `trl==0.19.0`, `peft==0.15.2`). A different toolchain may work but is not validated.

Sanity check:

```bash
python scripts/smoke_test.py
```

This verifies the task registry, reward routing, and metric dispatch without touching a GPU or downloading any model weights.

## Quick start

### 1. Download the image dataset

```bash
mkdir data && cd data
git clone https://huggingface.co/datasets/BAAI/ShareRobot
cd ..
```

### 2. Download the training/evaluation metadata

To keep this repo small, the JSON manifests live in a companion HuggingFace dataset (see [`metadata/README.md`](metadata/README.md) for details and schema):

```bash
huggingface-cli download whcpumpkin/GRPO-MA-data \
    --repo-type dataset \
    --local-dir metadata \
    --include "*.json"
```

This populates `metadata/grpo_sharerobot_trajectory_{train,test}.json`.

### 3. Download the base model

```bash
mkdir pretrained_weights && cd pretrained_weights
git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
cd ..
```

### 4. Run training

The runner expects a machine with `N` GPUs; the paper used `4×H100`. The script auto-detects `N` via `nvidia-smi`.

```bash
export THINK_NUM=4       # K thoughts per prompt
export ANSWER_NUM=4      # M post-thought continuations per thought
bash scripts/run_grpo_lora.sh
```

With `K=M=4` on 4 GPUs, each GPU draws `K=4` thoughts and `K*M=16` continuations from the same prompt per step (the synchronized sampler hands the same prompt to every rank), so one optimizer step processes `4×K=16` thoughts and `4×K×M=64` continuations globally for that prompt. See `scripts/README.md` for every knob exposed via environment variables.

## Project structure

```
GRPO-MA/
├── config/                 # GRPOConfig, GRPOScriptArguments, GRPOModelConfig
├── dataset/
│   └── dataset_grpo.py     # YAML-driven multi-image/video GRPO dataset
├── metadata/                  # JSON manifests (downloaded from HF, not in repo)
│   └── README.md              # Where to fetch them and the schema
├── model/
│   ├── qwen_module.py             # Qwen2.5-VL wrapper
│   ├── vlm_module.py              # Abstract VLM interface
│   ├── qwen2_5vl_monkey_patch.py  # Flash-attn / DS-Zero3 / torch.load patches
│   ├── reward_func.py             # Routes (completion, sol, question_type) → task rewards
│   └── task_configs.py            # Reads task templates from the registry
├── scripts/
│   ├── run_grpo_lora.sh           # Training entry point
│   ├── eval/eval.py               # Multi-task evaluation driver
│   ├── eval/run_eval_trajectory.sh
│   ├── smoke_test.py              # No-GPU sanity check
│   ├── train/grpo_trajectory.yaml # Training data manifest
│   └── zero2.json                 # DeepSpeed ZeRO-2 config
├── task/
│   ├── task_loader.py             # Auto-discovers task modules under task/
│   ├── processing_utils.py        # Coordinate-rescaling helpers
│   ├── trajectory_sharerobot.py   # The reference task
│   └── TEMPLATE.py                # Copy this when adding a new task
├── trainer/
│   └── grpo_ma_trainer.py         # GRPOMATrainer (K×M sampling, dual advantages)
├── train.py                       # CLI entry point
└── setup.py
```

## Training

`scripts/run_grpo_lora.sh` is the canonical entry point. The most important knobs:

| Variable | Meaning | Default |
|---|---|---|
| `THINK_NUM` | `K`: thoughts per prompt | `4` |
| `ANSWER_NUM` | `M`: post-thought continuations per thought | `4` |
| `BETA` | KL penalty weight | `0.04` |
| `MODEL_NAME_OR_PATH` | Path or HF id of the base VLM | `pretrained_weights/Qwen2.5-VL-3B-Instruct` |
| `DATASET_NAME` | Data manifest (YAML) | `scripts/train/grpo_trajectory.yaml` |
| `RUN_NAME` | Output folder prefix | `Qwen2.5-VL-3B-GRPO-lora-trajectory` |
| `MAX_PIXELS` | Upper bound on input image pixels | `1003520` |

Set `GRPO_MA_VERBOSE=1` to re-enable per-step diagnostic prints (sampler indices, generated completions, sign-inconsistent counts). They are silenced by default to keep training logs readable. Full hyperparameter documentation is in [`scripts/README.md`](scripts/README.md).

The training runs reported in the paper used `K=M=4`, `lr=1e-5`, LoRA rank `64`, frozen vision tower, DeepSpeed ZeRO-2, bf16, FlashAttention-2, and a `</think>` stop string.

## Evaluation

The evaluation driver is task-agnostic. Each task module owns its scoring via a `compute_metrics(predictions, **kwargs)` callable that is auto-registered by `task/task_loader.py`.

```bash
python scripts/eval/eval.py \
    --model_name_or_path pretrained_weights/Qwen2.5-VL-3B-Instruct \
    --adapter_path output/<run>/checkpoint-<step> \
    --dataset metadata/grpo_sharerobot_trajectory_test.json \
    --image_root data/ShareRobot/trajectory/images \
    --output_dir output/eval/<run>
```

Or use the convenience wrapper:

```bash
ADAPTER_PATH=output/<run>/checkpoint-<step> bash scripts/eval/run_eval_trajectory.sh
```

Outputs:

```
output/eval/<run>/
├── predictions.jsonl    # One record per test item with raw + parsed prediction
└── metrics.json         # Aggregated metrics, grouped by task_type
```

For the trajectory task this reports DFD, Hausdorff distance, RMSE, endpoint distance, and format compliance (matching Table 4 in the paper).

To add evaluation for a new task: implement `compute_metrics(...)` in the task module — see `task/trajectory_sharerobot.py` for the reference implementation.

## Adding a new task

The framework auto-discovers anything you drop into `task/`, so a new task is just one file plus a metadata manifest.

### 1. Prepare dataset samples

Place media (images, videos) under `data/` and create a metadata JSON/JSONL inside `metadata/`. Each record needs at minimum:

- `question`: the prompt shown to the model
- `answer`: the ground-truth answer (string, number, JSON, etc.)
- `question_type`: must match `TASK_CONFIG["task_type"]` in your task module
- Optional media fields: `image`, `video`, plus any task-specific attributes

Example (`metadata/my_task_train.json`):

```json
[
  {"question": "Which tool is highlighted?", "answer": "hammer", "image": "train/toolbox_001.jpg", "question_type": "my-task"},
  {"question": "Locate the object.", "answer": "[120, 48, 256, 220]", "image": "train/toolbox_002.jpg", "question_type": "my-task"}
]
```

### 2. Describe the dataset with a YAML manifest

```yaml
# scripts/train/grpo_my_task.yaml
datasets:
  - json_path: metadata/my_task_train.json
    sampling_strategy: "all"
    data_root: data/my_task/images
    data_modality: image
```

### 3. Implement the task module

```bash
cp task/TEMPLATE.py task/my_task.py
```

Edit:

- `TASK_CONFIG["task_type"]` — unique lowercase-with-hyphens identifier, matching `question_type`.
- `TASK_CONFIG["grpo_template"]` — prompt template. `{Question}` is replaced with the question text. Other config keys (`description`, `input_format`, etc.) are logged but do not affect training.
- `format_reward(completion, sol, **kwargs)` — cheap structural check (returns 0.0 or 1.0).
- `accuracy_reward(completion, sol, **kwargs)` — real task metric. May return `float` or `(float, dict)`; the dict is logged as auxiliary metrics during training.
- `process_answer(...)` *(optional)* — rescale pixel-space answers when the image is resized (use `task.processing_utils.scale_trajectory` / `scale_bbox_xyxy` / `scale_points`).
- `compute_metrics(predictions, **kwargs)` *(optional)* — used by the eval driver.

`register(...)` is auto-called by the task loader; leave it as-is.

### 4. Verify and train

```bash
python -c "from task import print_summary; print_summary()"
python scripts/smoke_test.py
```

```bash
export DATASET_NAME=scripts/train/grpo_my_task.yaml
export RUN_NAME=Qwen2.5-VL-3B-GRPO-lora-my-task
bash scripts/run_grpo_lora.sh
```

## Reproducibility notes

- All paper experiments used `Qwen2.5-VL-3B-Instruct` (also 7B for the scaling study) with LoRA (rank 64, alpha 128, dropout 0.05) on a 4×H100 80GB node.
- We did not use vLLM during training; generation runs through `model.generate` inside the trainer.
- Random seed: 42 (set via `--data_seed` and `set_seed`).
- The `warmup_ratio` is `0.0` in the paper. The trainer has logic that forces `M=1` during warmup (degrading GRPO-MA to GRPO); this is unused with `warmup_ratio=0.0`.
- The released ShareRobot trajectory metadata uses `id`, `image`, `question`, `question_type`, `answer` fields. Images are loaded relative to the YAML `data_root`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wang2026grpoma,
  title     = {Why Tree-Style Branching Matters for Thought Advantage Estimation in {GRPO}},
  author    = {Wang, Hongcheng and Huang, Yinuo and Wang, Sukai and Ren, Guanghui and Dong, Hao},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  year      = {2026},
  url       = {https://arxiv.org/abs/2509.24494}
}
```

## Acknowledgements

The trainer is built on top of the open-source GRPO implementation in [HuggingFace TRL](https://github.com/huggingface/trl). Vision-language model wrappers and the Flash-Attention monkey patches build on [HuggingFace Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft). The DAPO-style clip-higher and several GRPO stabilizers we compare against were proposed by their respective authors; see the paper for full citations. The ShareRobot dataset is released by [BAAI](https://huggingface.co/datasets/BAAI/ShareRobot).

## License

Released under the [Apache License 2.0](LICENSE). Portions derived from third-party Apache-2.0 projects are attributed in [NOTICE](NOTICE).


## Local research proposals

After cloning, run `python -m experiments.restore_artifacts` from repository root if you need the large archived gradient lattice. Its original bytes are stored as checksum-verified parts to fit GitHub's file limit. Git LFS is not required.

The [sequential thought-sampling proposal](SEQUENTIAL_THOUGHT_SAMPLING_PROPOSAL.md) records its completed CPU feasibility and K4 control STOP decisions. The subsequent [gradient-aware headroom diagnostic](refine-logs/gradient-headroom/20261011_run02/DECISION.md) evaluates fixed synthetic score geometries with exact true-p allocation oracles; its gradient-specific development screen also remains unmet. [Formulation and implementation](experiments/gradient_headroom/README.md). The next [server score-geometry plan](refine-logs/score-geometry/SERVER_RUN_PLAN.md) has runnable preflight/prompt-freeze commands; score export and target measurement are pending. CPU experiments and evidence from the previous JADA proposal are collected in [archive/jada_proposal](archive/jada_proposal/README.md), including instructions for running their checks.
