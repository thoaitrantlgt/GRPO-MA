# Metadata

Training and evaluation manifests are **not** stored in this repository to keep the source tree small. Download them from the companion HuggingFace dataset:

```bash
# From the repo root:
huggingface-cli download whcpumpkin/GRPO-MA-data \
    --repo-type dataset \
    --local-dir metadata \
    --include "*.json"
```

After downloading, this directory should contain (at minimum):

```
metadata/
├── grpo_sharerobot_trajectory_train.json
└── grpo_sharerobot_trajectory_test.json
```

> The HuggingFace dataset URL above is a placeholder pending public release.
> If you already have access to the metadata files via another channel, drop
> them here under the same filenames.

## Schema

Each manifest is a JSON list of records. Every record has at least:

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Prompt shown to the model. |
| `answer` | `str` / list / number | Ground-truth answer; format depends on the task. |
| `question_type` | `str` | Must match `TASK_CONFIG["task_type"]` of a registered task module. |
| `image` *(optional)* | `str` / list | Path(s) relative to the YAML manifest's `data_root`. |
| `video` *(optional)* | `str` / list | Path(s) relative to `data_root`. |

Example entry (`grpo_sharerobot_trajectory_train.json`):

```json
{
  "id": 4852,
  "image": "rtx_frames_success_38/62_robo_set#episode_12881/frame_0.png",
  "question": "...",
  "question_type": "trajectory-sharerobot",
  "answer": [[128, 121], [176, 116], [228, 94], [262, 45]]
}
```

The dataset YAML (`scripts/train/grpo_trajectory.yaml`) points at one of these JSON files via `json_path`. See the top-level README's "Adding a new task" section for the full workflow.
