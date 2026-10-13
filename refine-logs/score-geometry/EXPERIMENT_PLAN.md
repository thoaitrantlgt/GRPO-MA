# Score geometry audit plan

**Current execution plan:** user steered this task to a server run plan. Follow [SERVER_RUN_PLAN.md](SERVER_RUN_PLAN.md) and [job tracker](EXPERIMENT_TRACKER.md). The initial CPU-first design below is background; no target score extraction has run, and exporter/analyzer implementation remains pending.

**Problem:** synthetic gradient-aware headroom did not pass the pooled screen; target-task score geometry is unknown. **Date:**2026-10-11.

## Claim Map

| Question | Evidence | Block |
|---|---|---|
| What norm heterogeneity and cancellation exist on independently selected target replay? | Exact thought-token-mean score vectors, raw and centered Gram metrics with prompt intervals | B2 |
| Does measured score correspond to the trainer's thought update component? | Source mapping and weighted gradient equality under ratio1 | B1 |

No claim about allocation benefit, unknown-p pilot estimation or training GO.

## Blocks

B1 MUST: map trainer/grpo_ma_trainer.py thought masks (shift start/end minus1), first continuation strideM, mean token loss then mean thoughts. At ratio1, minus-loss gradient equals sum_i a_i g_i/K. Scores differentiate only parameters the checkpoint marks requires_grad. Full global normalization group is required; single-rank slices cannot infer contrast geometry of the complete group. Model.eval replay disables dropout and captures deterministic checkpoint geometry, not training dropout distribution. Frozen answers and KL terms excluded. Tests compare weighted score vector with exact thought-loss gradient.

B2 MUST: one independent group per prompt; stable salted hashes select at most128 prompts before reading score values. Require externally attested independent split and full normalization group. Measure exact vectors without normalization or sketches; Q=SS^T, H=PQP. Norm CV, centered norm CV, tr(H)/tr(Q), ||sum g||²/(K trQ), effective rank tr(H)²/tr(H²). Zero-risk denominators NA; no score-based group exclusions. Report all selected IDs and prompt-level2000-bootstrap intervals(seed1732052027). No inferential claim with fewer than32 independent prompts; report descriptive insufficiency instead.

B3 MUST: report missing target artifacts and whether replay can actually run. Available CPU tooling tests do not satisfy B2. Qwen0.5B cache and unit models cannot be substituted for target Qwen2.5-VL-3B/ShareRobot.

## Milestones and budget

M0 environment/trainer audit; M1 exporter/analyzer tests; M2 target replay once checkpoint and processed replay are available; M3 interpretation before any separate method proposal. GPU-hours0. No downloads, trainer modification or model training. CPU cost for target extraction cannot be estimated before replay token/image sizes and trainable parameter count are known.

## Risks and stop rules

Input independence is a provenance requirement, not verified by a JSON string. Norm dispersion alone does not prove update-error headroom. Unknown target artifacts block real measurements. CPU eval-mode exact LoRA gradients may be costly; preserve explicit incomplete status. No scalar synthetic p dataset is reused as a real replay.

## Tracker

| Run | Purpose | Status |
|---|---|---|
| S001 | Environment and trainer mapping | in progress |
| S002 | Exact export and geometry tooling tests | in progress |
| S003 | Independent Qwen2.5-VL-3B / ShareRobot geometry | pending checkpoint/replay |
