# Score geometry assumptions

ASK: never. Assurance: draft. Missing artifact paths requested asynchronously; no dependent real-data work without artifacts.

| ID | Decision | Choice | Class | Source |
|---|---|---|---|---|
| S01 | Target | Repository default Qwen2.5-VL-3B-Instruct / ShareRobot trajectory, with frozen training checkpoint and its intended LoRA parameter scope | semantic | repository |
| S02 | Score | Gradient of mean shifted per-token log probability over thought tokens, first continuation of each thought, matching compute_loss at ratio1 | semantic | trainer |
| S03 | Scope | Exact raw Euclidean gradients over all requires_grad parameters supplied by frozen checkpoint; no random projection, Adam preconditioning or unit-norm conversion | semantic | default |
| S04 | Replay | All thoughts in the full normalization group; current trainer globally gathers thought values. Exporter is single-process and requires replay already assembled as the full group. Caller owns multimodal input preparation | interface | default |
| S05 | Boundary | Only ratio1 thought-credit-dependent PPO component. Answer, KL, PPO clipping at changed parameters and optimizer update excluded | semantic | default |
| S06 | Determinism | Planned model.eval() on server CPU/GPU, no parameter mutation or optimizer; disabled dropout changes training-mode stochastic geometry, explicitly stated | semantic | default |
| S07 | Schema | Planned exact Gram Q[G,K,K] with unique group/prompt IDs, checkpoint/task/parameter_scope/split/full-normalization metadata; full score-vector storage optional if exact block accumulation and smoke contraction checks exist. Exporter not implemented | interface | user server plan / default |
| S08 | Selection | One group per prompt selected by smallest salted SHA256(group_id); select first128 prompts by salted SHA256(prompt_id), salt score-geometry-v1. No filtering by reward, norm or winners | semantic | default |
| S09 | Data status | Input metadata declares independent split; cannot prove independence from IDs alone. Freeze source selection before measurement, record selected IDs, no generated target results without real artifacts | semantic | default |
| S10 | Metrics | Raw norm CV, centered norm CV, contrast trace fraction, raw cancellation ratio, centered effective rank; zero denominators NA. Equal selected-group mean, bootstrap2000 prompt-level draws seed1732052027 | semantic | default |
| S11 | Decision | Descriptive audit only, no automated geometry threshold or method GO; prior5% screen remains failed. Findings cannot select synthetic geometry benchmark retroactively | semantic | default |
| S12 | Resources | Server plan: optional one available GPU for four-prompt smoke; project128-prompt cost from measured usage. No server job, checkpoint download or GPU rental launched. Initial local CPU-only restriction applies to completed synthetic diagnostics | interface | user server plan / default |
| S13 | Current readiness | Only read-only preflight and prompt-freeze CLIs exist. Exact exporter/analyzer are design tasks, not runnable measurement commands | interface | user server plan |
| S14 | Frozen cohort | Main128 independent prompts; first4 smoke. Missing images/insufficient cohort fail preparation without score-based replacement | semantic | default |
