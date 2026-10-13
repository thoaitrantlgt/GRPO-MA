# CPU estimator gate tracker

| Block | Status | Evidence |
|---|---|---|
| Distribution and seeds predeclared | DONE | experiments/cpu_gate/protocol.json and run probability_vectors.json |
| Covariance and local surrogate derivations | DONE | experiments/cpu_gate/DERIVATIONS.md |
| Jacobian and allocator sanity | PASS | Seven tests; verification.json |
| Fresh implementation review | CLEARED, provisional | EXPERIMENT_CODE_REVIEW.md; both blockers fixed before sampling |
| Exact anchors and K2 | DONE | 924 exact cells |
| Full paired CPU sweep | DONE | 1,728 MC cells, 5,184,000 trials; 42,432 total method evaluations |
| Diagnostic curves and tie sensitivity | DONE | improvement_curves.png/pdf/svg and diagnostic/tie CSVs |
| CPU scientific gate | NO-GO | JADA MSE +3.24% [2.89%,3.59%] versus variance-only, precommitted reuse |
| Frozen-model / gradient / online expansion | NOT RUN | CPU evidence does not pass gate; no GPU cost |

## Nonlinear-oracle gate

| Block | Status | Evidence |
|---|---|---|
| Fresh protocol and confirmation | DONE | 96 new vectors, saved before outcomes |
| Exact global nonlinear optima | DONE | 165/35 allocations, 396 exact cells |
| Tests and saved-objective audit | PASS | 12 tests; AUDIT.json |
| Fresh same-family review | CLEARED, provisional | nonlinear run CODE_REVIEW.md |
| Oracle scientific gate | HEADROOM, scoped CPU only | Adaptive reduction 10.22% [8.87%,11.57%] |
| Pilot-only posterior nonlinear allocator | NOT RUN | Next method-development step; new confirmation required |
| GPU/frozen-model/online training | NOT RUN | No deployable-method GO; GPU cost 0 |
