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
