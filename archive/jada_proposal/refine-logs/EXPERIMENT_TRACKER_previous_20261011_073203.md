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
| Pilot-only posterior nonlinear allocator | COMPLETED below | Separate fresh confirmation; primary NO_CLEAR_SIGNAL |
| GPU/frozen-model/online training | NOT RUN | No deployable-method GO; GPU cost 0 |

## Pilot-only posterior allocator gate

| Block | Status | Evidence |
|---|---|---|
| Priors, integration rule and fresh confirmation | DONE | Frozen protocol; 96 new vectors, unchanged mixture |
| Pilot-only nonlinear policies | DONE | Jeffreys primary and benchmark-mixture secondary; 4 tables |
| Fresh implementation review | CLEARED, provisional | posterior run CODE_REVIEW.md |
| Tests and saved-policy/result audit | PASS | 17 tests; 588 exact cells, 4,704 evaluations |
| Scientific gate | NO_CLEAR_SIGNAL | Primary MSE reduction 0.0125% [0.0085%,0.0165%]; threshold lower CI >5% |
| Positive-result ablations | SKIPPED | Primary method did not pass gate |
| GPU/frozen-model/online training | NOT RUN | No practical-method GO; GPU cost 0 |

## Consumed-confirmation information-gap diagnosis

Post hoc analysis completed; no refit or fresh sampling. m0=1 remains exactly uniform. At m0=2, the frozen policy recovers only 0.322% of the true-p oracle gain and changes allocation with 9.08% probability. Diagnostic curves cover pilot fraction, dispersion and leverage. The exact risk decomposition and unchanged-source audit pass. These findings do not prove pilot-only impossibility. [Diagnostic report](posterior-information-diagnostic/20261011_run01/RESULTS.md).

[Next CPU design](PILOT_INFORMATION_GATE_PLAN.md) is proposed, not run: isolate pilot information at fixed main budget and reuse, with separate matched-total-cost controls. No GPU cost.

## Pilot-information gate

| Block | Status | Evidence |
|---|---|---|
| Runtime-only feasibility and frozen design | DONE | Exact engine all six settings; FEASIBILITY.json/protocol.json |
| Fresh confirmation | DONE | 96 unchanged-mixture vectors, seed1618032026; six stress |
| Fixed-main and equal-total-cost evaluation | DONE | 612 exact cells, 3,672 method evaluations |
| Tests / end-to-end sanity / saved audit | PASS | 19 tests; SANITY.json and AUDIT.json |
| Implementation review | LOCAL-ONLY | Delegated reviewer unavailable: usage limit |
| Scientific gate | NO_CLEAR_SIGNAL | PILOT_INFORMATION_RESULTS.md |
| GPU/frozen-model/online training | NOT RUN | GPU cost0; no training GO |

## Final posterior convergence diagnosis

| Block | Status | Evidence |
|---|---|---|
| Nested integration levels / seeds | DONE | Jeffreys up to32,768; mixture up to49,152 prior points |
| Two m0=4 settings / both priors | DONE | Exact finite risks; eight independent scrambles |
| Consumed-data evaluation | DONE, development only | No new confirmation or new reward sampling |
| Tests and saved-policy audit | PASS | 20 tests; posterior-convergence AUDIT.json |
| Numerical disposition | RECOMMEND_STOP_CURRENT_ALLOCATOR_FAMILY | POSTERIOR_CONVERGENCE_RESULTS.md |
| Gradient objective | MEMO ONLY | No experiment, method implementation or training GO |
| GPU/model/online work | NOT RUN | GPU cost0 |

## Synthetic fixed-Gram gradient objective CPU gate

| Block | Status | Evidence |
|---|---|---|
| Fixed geometry/protocol/fresh confirmation | DONE | Seven Grams; new96 vectors seed1732052026 |
| Exact weighted objective/oracles/baselines | DONE | Three settings; 4158 cells, 37422 evaluations |
| Tests and end-to-end sanity | PASS | 23 tests; SANITY.json |
| Saved-table and global-minimum audit | PASS | AUDIT.json; scalar/null/control invariants |
| Oracle objective-specific gate | NO_OBJECTIVE_SPECIFIC_SIGNAL | GRADIENT_OBJECTIVE_RESULTS.md |
| Pilot posterior gradient policy | NOT RUN | New protocol/confirmation required; oracle-only gate |
| GPU/model/online training | NOT RUN | Synthetic fixed score geometry only; cost0 |
