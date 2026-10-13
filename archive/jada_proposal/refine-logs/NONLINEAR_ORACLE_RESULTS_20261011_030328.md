# Exact nonlinear-oracle CPU gate completed

**HEADROOM for CPU method development; no online-training GO.** Scope: K4/B16, m0=1,2, budget-precommitted reuse.

Fresh confirmation retains 96 vectors from the original six-component mixture, seed 2718282026 committed before outcomes. All allocations and pilot/main Bernoulli states are enumerated exactly.

| Fresh confirmation comparison against uniform/variance-only | MSE reduction | 95% interval |
|---|---:|---:|
| JADA | -2.40% | [-2.78%, -2.02%] |
| local-surrogate-oracle | -4.78% | [-6.73%, -2.83%] |
| fixed-nonlinear-oracle | 6.70% | [5.24%, 8.16%] |
| adaptive-nonlinear-oracle | 10.22% | [8.87%, 11.57%] |

Uniform and variance-only coincide under both small-budget settings, so the two controls are not independent evidence.

Completed 396 exact cells, 2,376 method evaluations and independent global-optimum audit. All 12 tests pass. GPU cost 0. Fresh review remains same-family provisional.

[Full report](nonlinear-oracle-gate/20261011_run01/RESULTS.md) · [Figure](nonlinear-oracle-gate/20261011_run01/headroom.png) · [Derivations](../experiments/nonlinear_oracle/DERIVATIONS.md)

The original JADA remains NO-GO. True-p oracles are information controls; these gains are not a deployable method. The next justified work is a separate CPU allocator targeting posterior expected nonlinear MSE, using only pilots at allocation time, with a new predeclared confirmation set. Larger budgets and GPU/frozen-model/online work were not run.
