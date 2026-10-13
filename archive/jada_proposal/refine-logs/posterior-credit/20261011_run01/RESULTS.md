# Posterior-credit prototype: consumed-data CPU diagnosis

**STOP_POSTERIOR_CREDIT_PROTOTYPE**. No fresh confirmation or training claim. GPU cost0.

K4/total16, uniform four reward samples per thought. The target is sample-std normalized true Bernoulli values with floor1e-4. Prior Jeffreys iid Beta(.5,.5); candidate is the posterior expectation of that target, not normalization of the posterior mean. All96 consumed mixture vectors, six stress regimes,27 fixed dispersion probes and seven frozen synthetic Grams retained.

Prior-scale control coefficient0.862610 was fitted from the prior alone and frozen before development risk access. Continuous integration uses four512-point Sobol supports, each augmented with all24 permutations (49152 support entries,2048 base points). Half-support maximum table difference0.042499; tolerance0.02.

| Four-Gram / consumed96 | Exact update MSE | Squared bias | Variance | Posterior reduction against row |
|---|---:|---:|---:|---:|
| plugin | 0.565857 | 0.190815 | 0.375042 | 16.42% |
| prior-scale | 0.513641 | 0.234573 | 0.279068 | 7.92% |
| posterior-credit | 0.472936 | 0.273778 | 0.199158 | 0.00% |

## All stress results, four primary fixed Grams

| Regime | Gram | Posterior reduction vs plug-in |
|---|---|---:|
| symmetric-wide | contrast-rank1 | 3.06% |
| symmetric-wide | heterogeneous-diagonal | -0.11% |
| symmetric-wide | paired-correlated | -8.87% |
| symmetric-wide | dense-correlated | -2.89% |
| symmetric-mid | contrast-rank1 | -2.28% |
| symmetric-mid | heterogeneous-diagonal | 0.91% |
| symmetric-mid | paired-correlated | -5.39% |
| symmetric-mid | dense-correlated | -6.65% |
| all-equal | contrast-rank1 | 51.94% |
| all-equal | heterogeneous-diagonal | 51.94% |
| all-equal | paired-correlated | 51.94% |
| all-equal | dense-correlated | 51.94% |
| near-equal | contrast-rank1 | 35.63% |
| near-equal | heterogeneous-diagonal | 22.90% |
| near-equal | paired-correlated | 14.05% |
| near-equal | dense-correlated | 25.05% |
| asymmetric | contrast-rank1 | 4.78% |
| asymmetric | heterogeneous-diagonal | 11.21% |
| asymmetric | paired-correlated | -7.21% |
| asymmetric | dense-correlated | 1.81% |
| sparse-extreme | contrast-rank1 | 9.08% |
| sparse-extreme | heterogeneous-diagonal | -1.89% |
| sparse-extreme | paired-correlated | -17.81% |
| sparse-extreme | dense-correlated | 0.12% |

## Findings

1. Observation: mean criteria pass; 1/24 primary stress cells regress more than10%; quadrature criterion fails. Interpretation: posterior integrated optimality does not imply pointwise robustness under a different population. Implication: the complete frozen rule yields **STOP_POSTERIOR_CREDIT_PROTOTYPE**. Next step: do not rescue a failed prototype with a different prior, budget or favorable geometry.

2. Observation: the frozen finite-prior posterior mean dominates both controls in integrated fixed-Gram risk. Interpretation: this verifies the conditional-expectation identity under its finite model; it does not certify the continuous prior or benchmark superiority. Implication: separate numerical validity from empirical gate success. Next step: keep this distinction in future research claims.

3. Observation: credits use count histories only; uniform sampling and target remain fixed. Interpretation: this is a credit-estimation test, distinct from the prior allocator tests. Implication: no inference about adaptive allocation, measured score geometry, clipping or online learning follows. Next step: revisit the information model only if an independently justified design is proposed; no automatic follow-up run.

Primary metrics average all four predeclared nontrivial Grams within each vector and equal-weight all six mixture components. No fresh intervals on consumed data. Direction/alignment/energy metrics, including direction-defined probability mass, are exported; null update energy is retained as an absolute metric. A cosine conditional on nonzero norms is not a replacement for total risk.

Local audit PASS: bias/variance residual5.55e-17, centering5.55e-16, symmetry6.66e-16; finite-prior Bayes maximum excess0. Runtime table625 rows; exact outcome-state enumeration. Continuous prior approximation remains separate from exact evaluation of the frozen table.

[Design and derivation](../../../experiments/posterior_credit/DESIGN.md) · [Frozen protocol](protocol.json) · [Raw metrics](metrics.csv) · [Frozen estimator](frozen_table.npz)
