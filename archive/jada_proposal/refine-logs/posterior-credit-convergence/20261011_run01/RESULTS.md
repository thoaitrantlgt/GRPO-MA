# Posterior-credit numerical convergence

**STOP_POSTERIOR_CREDIT_AFTER_CONVERGENCE**. All original model choices retained; no fresh confirmation, GPU cost0.

Eight independent Sobol scrambles, same original seed+replicate streams, nested512/2048/8192/32768 base points each. Symmetrizing joint integrals is algebraically equivalent to24-permutation support augmentation; tested against direct augmentation. Joint numerators/masses are pooled before division. Original prior-scale coefficient remains fixed.

| Points/scramble | Half max credit difference | Adjacent max credit change | Max stress risk change / plug-in | Reduction vs plug-in | Reduction vs scale control | Stress failures |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.023876 | — | — | 16.429% | 7.934% | 1/24 |
| 2048 | 0.006745 | 0.019396 | 1.1955% | 16.430% | 7.934% | 1/24 |
| 8192 | 0.001525 | 0.007468 | 0.2181% | 16.438% | 7.943% | 1/24 |
| 32768 | 0.000257 | 0.001278 | 0.0434% | 16.435% | 7.939% | 1/24 |

## Highest-level stress results

| Regime | Gram | Posterior reduction vs plug-in |
|---|---|---:|
| symmetric-wide | contrast-rank1 | 4.808% |
| symmetric-wide | heterogeneous-diagonal | 0.088% |
| symmetric-wide | paired-correlated | -7.692% |
| symmetric-wide | dense-correlated | -1.375% |
| symmetric-mid | contrast-rank1 | -1.012% |
| symmetric-mid | heterogeneous-diagonal | 1.102% |
| symmetric-mid | paired-correlated | -4.446% |
| symmetric-mid | dense-correlated | -5.556% |
| all-equal | contrast-rank1 | 51.899% |
| all-equal | heterogeneous-diagonal | 51.899% |
| all-equal | paired-correlated | 51.899% |
| all-equal | dense-correlated | 51.899% |
| near-equal | contrast-rank1 | 35.597% |
| near-equal | heterogeneous-diagonal | 22.917% |
| near-equal | paired-correlated | 14.046% |
| near-equal | dense-correlated | 25.047% |
| asymmetric | contrast-rank1 | 6.807% |
| asymmetric | heterogeneous-diagonal | 12.774% |
| asymmetric | paired-correlated | -5.147% |
| asymmetric | dense-correlated | 3.808% |
| sparse-extreme | contrast-rank1 | 10.467% |
| sparse-extreme | heterogeneous-diagonal | -1.298% |
| sparse-extreme | paired-correlated | -17.537% |
| sparse-extreme | dense-correlated | 1.297% |

## Findings

1. Observation: largest-level numerical stability passes all three frozen tolerances. Interpretation: this is finite numerical evidence of reproducibility, not an error bound for the continuous integral. Implication: STOP_POSTERIOR_CREDIT_AFTER_CONVERGENCE. Next step: retain the ceiling and decision; no adaptive quadrature extension or favorable-level selection.

2. Observation: highest-level mean criteria pass, with 1/24 stress failures. Interpretation: an integrated Bayes advantage need not hold pointwise across probability regimes. Implication: means cannot rescue a stress failure. Next step: apply the unchanged development contract before considering any fresh confirmation; none is started here.

3. Observation: first four lowest-level streams reproduce the original prototype; exact finite-prior Bayes and moment identities pass. Interpretation: efficient integral symmetrization preserves the mathematical estimator. Implication: differences across levels are quadrature changes, not a prior/target/budget/geometry or scalar-control change. Next step: keep raw tables and all levels for reproducibility.

Local audit PASS; original table reproduction residual1.11e-15. Prior fit31.26s CPU; evaluation1.81s. 10836 exact method/vector/Gram evaluations. All129 consumed/stress/path vectors and seven geometries retained. No population intervals on consumed data; local verification only.

The earlier prototype STOP remains in its original report. This diagnosis provides a separate numerical recommendation and is not a revised confirmation gate. Synthetic fixed score geometry only; no clipping/full-update, measured score, training, or novelty claim.

[Frozen convergence protocol](protocol.json) · [All metrics](metrics.csv) · [All stress/level outcomes](stress.csv) · [Numerical audit](AUDIT.json)
