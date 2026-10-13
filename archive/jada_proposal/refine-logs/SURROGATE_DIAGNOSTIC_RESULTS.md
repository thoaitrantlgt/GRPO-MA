# Exact surrogate mechanism diagnosis

Posthoc mechanism diagnosis, not a new confirmation or allocator gate. No fresh intervals or training claim; CPU only.

K4 / total16 / m0=2 / budget-precommitted reuse0.5. Six stress regimes, 96 consumed vectors from the unchanged predeclared mixture, and all 27 deterministic dispersion-path points. Paths are probes, not a selected benchmark distribution.

| Method | Exact MSE | Squared bias | Variance | Linear surrogate | Bias share |
|---|---:|---:|---:|---:|---:|
| uniform | 0.742004 | 0.235013 | 0.506991 | 5.275466 | 31.67% |
| JADA | 0.753401 | 0.237696 | 0.515704 | 5.589635 | 31.55% |
| local-surrogate-oracle | 0.771011 | 0.244651 | 0.526360 | 5.172296 | 31.73% |
| fixed-nonlinear-oracle | 0.704758 | 0.206951 | 0.497807 | 5.867546 | 29.36% |
| adaptive-nonlinear-oracle | 0.680017 | 0.192899 | 0.487118 | 6.066346 | 28.37% |

## Decomposition and interpretation

Let delta=f(V)-f(p), L=J(p)(V-p), R=delta-L. Exact MSE = squared bias + variance = E||L||²/K + 2E[LᵀR]/K + E||R||²/K. The raw estimate and L are unbiased under fixed reuse even with pilot-dependent allocation. Normalized advantage can be biased.

| Method | Linear | Cross term | Remainder energy | Exact |
|---|---:|---:|---:|---:|
| uniform | 5.275466 | -8.604771 | 4.071309 | 0.742004 |
| JADA | 5.589635 | -9.213501 | 4.377267 | 0.753401 |
| local-surrogate-oracle | 5.172296 | -8.447167 | 4.045882 | 0.771011 |
| fixed-nonlinear-oracle | 5.867546 | -9.718363 | 4.555575 | 0.704758 |
| adaptive-nonlinear-oracle | 6.066346 | -10.130608 | 4.744279 | 0.680017 |

Both diagnostic tolerances (risk calibration within10%, fixed allocation regret at most1%) hold for 10/96 consumed vectors. Median fixed local-allocation regret is 7.261%. These are descriptive tolerances, not sufficient mathematical conditions or a population guarantee.

At equal probabilities, the target advantage and symmetric uniform-estimator bias are zero. Exact risk stays below1, whereas J scales as epsilon^(-1/2) and its surrogate can be enormous. Thus failed linearization can occur with zero bias. For other vectors, bias and curvature both contribute; attributing the full discrepancy to bias alone is incorrect.

The local expansion requires errors small relative to sqrt(dispersion+epsilon), with control of tails. The exported centered-noise/dispersion ratio measures typical scale, but does not certify tails or establish a universal threshold. Leverage and all deterministic paths are retained without selecting favorable directions.

## Dispersion bins: consumed vectors only

| Dispersion interval | Vectors | Mean local fixed regret | Joint tolerance passes |
|---|---:|---:|---:|
| [0, 0.001) | 9 | 7.601% | 0/9 |
| [0.001, 0.01) | 30 | 15.828% | 0/30 |
| [0.01, 0.05) | 30 | 25.078% | 0/30 |
| [0.05, 0.25) | 27 | 9.906% | 10/27 |

## Stress regimes

| Regime | Uniform exact | Uniform surrogate | Uniform bias share | Local fixed regret |
|---|---:|---:|---:|---:|
| symmetric-wide | 0.203057 | 0.239364 | 5.10% | 0.000% |
| symmetric-mid | 0.280224 | 0.326525 | 7.03% | 9.097% |
| all-equal | 0.972347 | 46875.000000 | 0.00% | 0.000% |
| near-equal | 1.649357 | 21.444713 | 42.56% | 0.000% |
| asymmetric | 0.201161 | 0.203369 | 5.06% | 6.019% |
| sparse-extreme | 0.068029 | 0.073612 | 1.82% | 0.000% |

Audit PASS: Taylor residual 1.46e-11; linear/covariance formula residual 1.46e-11; raw bias 1.19e-16. All35 positive main allocations enumerated; allocation optimum checked against complete candidate list.

[Figure](surrogate-diagnostic/20261011_run02/surrogate_diagnosis.png) · [Raw moments](surrogate-diagnostic/20261011_run02/moments.csv) · [Allocation diagnostics](surrogate-diagnostic/20261011_run02/allocation_diagnosis.csv) · [Frozen protocol](surrogate-diagnostic/20261011_run02/protocol.json)
