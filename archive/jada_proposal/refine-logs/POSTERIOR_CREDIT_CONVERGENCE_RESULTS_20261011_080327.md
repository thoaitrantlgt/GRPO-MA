# Posterior-credit convergence completed

**STOP_POSTERIOR_CREDIT_AFTER_CONVERGENCE**. Largest-level numerical criteria pass; the stress regression remains. GPU cost0.

| Points per scramble | Half max credit difference | Reduction vs plug-in | Reduction vs frozen scale | Stress failures |
|---:|---:|---:|---:|---:|
| 512 | 0.023876 | 16.429% | 7.934% | 1/24 |
| 2048 | 0.006745 | 16.430% | 7.934% | 1/24 |
| 8192 | 0.001525 | 16.438% | 7.943% | 1/24 |
| 32768 | 0.000257 | 16.435% | 7.939% | 1/24 |

Highest-level adjacent credit change0.001278 (tolerance0.01); maximum normalized stress-risk change0.0434% (tolerance1%). The sparse-extreme/paired-correlated stress cell still worsens17.537%, above the unchanged10% threshold. Mean improvement cannot rescue this failure.

Eight independent scrambles, four nested levels declared before integration, same prior seed streams. Largest level262,144 base points; 24-permutation symmetry represents6,291,456 equally weighted entries without explicitly expanding support. Joint-integral symmetrization matches direct augmentation. Prior, target, uniform K4/total16 budget, all129 vectors, seven geometries and original scalar0.862610 are unchanged.

Numerical stability is diagnostic evidence, not a rigorous bound on continuous-prior integration error. The original prototype STOP remains preserved. All data are consumed development, no fresh intervals or new confirmation gate. Synthetic fixed-score update error does not establish clipping, real-score, reward or training gains.

Completed 10836 exact method/vector/Gram evaluations; prior integration31.26s CPU. All35 repository tests PASS; finite-prior/local identity audit and independent saved-table reconstruction PASS. Review is local-only.

A missing original source snapshot interrupted evaluation after integration; recovery reused frozen tables and verified every source vector against original moments. RESUME_NOTE.json records the interruption; no protocol change or refit occurred.

[Full convergence report](posterior-credit-convergence/20261011_run01/RESULTS.md) · [Plot](posterior-credit-convergence/20261011_run01/convergence.png) · [All metrics](posterior-credit-convergence/20261011_run01/metrics.csv)

Recommendation: stop the current Jeffreys posterior-credit prototype after the declared numerical ceiling. No alternative prior, estimator tuning, confirmation, trainer integration or GPU run was launched. This is not a universal impossibility result for Bayesian credit estimation.
