# Final numerical convergence diagnosis

**RECOMMEND_STOP_CURRENT_ALLOCATOR_FAMILY**. Original CPU gates remain unchanged.

| Prior | m0=4 setting | Highest prior points | Half-policy TV | Consumed-development MSE reduction |
|---|---|---:|---:|---:|
| posterior-Jeffreys | fixed-main-m0-4 | 32768 | 0.000000 | -0.0264% |
| posterior-Jeffreys | fixed-total-m0-4 | 32768 | 0.000000 | 0.1088% |
| posterior-mixture | fixed-main-m0-4 | 49152 | 0.000000 | 0.0020% |
| posterior-mixture | fixed-total-m0-4 | 49152 | 0.000000 | 0.1447% |

Eight independently scrambled Sobol replicates; three nested component-balanced levels; settings and counts frozen before this numerical run. Exact reward-state risk, approximate continuous prior integration. Previous confirmation is consumed development; no fresh population inference or new GO. No method/prior winner selection.

All 20 tests pass, including chunked-prefix joint-integral checks against direct finite-prior integration. Reconstruction/balance/oracle audit passes. GPU cost0. Local diagnostic review only; no fresh reviewer clearance. Not an exact continuous Bayes-optimality certificate or impossibility theorem.

[Full convergence report](posterior-convergence/20261011_run01/RESULTS.md) · [Figure](posterior-convergence/20261011_run01/convergence.png)

[Distinct objective memo](GRADIENT_OBJECTIVE_MEMO.md): fixed-score update error is E[delta^T G delta], with Gram G positive semidefinite. This suggests a CPU falsification of gradient geometry before any model experiment; no gradient-aware method or training run has been evaluated.
