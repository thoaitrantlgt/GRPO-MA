# Estimator research results

Original JADA CPU gate: **NO-GO**. On the predeclared broad mixture, budget-precommitted JADA increases MSE by 3.24% [2.89%,3.59%]. [Full CPU sweep](cpu-estimator-gate/20261011_run01/RESULTS.md).

Exact nonlinear-oracle gate: **HEADROOM for CPU method development** at K4/B16/m0=1,2. The previous fresh confirmation showed fixed true-p nonlinear allocation reducing MSE by 6.70% [5.24%,8.16%] and adaptive true-p-plus-pilot allocation by 10.22% [8.87%,11.57%]. The local-surrogate oracle worsened MSE by 4.78%. [Oracle results](nonlinear-oracle-gate/20261011_run01/RESULTS.md).

New pilot-only posterior allocator gate: **NO_CLEAR_SIGNAL**. On a separate 96-vector confirmation set, Jeffreys and the benchmark-informed mixture prior each reduce MSE by only 0.0125% [0.0085%,0.0165%], far below the declared 5% lower-confidence threshold. m0=1 exactly equals uniform. The true-p adaptive oracle retains 9.82% [8.68%,10.95%] headroom on these vectors. Runtime decisions use only pilot counts; no true p is available to the posterior allocator. [Posterior results](posterior-allocator-gate/20261011_run01/RESULTS.md).

Uniform and variance-only coincide in the two scoped nonlinear gates. Both posterior priors were predeclared and produced the same policy. Confidence intervals are conditional on the frozen approximate prior-integrated policies. All 17 tests pass; saved-table audit covers 588 exact cells and 4,704 method evaluations. Same-family implementation review remains provisional. All work used CPU; GPU cost 0.

The practical method has not recovered meaningful oracle gains. No training GO is established; positive-result ablations are skipped. Further work requires a distinct hypothesis and another predeclared confirmation set. No GPU/frozen-model/online run was started.

## Consumed-confirmation information-gap diagnosis

Post hoc analysis completed; no refit or fresh sampling. m0=1 remains exactly uniform. At m0=2, the frozen policy recovers only 0.322% of the true-p oracle gain and changes allocation with 9.08% probability. Diagnostic curves cover pilot fraction, dispersion and leverage. The exact risk decomposition and unchanged-source audit pass. These findings do not prove pilot-only impossibility. [Diagnostic report](posterior-information-diagnostic/20261011_run01/RESULTS.md).

[Next CPU design](PILOT_INFORMATION_GATE_PLAN.md) is proposed, not run: isolate pilot information at fixed main budget and reuse, with separate matched-total-cost controls. No GPU cost.
