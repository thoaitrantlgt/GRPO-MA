# Estimator research results

Original JADA CPU gate: **NO-GO**. On the predeclared broad mixture, budget-precommitted JADA increases MSE by 3.24% [2.89%,3.59%]. [Full CPU sweep](cpu-estimator-gate/20261011_run01/RESULTS.md).

Exact nonlinear-oracle gate: **HEADROOM for CPU method development** at K4/B16/m0=1,2. The previous fresh confirmation showed fixed true-p nonlinear allocation reducing MSE by 6.70% [5.24%,8.16%] and adaptive true-p-plus-pilot allocation by 10.22% [8.87%,11.57%]. The local-surrogate oracle worsened MSE by 4.78%. [Oracle results](nonlinear-oracle-gate/20261011_run01/RESULTS.md).

New pilot-only posterior allocator gate: **NO_CLEAR_SIGNAL**. On a separate 96-vector confirmation set, Jeffreys and the benchmark-informed mixture prior each reduce MSE by only 0.0125% [0.0085%,0.0165%], far below the declared 5% lower-confidence threshold. m0=1 exactly equals uniform. The true-p adaptive oracle retains 9.82% [8.68%,10.95%] headroom on these vectors. Runtime decisions use only pilot counts; no true p is available to the posterior allocator. [Posterior results](posterior-allocator-gate/20261011_run01/RESULTS.md).

Uniform and variance-only coincide in the two scoped nonlinear gates. Both posterior priors were predeclared and produced the same policy. Confidence intervals are conditional on the frozen approximate prior-integrated policies. All 17 tests pass; saved-table audit covers 588 exact cells and 4,704 method evaluations. Same-family implementation review remains provisional. All work used CPU; GPU cost 0.

The practical method has not recovered meaningful oracle gains. No training GO is established; positive-result ablations are skipped. Further work requires a distinct hypothesis and another predeclared confirmation set. No GPU/frozen-model/online run was started.

## Consumed-confirmation information-gap diagnosis

Post hoc analysis completed; no refit or fresh sampling. m0=1 remains exactly uniform. At m0=2, the frozen policy recovers only 0.322% of the true-p oracle gain and changes allocation with 9.08% probability. Diagnostic curves cover pilot fraction, dispersion and leverage. The exact risk decomposition and unchanged-source audit pass. These findings do not prove pilot-only impossibility. [Diagnostic report](posterior-information-diagnostic/20261011_run01/RESULTS.md).

[Pilot-information design](PILOT_INFORMATION_GATE_PLAN.md), subsequently completed below: isolate pilot information at fixed main budget and reuse, with separate matched-total-cost controls. No GPU cost.

## Completed pilot-information CPU gate

**NO_CLEAR_SIGNAL**. CPU only; GPU cost 0.

| Setting | Primary reduction vs uniform | CI95 |
|---|---:|---:|
| fixed-main-m0-1 | 0.351% | [0.300%, 0.401%] |
| fixed-main-m0-2 | 0.017% | [0.005%, 0.028%] |
| fixed-main-m0-4 | -0.066% | [-0.110%, -0.022%] |
| fixed-total-m0-1 | 0.000% | [-0.000%, 0.000%] |
| fixed-total-m0-2 | -0.115% | [-0.153%, -0.076%] |
| fixed-total-m0-4 | 0.083% | [0.030%, 0.136%] |

Fixed-main holds main8 and lambda=.5 with total costs12/16/24. Fixed-total holds total24 with main20/16/8 and budget-precommitted reuse. Primary information gate requires lower95 limit>5% at fixed-main/m0=4; practical equal-cost gate adds >5% at fixed-total/m0=4 and positive points at all matched-total settings.

Jeffreys primary and benchmark-informed mixture secondary were frozen before evaluation; secondary results cannot rescue primary failure. Unchanged six-component mixture, fresh96 vectors seed1618032026 plus six stress regimes. All methods, settings and regimes retained; this confirmation is now consumed.

19 tests passed before deployment; end-to-end finite-prior sanity and saved-result audit pass. Exact reward-state evaluation covers 612 cells / 3,672 method evaluations. Prior integration is approximate and independent-scramble stability is reported separately. Confidence intervals are conditional on frozen policies.

Implementation review is local-only: the requested fresh reviewer was unavailable due to a usage limit. Do not describe this as independent review clearance. No training or LLM claim is supported.

Neither predeclared gate passes. The primary fixed-main/m0=4 outcome worsens MSE by 0.066%; matched total24/m0=4 improves it only 0.083%. The benchmark-informed secondary does not rescue the gate. Increasing pilots alone did not recover material benefit under these frozen policies. This is not a general impossibility theorem, and primary matched-cost quadrature instability remains a limitation.

[Full results](pilot-information-gate/20261011_run01/RESULTS.md) · [Figure](pilot-information-gate/20261011_run01/information_gate.png) · [Derivations](../experiments/posterior_allocator/INFORMATION_DERIVATIONS.md)


## Final convergence diagnosis

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

