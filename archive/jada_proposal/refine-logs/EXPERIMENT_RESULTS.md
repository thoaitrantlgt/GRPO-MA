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


## Synthetic gradient objective headroom

**NO_OBJECTIVE_SPECIFIC_SIGNAL**. Information headroom only; GPU cost0.

| Four-Gram primary comparison | Gradient-MSE reduction | CI95 |
|---|---:|---:|
| adaptive-gradient-oracle-vs-uniform | 11.358% | [10.363%, 12.353%] |
| adaptive-gradient-oracle-vs-adaptive-scalar-oracle | 3.091% | [2.392%, 3.789%] |
| fixed-gradient-oracle-vs-uniform | 6.055% | [4.862%, 7.249%] |
| gradient-surrogate-oracle-vs-uniform | -5.102% | [-6.644%, -3.559%] |
| gradient-JADA-vs-uniform | -1.623% | [-1.977%, -1.268%] |
| JADA-vs-uniform | -1.891% | [-2.251%, -1.531%] |

Seven predeclared trace-one PSD Grams, three K4 settings, 96 new probability vectors from the unchanged six-component mixture (seed1732052026), 96 consumed development vectors and six stress regimes. Primary geometry is the equal-weight four nontrivial fixed matrices; scalar-equivalent and null matrices are controls. No Gram or distribution was selected after winners.

Oracle headroom requires both lower95 limits>5% versus uniform and versus adaptive scalar oracle. All risks are exact over reward states and complete integer allocations; intervals cluster repeated geometries/settings inside each fresh vector. Fixed G is supplied by the synthetic model, not estimated from real LLM gradients. Gradient-JADA is a known-G pilot plug-in diagnostic; primary gate is oracle-only.

Completed 4158 cells / 37422 method evaluations; 23 tests, sanity and saved-table audit PASS. Review status: same-family provisional: GPT-6-Astra xhigh fresh review; no blocking correctness issues. GPU cost0.

Old JADA scalar NO-GO and posterior NO_CLEAR_SIGNAL remain unchanged. No deployable-method, measured-gradient or training GO follows. Fresh confirmation is consumed; a later weighted posterior method requires a new protocol/confirmation. Oracle-only headroom does not trigger positive-method ablations.

The additional oracle effect is positive but below the declared5% materiality threshold; failure is not proof of zero geometry effect. Known-G pilot gradient-JADA worsens weighted error1.623%, and true-p local surrogate worsens it5.102%. These results do not justify GPU work or another automatic allocator variant.

[Full results](gradient-objective-gate/20261011_run01/RESULTS.md) · [Figure](gradient-objective-gate/20261011_run01/gradient_headroom.png) · [Derivations](../experiments/gradient_gate/DERIVATIONS.md)


## 2026-10-11: Exact surrogate mechanism diagnosis and consolidated CPU report

Completed645 exact method/vector evaluations on96 consumed vectors, six stress regimes, and27 fixed dispersion probes, K4/B16/m0=2. Uniform MSE0.742004, squared bias0.235013 (31.67%), variance0.506991; linear surrogate5.275466. True-p local allocation raises exact MSE to0.771011. Equal-probability uniform bias is zero while surrogate46875 versus exact0.972347, so bias alone does not explain failure. Median fixed local regret7.261%; joint10%-calibration/1%-regret tolerance passes10/96. All26 tests and identity/allocator audit PASS. Descriptive consumed-data analysis only; no new intervals or gate change, no independent reviewer claim. Failed CSV-export run01 preserved; completed run02. CPU-only, GPU cost0.

See [consolidated report](ESTIMATOR_RESEARCH_REPORT.md) and [diagnosis](surrogate-diagnostic/20261011_run02/RESULTS.md). Stop current allocator family; no automatic method variant or training escalation.

# Normalization amplitude probe completed

**REJECT_FIXED_HALF_AS_GENERAL_FOLLOWUP**. Exact consumed-data CPU diagnosis; no fresh confirmation. GPU cost0.

| Total budget | Fixed-half mean MSE reduction |
|---|---:|
| 8 | 17.51% |
| 16 | 13.49% |
| 32 | 4.42% |

The average criteria pass, but 11/18 stress/budget cells worsen more than10%, so the complete predeclared development rule rejects fixed-half. No scalar sweep, method replacement, benchmark distribution change, or fresh confirmation was used.

Uniform samples, unchanged normalized target, 96 consumed mixture vectors, six original stress regimes and27 fixed dispersion paths. Exact finite reward enumeration covers all count states at budgets8/16/32, for1,548 method/vector evaluations. Bias/variance and quadratic-risk audit PASS; all29 repository tests PASS (python -m unittest discover -s tests). Local verification only; no independent reviewer claim.

Mathematically, R(c)=a*c²-2*b*c+d, so half amplitude improves precisely when b<0.75*a. Variance quarters, but high-signal bias can grow. Fixed c<1 is inconsistent for a nonconstant normalized target as samples grow; constant scaling can overlap learning-rate scaling and is not itself a distinct training method.

Status: normalization tradeoff is coherent as stated; universal fixed shrinkage benefit is disproved by the noiseless boundary. Benchmark-average improvement is descriptive and cannot override the frozen stress-regression criterion. Previous allocator and gradient gates remain unchanged.

[Full results](normalization-probe/20261011_run01/RESULTS.md) · [Plot](normalization-probe/20261011_run01/normalization_tradeoff.png) · [Derivation package](../DERIVATION_PACKAGE.md)

Next research decision: specify whether the intended credit/update target should remain the current normalized advantage or change for an independently justified reason. Do not launch fixed-half confirmation, tune coefficients, or rent GPU on these results alone.

# Concrete posterior-credit prototype completed

**STOP_POSTERIOR_CREDIT_PROTOTYPE**. Development data consumed; no new confirmation. GPU cost0.

The estimator is E[h_tau(p)|counts] under fixed iid Jeffreys prior; allocation stays uniform, K4/total16. Target uses sample standard deviation with floor1e-4. Runtime uses observed counts and a frozen625-entry table only. It does not know true p or score geometry.

| Four-Gram consumed-data comparison | Update-MSE reduction |
|---|---:|
| posterior-credit vs plugin | 16.421% |
| posterior-credit vs prior-scale | 7.925% |

The mean criteria pass, but 1/24 primary stress cells worsen >10%; sparse-extreme/paired-correlated worsens17.815%. Half-support maximum credit difference is0.042499, exceeding the frozen0.02 stability tolerance. Both failures block continuation under the complete declared rule. Do not override them with favorable averages.

The scalar control coefficient0.862610 was fitted from prior predictive risk only, before development risk access. All96 consumed mixture vectors, six stress regimes,27 fixed paths and seven existing Grams are retained. The primary average uses the same four nontrivial synthetic geometries. No prior, geometry, distribution, coefficient or method switch was selected after outcomes.

Prior fit uses2048 base Sobol points across four independent scrambles, each augmented with24 coordinate permutations (49,152 weighted support entries). Finite-support Bayes optimality, coordinate exchangeability, centering, null updates and bias/variance identities pass local audit. The continuous-prior integration is approximate and fails its stability check; exact outcome risk refers to the frozen numerical estimator, not the exact continuous posterior.

Completed 2709 exact method/vector/Gram evaluations; prior fit5.68s CPU, evaluation0.67s. All32 repository tests PASS. Local verification only; no fresh independent reviewer clearance.

[Full results](posterior-credit/20261011_run01/RESULTS.md) · [Prototype design](../experiments/posterior_credit/DESIGN.md) · [Raw metrics](posterior-credit/20261011_run01/metrics.csv) · [Plot](posterior-credit/20261011_run01/posterior_credit.png)

No trainer integration, GPU, model download, new confirmation, numerical rescue run or alternative prior was launched. Previous allocator and gradient gates remain unchanged. The prototype demonstrates a mathematically coherent conditional credit estimator but does not pass its development gate.

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
