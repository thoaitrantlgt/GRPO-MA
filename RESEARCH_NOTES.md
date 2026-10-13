# Research notes

## 2026-10-11: JADA-GRPO proposal review

Full report: [JADA_GRPO_REVIEW.md](JADA_GRPO_REVIEW.md). Source proposal v0.2, 2026-09-30. Reviewer `/root/jada_reviewer`, model `gpt-6-astra`, ultra; same-family, provisional. Traces: `.aris/traces/research-review/2026-10-11_run01/`.

- Fixed-weight raw pilot/main thought values are unbiased under fixed policy, independent pilot/main streams, pilot-measurable counts and at least one main sample per thought. The exact reused-value covariance is diagonal with entries `sigma_j² [lambda²/m0 + (1-lambda)² E(1/M_j)]`. A common fixed lambda preserves the allocation-dependent c/M surrogate; do not claim a fatal estimator mismatch or unavoidable adaptive covariance.
- Default K4/m0=2/main8/lambda=.5 exact Bernoulli diagnostics give only 0.73% advantage-MSE improvement on `[.95,.55,.45,.05]` and worse MSE on several ordinary regimes. Variance-only greedy counts are always uniform at this default. This does not pass the proposal's broad Gate 1; do CPU revisions before expensive training.
- Fixed nonzero lambda with fixed pilot size creates a variance floor across growing budgets. Include precommitted budget-dependent weights; do not choose them from current pilot outcomes.
- Equal per-thought outer loss weights alone do not preserve answer normalization or KL semantics. Define weighted answer moments, pilot answer-loss usage, token reduction and global group size.
- Repo K is per GPU rank; theory/allocation must use the same normalization group as training. Repo sample-std/clamp differs from proposal population-variance/sqrt-epsilon.
- GRPO-MA v4 already supplies normalization sensitivity; EPIG-Tree supplies sensitivity-weighted square-root allocation. Novelty is a specific finite-budget normalized-credit procedure and measured benefit, not these foundational derivations.
- Advantage-MSE reduction is not a guarantee of gradient-MSE or online gains. Use token-reduction-matched conditional LoRA gradient comparisons and measured generation/time costs.
- Next gates: CPU estimator sweep -> independent frozen pools -> token-matched gradient calibration -> twelve-run controlled online pilot (native uniform, two-stage uniform, variance-only, JADA x 3 seeds), only if earlier gates pass. No JADA model training was performed for this review.

## 2026-10-11: Completed CPU estimator gate

Run: archive/jada_proposal/refine-logs/cpu-estimator-gate/20261011_run01/RESULTS.md. Protocol and 408 probability vectors saved before new reward sampling; equal-weight six-component heterogeneous mixture and six named regimes retained. All 26 feasible settings and 16 method combinations evaluated: 924 exact cells, 1,728 paired-MC cells and 5,184,000 paired trials.

- Primary JADA versus variance-only with budget-precommitted reuse increases normalized-advantage MSE by 3.24%, 95% interval [2.89%,3.59%]. K4 +13.23%, K8 +1.37%, K16 +0.18%; K2 is an exact equality control. CPU gate NO-GO.
- Exact covariance, naive moments, local-risk limitations, analytic Jacobian and integer allocator invariants are documented and verified. Seven tests passed. Fixed .5 reuse gives a tiny matched-baseline gain but worse absolute error than precommitted reuse/uniform pooling.
- Same-family provisional fresh review found and cleared K2 numerical cancellation and incorrect stratified interval df before sampling. Frozen-model/online work was not started; GPU cost zero.

## 2026-10-11: Exact nonlinear-oracle headroom gate

Run: archive/jada_proposal/refine-logs/nonlinear-oracle-gate/20261011_run01/RESULTS.md. Scoped exact K4/B16, m0=1,2 with budget-precommitted reuse. Exhaustively enumerated 165/35 positive integer allocations and all pilot/main count states. Six comparators evaluated on 96 old development vectors, six stress regimes and 96 fresh confirmation vectors (same mixture; seed 2718282026 saved before outcomes).

- Fresh confirmation: adaptive true-p nonlinear oracle reduces MSE by 10.22%, 95% interval [8.87%,11.57%], relative to uniform/variance-only, which coincide in this small-budget design. Fixed true-p nonlinear oracle reduces MSE by 6.70% [5.24%,8.16%]. True-p local-surrogate oracle increases MSE by 4.78% [2.83%,6.73%]; current JADA increases it by 2.40%.
- HEADROOM for CPU method development; original JADA remains NO-GO. Knowing true p is unavailable to a deployable method, so oracle gains are not learnability or training evidence. Budgets 32/64 and other K/reuse are outside this exact gate.
- All 12 tests passed; independent saved-table audit verifies global optima/coverage on 396 exact cells and 2,376 method evaluations. Direct-risk spot-check residual <=9.99e-16. Fresh same-family provisional review cleared the code. GPU cost zero.
- Next justified work: a pilot-only allocator targeting posterior expected nonlinear MSE, developed on existing material and evaluated on another predeclared confirmation set. The current confirmation set is consumed; no deployable-method or GPU run was performed here.

## 2026-10-11: Pilot-only posterior nonlinear allocator gate

Run: archive/jada_proposal/refine-logs/posterior-allocator-gate/20261011_run01/RESULTS.md. K4/B16/m0=1,2, budget-precommitted reuse. Primary iid Jeffreys prior and secondary known six-component mixture prior declared before fitting/evaluation. Eight Sobol scrambles, 8,192/12,288 prior points; runtime policy accepts only pilot successes and independent tie randomness. Continuous prior integration is approximate; fresh reward-state risk is exact.

- Separate 96-vector confirmation seed 3141592026, unchanged distribution. Both posterior policies reduce MSE only 0.0125%, CI95 [0.0085%,0.0165%]; m0=1 equals uniform, m0=2 gives a tiny benefit. Primary NO_CLEAR_SIGNAL under lower CI >5% plus positive both pilot sizes. The benchmark-informed secondary cannot rescue it.
- True-p adaptive oracle still reduces MSE 9.82% [8.68%,10.95%]; JADA worsens it 2.33%. This leaves a gap between information controls and recovered pilot-only benefit. Uniform/variance-only coincide. The two priors produce identical policies; half-scramble policy disagreement is zero.
- 588 exact cells, 4,704 evaluations; all 17 tests and saved-table audit pass, direct residual <=1.28e-15. Fresh same-family provisional review cleared implementation. GPU cost zero.
- Original JADA NO-GO retained; positive-result ablations skipped. Confirmation is consumed. Further experiments need a distinct hypothesis and new predeclared confirmation; no frozen-model or online training GO.

## 2026-10-11: Consumed-confirmation information diagnosis

Saved-risk decomposition verifies uniform-to-oracle gain = recovered policy gain + remaining oracle regret. At m0=2 only 0.322% of oracle gain is recovered; nonuniform allocation occurs with probability 9.08%. m0=1 is uniform for every pilot history. Dispersion/leverage/component groups are post hoc descriptive results, without significance or generalization claims. Source hashes remained unchanged. Next proposed CPU design separates pilot information from main budget/reuse and includes matched-cost controls; fresh seed reserved, not sampled. Primary NO_CLEAR_SIGNAL remains.

## 2026-10-11: Pilot-information CPU gate

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

## 2026-10-11: Final posterior convergence diagnosis

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

## 2026-10-11: Synthetic fixed-Gram gradient objective gate

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

See [consolidated report](archive/jada_proposal/refine-logs/ESTIMATOR_RESEARCH_REPORT.md) and [diagnosis](archive/jada_proposal/refine-logs/surrogate-diagnostic/20261011_run02/RESULTS.md). Stop current allocator family; no automatic method variant or training escalation.

## 2026-10-11: Fixed normalization-amplitude diagnosis

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

[Full results](archive/jada_proposal/refine-logs/normalization-probe/20261011_run01/RESULTS.md) · [Plot](archive/jada_proposal/refine-logs/normalization-probe/20261011_run01/normalization_tradeoff.png) · [Derivation package](archive/jada_proposal/DERIVATION_PACKAGE.md)

Next research decision: specify whether the intended credit/update target should remain the current normalized advantage or change for an independently justified reason. Do not launch fixed-half confirmation, tune coefficients, or rent GPU on these results alone.

## 2026-10-11: Advantage target and update contract

# Advantage target and update contract completed

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** The immediate target is the
trainer-aligned conditional thought value, transformed by an explicitly
declared sample-standard-deviation floor. The primary research loss is the
error of the thought reward-surrogate ascent gradient, with the same token
masks and token/thought reduction as the baseline.

Existing CPU gates used population variance with additive epsilon; they
remain valid for that target and cannot be renamed trainer-exact. Thought
and answer normalization are distinct, and the answer/KL branches must be
held fixed when isolating thought-credit estimation.

At ratio1, fixed-score update error is exactly E[delta^T S^T S delta].
Away from ratio1, clipping branches depend on the advantage sign, requiring
direct clipped-gradient comparison. Constant thought scaling alters its
weight relative to unchanged answer and KL gradients; it is not universally
equivalent to scaling the full optimizer learning rate.

A future estimator must beat both uniform and a development-frozen scalar
calibration control on a separately declared confirmation, with lower95
update-MSE reductions above5% and no named exact stress regression above10%.
These are proposed future criteria, not new criteria for completed gates.
Zero-target cases use absolute spurious-update energy; relative direction
metrics are undefined. Runtime information, sample cost, grouping, std-floor
and clipping behavior must be fixed before evaluation.

No concrete new normalization has been selected. Full expected-reward
gradient claims require additional treatment of baseline dependence,
length weighting and reward-score dependence. Existing NO-GO decisions
remain unchanged. No benchmark, model, trainer change, GPU or new confirmation.

[Full target contract](archive/jada_proposal/idea-stage/docs/ADVANTAGE_TARGET_CONTRACT.md) ·
[Updated derivation](archive/jada_proposal/DERIVATION_PACKAGE.md) ·
[Local algebra audit](archive/jada_proposal/refine-logs/advantage-target-contract/20261011_run02/AUDIT.json)

Local verification: 11 checks PASS; finite-difference residuals 3.59e-11 at ratio1 and 7.28e-12 under clipping. CPU PyTorch 2.12.1+cpu; pinned training environment not executed. Local-only review. GPU cost0.

## 2026-10-11: Concrete posterior-credit estimator

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

[Full results](archive/jada_proposal/refine-logs/posterior-credit/20261011_run01/RESULTS.md) · [Prototype design](archive/jada_proposal/experiments/posterior_credit/DESIGN.md) · [Raw metrics](archive/jada_proposal/refine-logs/posterior-credit/20261011_run01/metrics.csv) · [Plot](archive/jada_proposal/refine-logs/posterior-credit/20261011_run01/posterior_credit.png)

No trainer integration, GPU, model download, new confirmation, numerical rescue run or alternative prior was launched. Previous allocator and gradient gates remain unchanged. The prototype demonstrates a mathematically coherent conditional credit estimator but does not pass its development gate.

## 2026-10-11: Posterior-credit numerical convergence

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

[Full convergence report](archive/jada_proposal/refine-logs/posterior-credit-convergence/20261011_run01/RESULTS.md) · [Plot](archive/jada_proposal/refine-logs/posterior-credit-convergence/20261011_run01/convergence.png) · [All metrics](archive/jada_proposal/refine-logs/posterior-credit-convergence/20261011_run01/metrics.csv)

Recommendation: stop the current Jeffreys posterior-credit prototype after the declared numerical ceiling. No alternative prior, estimator tuning, confirmation, trainer integration or GPU run was launched. This is not a universal impossibility result for Bayesian credit estimation.

## 2026-10-11: Proposal v0.3 revision

Wrote [JADA-GRPO proposal v0.3](JADA_GRPO_research_proposal_v0.3.md), retaining the original question and exact reuse/Jacobian/allocator theory while recording the negative CPU gate and stable posterior-credit stress regression. Historical target f_epsilon and trainer-aligned target h_tau are distinct; conditional thought-update MSE is the proposed future primary endpoint. No successful replacement method, frozen-model escalation, online benefit or compute-efficiency gain is claimed. Original v0.2 PDF and all earlier gates are preserved. Future development/confirmation conditions apply only to a separately declared new design. Source effects/intervals, links and sample-std Jacobian formulas pass local checks. No new experiment, independent reviewer clearance or GPU use.


## 2026-10-11: Sequential thought-sampling proposal v0.1

Wrote [standalone proposal](SEQUENTIAL_THOUGHT_SAMPLING_PROPOSAL.md) in Vietnamese. Core gap: pilot information insufficiency; concrete one-action nonlinear-risk acquisition with posterior-reliability threshold 0.90 and balanced fallback. Fixed terminal plug-in isolates allocation; predictable adaptive posterior validity does not imply unbiased empirical means or global improvement. Includes unchanged six-component distribution, six K4 stress vectors, 26 feasible configurations, paired Monte Carlo, CPU/gradient decision gates, overhead and stop rules. Integration counts remain a feasibility-to-freeze obligation, so this is not an execution-ready preregistration. Existing proposals and STOP verdicts preserved; no experiments or GPU launched. [Local audit](refine-logs/sequential-proposal/20261011_run01/AUDIT.json).


## 2026-10-11: Previous-proposal experiment archive

Moved previous experiment implementations, tests, research contracts, derivations and all previous result/report files into [archive/jada_proposal](archive/jada_proposal/README.md). Kept the active sequential proposal and its writing artifacts separate. Updated root document links; archived file contents remain byte-identical. The relocation manifest records old/new paths and SHA-256; historical protocols and snapshot hashes remain preserved. Run archived checks from the archive working directory. No experiment rerun or GPU use.


## 2026-10-11: Sequential-sampling CPU feasibility

[Corrected run and decision](refine-logs/sequential-feasibility/20261011_run02/DECISION.md): K2/K4, B16, m0=1/2; all96 consumed heterogeneous vectors plus6 stresses;2040 method/vector evaluations. Gated sequential risk identical to uniform (0% reduction); ungated K4 improves0.0218%, below practical5% threshold. Run02 CPU70.05s, no GPU. Reward-state risks exact; continuous posterior integration approximate with two nested low-resolution supports. Numerical fallback dominates; paired MC512 episodes remains inconclusive for rare deviations. Ten tests and independent uniform-risk reconstruction PASS. Initial run preserved with a superseded notice; corrected K2 Jacobian cancellation, zero-coefficient allocator ties, randomized tie evaluation and MC discrepancy labels. Same-family assumption audit round02 identifies no new gaps, provisional only.

[Mechanism diagnostic](refine-logs/sequential-feasibility/20261011_run02/mechanism-diagnostic/RESULTS.md): if one action then balanced completion preserves terminal counts, terminal-credit law and risk are unchanged for every p. Early acquisition is therefore invisible to this lookahead. Exploratory finite-prior K2 Bellman control shows only0.1204% consumed-data reduction and projected stress failures. Stop this one-action/balanced-completion design before full sweep; future planning must allow observation-dependent terminal allocation. No full26-grid confirmation, leverage/gradient validation or training launched.

## 2026-10-11: K4 full-planning information control

[Decision and exact outcome results](refine-logs/information-control-k4/20261011_run01/DECISION.md): K4/B16/m0=1/2 with unchanged prior/target/96 consumed vectors/six stresses. Exchangeable finite-prior Bellman quotient has23,987 states. All three frozen nested levels16/64/256 completed in22.51CPU seconds, peak206.22MiB. Highest-level ratio-of-mean-risks reduction1.4866%, descriptive two-sided95% interval[1.0203%,1.9956%]; maxstressregression0.7028% passes10%, but stress integration change5.5695% fails1%. No confirmationGO/fullsweep/GPU. Fifteen tests and independent direct-error reconstruction/budget/Bellman/hash audit PASS. Same-family semantic review remains provisional; metric/interval/pooling declarations clarified after round1 without changing outputs. This finite-prior optimum is not a benchmark oracle or an impossibility result for sequential sampling.

## 2026-10-11: Gradient-aware fixed-score headroom

[Derivation](refine-logs/gradient-headroom/DERIVATION_PACKAGE.md), [frozen-before-outcomes plan](refine-logs/gradient-headroom/20261011_run02/EXPERIMENT_PLAN.md), and [decision](refine-logs/gradient-headroom/20261011_run02/DECISION.md). Fixed synthetic score Gram matrix contracts exact nonlinear credit-error second moment, including bias/cross covariance. K4/B16/m0=1/2 means minimum precommitted counts, not adaptive pilots. All455 count vectors,102 unchanged probability vectors,24 coordinate orders and four geometries evaluated. True-p update oracle improves16.1234% over uniform, descriptive95% interval[14.1525%,18.3822%]; only4.3856% over credit oracle,[3.9052%,4.8412%]. One of three informative geometries clears both point5% checks; pooled gradient-specific screen fails. Unequal-norm9.3183% conditional signal is retained without changing benchmark or threshold. Null aligned geometry ratiosNA by centering. Completed CPU run02:17.27s,GPU0. Active suite19tests and exact direct-error/global-minimum/saved-moment/hash/bootstrap audits PASS; no independent-family semantic certification. Run01 audit boolean serialization failure preserved. No deployable method or training GO. Future gradient proposal needs independent evidence of relevant target-task score geometry before declaring another method protocol.

## 2026-10-11: Server score-geometry plan

[Server run plan](refine-logs/score-geometry/SERVER_RUN_PLAN.md) targets frozen Qwen2.5-VL-3B / ShareRobot geometry. Trainer mapping: shifted thought-token mean log-probability, first answer continuation, ratio1 thought PPO component; preserve full global normalization-group scope. Local inventory finds torch CPU but no target checkpoint/replay/ShareRobot data, and missing peft/trl; metadata download URL is explicitly a placeholder. User steered preparation to a server plan before exporter implementation. Four-prompt smoke then128 independent hash-selected prompts; package/path preflight and manifest-freeze CLIs exist. Exporter/analyzer remain not implemented, with no fictional main-run command, real geometry result or GPU job. Geometry audit is descriptive; old synthetic screen remains failed and does not grant training GO.
