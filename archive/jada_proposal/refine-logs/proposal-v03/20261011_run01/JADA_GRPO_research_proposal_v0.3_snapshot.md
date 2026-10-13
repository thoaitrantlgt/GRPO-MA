# Research Proposal: JADA-GRPO v0.3

## Evidence-Gated Thought-Credit Estimation and Continuation Allocation for GRPO-MA

**Version:** 0.3  
**Date:** 2026-10-11  
**Status:** Revised research proposal after completed CPU gates. Current JADA allocation and Jeffreys posterior-credit prototypes do not qualify for escalation to frozen-model or online-training experiments.  
**Predecessor:** [JADA-GRPO v0.2, 2026-09-30](JADA_GRPO_research_proposal.pdf).

JADA-GRPO is retained as the historical project and method identifier. This revision does not present a successful replacement algorithm. It specifies the validated theory, the negative empirical findings, and the conditions under which a materially different method could reopen the research pipeline.

## 1. Executive Summary

GRPO-MA samples several thought prefixes and several answer continuations per thought. Continuation rewards estimate each thought's conditional value; normalization converts those value estimates into relative credits used by the policy loss. Under a fixed continuation budget, estimating these credits reliably is a coherent research problem.

Version0.2 proposed Jacobian-aware adaptive allocation, supported by stabilized pilot statistics and precommitted pilot reuse. Its central hypothesis was that normalization sensitivity would improve allocation beyond reward variance alone, yielding better credit estimates and ultimately more efficient training. Completed CPU experiments do not support that practical-benefit hypothesis under the declared benchmark. The broad JADA comparison increases normalized-advantage MSE by3.24% relative to variance-only allocation. The exact raw-value covariance, analytic Jacobian and separable integer allocator remain correct, but correctness of those components does not establish finite-budget benefit.

Subsequent studies separate three issues: nonlinear objective headroom, limited pilot information, and objective fidelity. Exact true-probability nonlinear oracles show headroom, while deployable pilot-only allocation policies recover little of it. The true-probability local surrogate itself can worsen exact risk, so noisy pilot statistics are not the entire explanation. Fixed synthetic score geometry does not establish a material additional objective-specific benefit or a practical training improvement.

A distinct posterior-credit estimator, with allocation held uniform, reduces mean synthetic update MSE on consumed development vectors. After a predeclared numerical convergence study, however, the original stress cell still worsens17.54%, exceeding its10% regression threshold. This prototype also remains stopped. A favorable average is not sufficient to override a declared stress criterion.

The immediate plan is therefore to consolidate this evidence and keep the expensive stages gated. The research question survives; neither JADA nor posterior-credit is promoted to an effective training method. A future candidate needs a concrete mechanism, a declared estimation/update target, matched baselines and costs, numerical validation, and untouched confirmation after development.

## 2. Problem Anchor and Scope

**Research question:** Can an implementable estimator or continuation-allocation rule improve conditional thought-update estimation under a fixed sampling budget, beyond uniform sampling and simple scale calibration, without unacceptable regression across the declared regimes?

Condition on a prompt, sampled thought prefixes and a fixed continuation-policy snapshot. Define the conditional value

\[
\mu_j=\mathbb E[R\mid x,T_j,\pi_{\mathrm{old}}].
\]

The reward R is the specified reward-function sum. In the Bernoulli CPU model, \(\mu_j=p_j\); this model does not represent every reward used by the trainer. Error in estimating values, error introduced by normalization, and error in the resulting policy-loss gradient are separate quantities.

This proposal concerns conditional estimation before an optimizer step. It does not treat thought sampling, answer-policy gradients, clipping, KL regularization, optimizer dynamics or task performance as consequences of a scalar-credit MSE result. Any extension to those objects needs its own assumptions and measurements.

Current scope is CPU evidence and source-grounded analysis. No new model, dataset, GPU rental or training run is initiated by this revision.

## 3. What Changes from v0.2

| v0.2 component | v0.3 treatment |
|---|---|
| Jacobian-aware allocation as the proposed main method | Retain as an evaluated hypothesis that failed its current practical gate |
| Fixed-weight pilot reuse | Retain the exact value-level guarantee and sampling assumptions |
| Delta-method information score | Retain as a local surrogate; explicitly separate it from exact nonlinear risk |
| Surrogate oracle as an information ceiling | Label it surrogate-specific; use exhaustive nonlinear oracles for exact-risk headroom |
| Smaller advantage error leading to better training | Separate credit, conditional update, frozen-model and online claims |
| Synthetic validation before expensive work | Mark the broadened CPU stage completed with a negative result |
| Frozen-model pool and LoRA-gradient evaluation | Preserve as conditional future stages, currently blocked by the evidence gate |
| Online RL, model scaling and multi-task expansion | Defer; the current methods provide no escalation basis |
| Expected positive accuracy/token/GPU-hour examples | Remove from the evidence narrative; no such gains were measured |

The original Gate1, section32/page23, required consistent Jacobian-aware improvement over variance-only allocation across configurations and budgets. It was not passed. This revision records that outcome rather than redefining the historical gate to obtain a positive result.

## 4. Mathematical Foundation That Remains Valid

### 4.1 Fixed-policy two-stage sampling

Each thought receives \(m_0\) pilot continuations. A pilot-measurable rule assigns positive main counts \(M_j\), with

\[
B_{\mathrm{total}}=K m_0+B_{\mathrm{main}},\qquad
\sum_j M_j=B_{\mathrm{main}},\qquad M_j\ge1.
\]

The allocation is finalized before main sampling. Main continuations are fresh draws under the same frozen policy. Pilot streams are independent across thoughts; main streams are conditionally independent given the pilot history. Additional outcome-dependent stopping is outside this design.

For a scalar \(\lambda\) committed before current pilot outcomes, use raw sample means:

\[
\widetilde\mu_j=\lambda\bar R_j^P+(1-\lambda)\bar R_j^M.
\]

Posterior-smoothed pilot means may be used to construct allocation coefficients, but replacing the raw pilot means in this final estimator would change its unbiasedness argument.

### 4.2 Exact reuse covariance

Conditional mean-zero main errors eliminate pilot/main cross terms even when all main counts depend on the full pilot history. Under finite reward variances \(\sigma_j^2\), the unconditional identities are

\[
\mathbb E[\widetilde\mu]=\mu,
\qquad
\operatorname{Cov}(\widetilde\mu)=
\operatorname{diag}\!\left\{
\sigma_j^2\left[\frac{\lambda^2}{m_0}
 +(1-\lambda)^2\mathbb E\!\left(\frac1{M_j}\right)\right]
\right\}.
\]

Shared allocation decisions do not, by themselves, create off-diagonal value covariance under these assumptions. Naive pooled adaptive reuse has a random pilot weight and does not inherit this guarantee. Neither unbiased values nor diagonal value covariance implies unbiased normalized credits or unbiased policy gradients.

### 4.3 Historical normalization and analytic Jacobian

The original CPU target was

\[
P=I-\mathbf1\mathbf1^\top/K,\quad z=P\mu,\quad
v=\|z\|^2/K,\quad s=\sqrt{v+\epsilon},\quad
f_\epsilon(\mu)=z/s,\quad\epsilon=10^{-6}.
\]

Its Jacobian is exactly

\[
J_f=\frac P s-\frac{zz^\top}{K s^3}.
\]

The squared column norms are

\[
q_j=\|J_{f,:j}\|^2
=\frac{1-1/K}{s^2}
-\frac{z_j^2(v+2\epsilon)}{K s^6}.
\]

These expressions have been checked numerically. At K=2 the column norms are equal, so the Jacobian factor does not distinguish groups beyond their variance; K=1 has no relative group-credit signal.

### 4.4 What the allocator actually optimizes

The local delta approximation gives

\[
\mathbb E\|f_\epsilon(\widetilde\mu)-f_\epsilon(\mu)\|^2
\approx\operatorname{tr}(J_f\operatorname{Cov}(\widetilde\mu)J_f^\top).
\]

For a common fixed reuse coefficient, the main-allocation part is proportional to

\[
\sum_j\frac{c_j}{M_j},\qquad c_j=\sigma_j^2 q_j\ge0.
\]

The unconstrained continuous rule is \(M_j\propto\sqrt{c_j}\). With positive integer counts and equal per-continuation costs, start each count at its mandatory minimum and allocate each additional sample to the largest decrease \(c_j/[M_j(M_j+1)]\). These decreases are nonincreasing within each group, yielding an optimum of the separable integer surrogate.

This proof concerns fixed coefficients and the surrogate. It does not establish minimum exact nonlinear MSE, finite-budget validity of noisy pilot coefficients, token-cost optimality or training improvement. See the [CPU derivations](experiments/cpu_gate/DERIVATIONS.md).

## 5. Why Correct Local Theory Was Insufficient

Let \(e=\widetilde\mu-\mu\), \(\delta=f_\epsilon(\widetilde\mu)-f_\epsilon(\mu)\), \(L=J_f e\), and \(Q=\delta-L\). The exact identities are

\[
\operatorname{MSE}_A
=\frac{\|\mathbb E\delta\|^2+\operatorname{tr}\operatorname{Cov}(\delta)}K
=\frac{\mathbb E\|L\|^2+2\mathbb E[L^\top Q]+\mathbb E\|Q\|^2}K.
\]

The local surrogate retains only the linear term. Remainder energy and its cross term can be substantial. In the consumed-data K4/total16/m0=2 diagnosis, uniform exact MSE is0.742004, while the linear prediction is5.275466. Squared bias accounts for31.67% of the exact risk, so attributing all surrogate failure to bias would be incorrect.

At equal true probabilities, the target is zero and uniform normalized-credit bias is zero by symmetry. Exact MSE is0.972347, but the linear prediction is46875. This supplies an explicit case in which local calibration fails despite zero bias. Only10/96 consumed vectors passed both descriptive tolerances of10% risk calibration and1% fixed-allocation regret. These tolerances are diagnostics, not a universal theorem.

Dispersion, leverage and sampling noise characterize the failure region, but dispersion alone does not certify Taylor validity or allocation ranking. Relevant tail/remainder control is required. The [surrogate diagnosis](refine-logs/surrogate-diagnostic/20261011_run02/RESULTS.md) contains the full decomposition and retained dispersion paths.

## 6. Trainer-Aligned Target and Update

### 6.1 Two normalization conventions

The inspected trainer gathers thought-value estimates and applies `.std().clamp(min=1e-4)`. The local CPU check confirms sample-standard-deviation semantics in the installed environment. Define the implementation-aligned ideal target by substituting true conditional values:

\[
A^*=h_\tau(\mu)=
\frac{P\mu}{\max\{\sqrt{\|P\mu\|^2/(K-1)},\tau\}},
\qquad \tau=10^{-4},\quad K\ge2.
\]

This is distinct from \(f_\epsilon\), and from \(\mathbb E[h_\tau(V)]\). Historical results remain attached to their original targets; they are not silently relabeled trainer-exact. The local PyTorch version also differs from the pinned training environment, which has not been executed for this study.

When the floor is inactive, the derivative of h is

\[
J_h=P/s_h-zz^\top/[(K-1)s_h^3],\quad
s_h=\sqrt{\|z\|^2/(K-1)}.
\]

In the strictly floor-active region it is \(P/\tau\); the threshold requires explicit nonsmooth handling. Empty masks, all-invalid groups, singleton groups and cross-rank grouping must be specified before any integration.

### 6.2 Conditional thought update

The trainer reduces thought loss by averaging active tokens within each thought, then averaging over thoughts. It uses one thought sequence per prefix rather than counting its repeated answer copies as separate thought updates. Define

\[
s_j=\frac1K\frac1{|\mathcal T_j|}
\sum_{t\in\mathcal T_j}\nabla_\theta\log\pi_\theta(t),\qquad
S=[s_1,\ldots,s_K].
\]

Set the score to zero for an empty thought mask. At the old-policy point, inside PPO clip bounds, the reward-surrogate ascent gradients are

\[
g^*=S A^*,\qquad\widehat g=S\widehat A.
\]

With S fixed conditional on thoughts, masks and policy state,

\[
\mathcal L_{\mathrm{update}}
=\mathbb E\|\widehat g-g^*\|^2
=\operatorname{tr}(G\operatorname{Cov}(\delta))
 +\mathbb E[\delta]^\top G\mathbb E[\delta],
\qquad G=S^\top S,\quad\delta=\widehat A-A^*.
\]

This is the primary estimation endpoint for a future trainer-aligned method. Scalar advantage MSE corresponds to a special Gram weighting and is a secondary diagnostic. If scores are reward-correlated or reveal information about unknown values, their joint distribution and conditioning must be retained.

Away from ratio1, PPO's active clipping branch depends on the advantage sign. Estimated and target credits can select different branches. Evaluate the exact clipped-gradient difference instead of assuming one fixed Gram describes all updates.

### 6.3 Thought, answer and KL losses

The answer branch normalizes individual continuation rewards separately and has its own token/sequence reduction. Holding that branch and KL fixed isolates thought-credit estimation. Scaling thought advantage changes its relative weight against these unchanged gradients; it is not universally equivalent to changing the full optimizer learning rate.

For any future variable-count design, declare whether answer statistics and losses are per-thought weighted or continuation weighted, whether pilot answers enter training, and how reference normalization avoids allocation-dependent weighting. Merely averaging loss within each thought does not establish that a shared reward normalizer is allocation invariant.

The conditional group-credit target is not automatically an unbiased expected-reward policy gradient: group-dependent baselines, random normalization and token-mean weighting matter. The [target contract](idea-stage/docs/ADVANTAGE_TARGET_CONTRACT.md) records the full assumptions and source mapping.

## 7. Completed Empirical Evidence

Define relative reduction as \((R_{\mathrm{baseline}}-R_{\mathrm{candidate}})/R_{\mathrm{baseline}}\). Negative values mean worse error. Each interval below belongs to its own original protocol and confirmation sample; rows are not pooled into a meta-analysis.

| Study / comparison | Relative reduction | Original paired95% interval | Interpretation |
|---|---:|---:|---|
| Broad JADA vs variance-only, budget-precommitted reuse | -3.24% | [-3.59%, -2.89%] | Original practical allocation gate failed |
| Adaptive exact nonlinear oracle vs uniform | 10.22% | [8.87%, 11.57%] | True-value information headroom |
| Fixed exact nonlinear oracle vs uniform | 6.70% | [5.24%, 8.16%] | Headroom without pilot-adaptive oracle choices |
| True-value local-surrogate oracle vs uniform | -4.78% | [-6.73%, -2.83%] | Surrogate failure persists without pilot estimation noise |
| Pilot-only Jeffreys posterior allocation vs uniform | 0.0125% | [0.0085%, 0.0165%] | Too small for the materiality gate |
| Pilot-information primary, fixed-main/m0=4 | -0.0660% | [-0.1103%, -0.0218%] | More pilots did not rescue the primary |
| Pilot-information primary, fixed-total/m0=4 | 0.0832% | [0.0304%, 0.1361%] | Matched-total gain remained immaterial |
| Adaptive weighted oracle vs uniform, synthetic Grams | 11.36% | [10.36%, 12.35%] | Synthetic known-information headroom |
| Adaptive weighted oracle vs adaptive scalar oracle | 3.09% | [2.39%, 3.79%] | Positive additional effect below the declared5% materiality criterion |
| Known-Gram pilot gradient-JADA vs uniform | -1.62% | [-1.98%, -1.27%] | No practical rescue from the weighted plug-in |

Sources: [broad gate](refine-logs/cpu-estimator-gate/20261011_run01/summary.json), [nonlinear oracle](refine-logs/nonlinear-oracle-gate/20261011_run01/summary.json), [posterior allocator](refine-logs/posterior-allocator-gate/20261011_run01/summary.json), [pilot information](refine-logs/pilot-information-gate/20261011_run01/summary.json), and [synthetic gradient gate](refine-logs/gradient-objective-gate/20261011_run01/summary.json).

### 7.1 Broad benchmark coverage

The broad study covered K in {2,4,8,16}, total budgets in {16,32,64}, and m0 in {1,2,4}, retaining all26 settings with at least one main sample per thought. Allocation {uniform, variance-only, JADA, true-value surrogate oracle} was crossed with reuse {main-only, naive pooled, fixed0.5, budget-precommitted}. It included924 exact cells and1728 paired Monte Carlo cells, totaling5,184,000 paired trials.

The heterogeneous population was fixed before method outcomes: equal-weight iid Beta(.5,.5), Beta(2,2), Beta(8,8), Beta(.5,3), Beta(3,.5), and a shared-location logistic-normal component. The latter uses location N(0,1), scale exp(U(log.15,log3)), and independent standard-normal coordinate offsets. The original six stress regimes were all retained:

| Regime | K4 probabilities |
|---|---|
| symmetric-wide | [.95,.55,.45,.05] |
| symmetric-mid | [.90,.52,.48,.10] |
| all-equal | [.50,.50,.50,.50] |
| near-equal | [.55,.52,.48,.45] |
| asymmetric | [.90,.80,.70,.10] |
| sparse-extreme | [.99,.99,.50,.01] |

Paired population intervals cluster repeated settings within independent probability vectors and respect the mixture strata. Exact reward-state evaluation removes reward-simulation error; it does not remove uncertainty over the probability-vector population. Pilot fraction, dispersion and leverage breakdowns are descriptive and do not override the primary estimand.

### 7.2 Normalization and posterior-credit follow-ups

The fixed-half amplitude probe retained the historical target and uniform sampling. At total16 it reduced mean scalar MSE13.49% on consumed data, but11/18 stress/budget cells regressed more than10%; it was rejected. Its variance reduction does not by itself imply useful new credit information or a training algorithm.

The posterior-credit prototype instead estimates the trainer-aligned target directly:

\[
\widehat A(C)=\mathbb E[h_\tau(p)\mid C]
\]

under fixed iid Jeffreys prior and uniform four Bernoulli samples per thought. Conditional expectation minimizes integrated fixed-Gram posterior risk under that prior, but does not guarantee pointwise superiority across regimes. This estimator is distinct from the posterior allocation policies and from normalizing a posterior mean.

At the largest declared convergence level, its consumed-data mean update MSE is0.472861, versus0.565857 for plug-in uniform and0.513641 for the prior-frozen scalar control. Reductions are16.43% and7.94%. However, sparse-extreme/paired-correlated risk is0.041711 versus plug-in0.035487: a17.54% regression. The complete development gate remains failed.

Eight independent Sobol scrambles reached32768 base points each. Half-scramble max credit difference0.000257, adjacent-level max change0.001278, and normalized max stress-risk change0.0434% passed the predeclared numerical criteria. This supports the stop recommendation after numerical stabilization; it is not a rigorous continuous-prior error certificate. See [convergence results](refine-logs/posterior-credit-convergence/20261011_run01/RESULTS.md).

All posterior-credit and normalization follow-ups use consumed development vectors. Their gains are descriptive; they have no new population-confidence claim. No prior, geometry, favorable subgroup or quadrature level was chosen to replace a failed primary result.

## 8. Research Questions and Claim Status

| Question | Current answer | Permissible claim |
|---|---|---|
| Does fixed-weight pilot reuse preserve raw values? | Yes, under the stated two-stage assumptions | Exact conditional-sampling theorem |
| Is the analytic Jacobian/greedy surrogate allocator correct? | Yes, for its declared normalization and separable objective | Component correctness |
| Does JADA reliably improve finite-budget credit estimation? | Not on the declared benchmark | The practical hypothesis failed this gate |
| Is exact nonlinear allocation headroom absent? | No; true-value oracles improve risk | Information headroom, not deployability |
| Does a gradient weighting rescue the current method? | No material objective-specific gate or practical gain established | No current escalation basis |
| Does stable posterior credit meet the declared robustness criterion? | No; one stress cell remains above the regression threshold | Stop this particular prototype |
| Is training compute efficiency improved? | Not tested | No claim |

The defensible contribution currently consists of exact conditional estimator analysis and a reproducible account of when correct local allocation theory fails to yield practical finite-budget gains. Its novelty and publication value are not certified by this document. A targeted related-work update would be needed before any submission claim.

## 9. Revised Three-Block Roadmap

### BlockA: theory and CPU evidence — completed

Preserve derivations, boundary cases, raw tables, exact/paired Monte Carlo code, frozen protocols, audits and all negative gates. Document target differences and conditional-update assumptions. The present decision is to stop the tested allocator family and Jeffreys posterior-credit prototype.

There is no automatic additional prior sweep, normalization search or method variant. A future idea must identify a concrete missing information source or objective mechanism, explain why existing failed controls do not address it, and specify its runtime information and cost before a new experiment is opened.

### BlockB: conditional estimator/update confirmation — conditional, not started

If a materially distinct candidate survives a frozen development contract, declare its exact target, budget, baselines, geometries, sample size and paired inference plan before accessing untouched confirmation. It must use deployable information and preserve all named stress regimes. Reusing consumed vectors as confirmation is prohibited.

Only after that CPU gate passes should a separately scoped frozen-model pilot be considered. Reuse available logs or recorded pools first where appropriate. A new reference pool would need disjoint estimator/reference draws, declared reference precision, fixed policy/prompt grouping, matched token reductions and explicit reward-score dependence. LoRA or a declared parameter subset is an economical conditional-gradient measurement, not a substitute for validating the objective.

### BlockC: online efficiency — deferred

Online RL requires a successful frozen-model/update gate and a separately reviewed implementation contract. Start with one model and task, matched evaluation, multiple predeclared seeds and attributed baselines. Use a two-stage uniform control if changing the sampling/reuse pipeline. Scaling to larger models or more tasks is contingent on a material effect.

Measure generated tokens, wall-clock time and GPU-hours in addition to continuation counts. Equal answer counts are not equal cost. No accuracy, token-saving or GPU-hour target is asserted here, and no compute purchase is scheduled.

## 10. Conditions to Reopen the Pipeline

These conditions apply to a future declared design; they do not rewrite historical gates.

### Development screen

- Fix the estimator, reward model, runtime information, sample cost, prior if any, target normalizer and clipping regime before its outcomes.
- Compare equal-cost uniform plug-in with a scalar-calibration control receiving the same information; freeze calibration before confirmation. If adaptive allocation is claimed, include variance-only and the required reuse/pipeline controls.
- Require >5% mean conditional update-MSE reduction against both uniform and scalar calibration, no >10% exact regression in any named primary stress/budget cell, and explicitly declared numerical-stability checks.
- Retain every predeclared geometry, regime and feasible setting. A failure stops that declared candidate; any later change is a new hypothesis and version.

### Untouched CPU confirmation

Require the lower95% paired relative-reduction limit to exceed5% against both primary controls, with the same stress and numerical requirements. Declare sample size and interval construction before access; report the full result regardless of outcome. Oracle controls are information diagnostics, not deployable competitors.

### Frozen-model and online gates

Require matched token/sequence reductions, no runtime leakage, reference-precision checks, direct gradient agreement beyond a scalar-scale explanation, and acceptable measured overhead. If the answer or KL objective changes, isolate that change with its own controls. Full-training claims require full training evidence, rather than an estimation bound or projected-gradient diagnostic alone.

## 11. Metrics and Statistical Practice

The primary conditional metric is update MSE. Report squared bias and variance, target alignment, gradient magnitude, and direction error with its defined-probability mass. Retain scalar advantage MSE, value bias/covariance, sign error and rank information as diagnostics.

When the target update is zero, direction and relative-to-target metrics are undefined; report absolute spurious-update energy. A cosine conditional on nonzero norms must not silently exclude the probability of zero estimated updates. Null geometries remain explicit controls rather than opportunities to manufacture relative gains.

Use exact count-state enumeration for manageable Bernoulli cases. For larger cases, pair Monte Carlo randomness across methods and report reward-simulation uncertainty separately from probability-vector population uncertainty. Cluster repeated budgets, pilot sizes or Grams within the independent unit; they are not extra independent observations. Mixture-component weighting, primary comparisons, seeds, feasible-cell exclusions and all thresholds must be committed before outcomes.

Consumed confirmations are development data for later analyses. Quadrature replicates diagnose integration, not population generalization. Numerical tolerance does not constitute a formal integration-error certificate. Claims must remain attached to the target, budget, population and score model actually evaluated.

## 12. Implementation and Verification

The existing training reference remains unchanged. CPU implementations live in [experiments](experiments), with explicit run protocols and raw results under [refine-logs](refine-logs). The relevant modules cover the original gate, nonlinear oracles, posterior allocators, surrogate diagnosis, fixed-amplitude probe, and posterior-credit convergence.

Required invariants include feasible positive integer counts, exact budget conservation, pilot-only decisions, precommitted reuse, fixed policy during sampling, analytic-Jacobian checks, conditional mean-zero main errors, complete small-case enumeration, and consistent target normalization. For a future trainer bridge, add prompt/rank ordering, per-thought answer weights, pilot/main inclusion, clipping/sign changes, std-floor, singleton and empty-mask checks.

The latest posterior-credit convergence run passed35 repository tests, local finite-prior/moment audits and saved-table reconstruction. Some earlier reviews are same-family provisional; other diagnostics have local-only review. This v0.3 rewrite has source/result consistency checks and no new independent reviewer clearance. None of these checks is evidence of model-training effectiveness.

## 13. Risks and Limits

| Risk | Consequence and mitigation |
|---|---|
| Local surrogate failure | Prefer exact nonlinear evaluation; label any approximation and its validity conditions |
| Weak pilot information | Oracle headroom is insufficient; require a deployable information model |
| Prior misspecification | Bayes-optimal average can coexist with stress regressions; retain robustness criteria |
| Target/implementation mismatch | Freeze normalization, grouping, reduction and runtime version before integration |
| Score/reward dependence | Preserve joint samples; do not multiply unrelated marginal covariances |
| Scalar attenuation masquerading as new credit | Compare a matched-information scale control and separate thought/answer/KL contributions |
| Consumed-data optimism | Keep development and untouched confirmation labels; do not recycle intervals |
| Compute accounting mismatch | Measure tokens and elapsed/device cost before efficiency claims |

These results do not prove that adaptive sampling, Bayesian credits or normalization changes can never help. They rule out escalation of the tested methods under the declared gates. A different target or information source must be justified explicitly, rather than introduced as a posthoc rescue.

## 14. Current Decision and Deliverables

**Current decision:** retain the research question and validated conditional theory; stop the evaluated JADA and Jeffreys posterior-credit routes. Frozen-model and online-training stages remain deferred. GPU cost of the completed research gates and diagnostics is0.

The immediate deliverables are this revised proposal, the linked [consolidated evidence report](refine-logs/ESTIMATOR_RESEARCH_REPORT.md), the [target/update contract](idea-stage/docs/ADVANTAGE_TARGET_CONTRACT.md), and preserved reproducible protocols/results. No successful replacement method is claimed or selected by this revision.

A reopening memo must state the distinct mechanism, why it could address the identified failure, what information it uses, what objective it estimates and exactly what would falsify it. Without that memo and a surviving development screen, the scientifically supported next action is to close this branch rather than expand compute.

## 15. Evidence Index

| Item | Source |
|---|---|
| Original proposal and Gate1/Gate2 | [v0.2 PDF](JADA_GRPO_research_proposal.pdf), sections32 and36, pages23 and25 |
| Initial same-family provisional review | [JADA_GRPO_REVIEW.md](JADA_GRPO_REVIEW.md) |
| Broad estimator gate | [results](refine-logs/cpu-estimator-gate/20261011_run01/RESULTS.md), [protocol](experiments/cpu_gate/protocol.json) |
| Nonlinear oracle headroom | [results](refine-logs/nonlinear-oracle-gate/20261011_run01/RESULTS.md) |
| Pilot-only posterior allocator | [results](refine-logs/posterior-allocator-gate/20261011_run01/RESULTS.md) |
| Pilot-information and allocator convergence | [information](refine-logs/pilot-information-gate/20261011_run01/RESULTS.md), [convergence](refine-logs/posterior-convergence/20261011_run01/RESULTS.md) |
| Fixed synthetic gradient objective | [results](refine-logs/gradient-objective-gate/20261011_run01/RESULTS.md) |
| Local-surrogate mechanism diagnosis | [results](refine-logs/surrogate-diagnostic/20261011_run02/RESULTS.md) |
| Fixed-half normalization probe | [results](refine-logs/normalization-probe/20261011_run01/RESULTS.md) |
| Trainer target and update mapping | [contract](idea-stage/docs/ADVANTAGE_TARGET_CONTRACT.md), [trainer source](trainer/grpo_ma_trainer.py) |
| Posterior-credit prototype and convergence | [prototype](refine-logs/posterior-credit/20261011_run01/RESULTS.md), [convergence](refine-logs/posterior-credit-convergence/20261011_run01/RESULTS.md) |

This is a revision grounded in local evidence, not a fresh literature or novelty audit. External bibliographic and venue claims should be checked separately before submission. The original v0.2 PDF and historical protocols remain preserved.
