# Critical review of JADA-GRPO v0.2

Reviewed 2026-10-11. Source: `JADA_GRPO_research_proposal.pdf`, dated 2026-09-30, 43 pages. This is a proposal review, not an audit of completed JADA training results. The proposal's example accuracy/token/GPU-hour improvements are hypothetical.

Reviewer route: fresh Codex reviewer, `gpt-6-astra`, reasoning `ultra`; agent `/root/jada_reviewer`. `review_independence: same-family`; `acceptance_status: provisional`. These describe reviewer independence, not scientific acceptance. Full prompts, responses, and decisions are in [.aris/traces/research-review/2026-10-11_run01](.aris/traces/research-review/2026-10-11_run01/).

## Assessment

The research question is coherent, and the fixed-weight reuse argument is correct under its stated sampling assumptions. The current default configuration does **not** establish sufficient estimator benefit to pass the proposal's own gate for expensive online RL. Narrow the novelty claim, complete the estimator analysis, resolve answer normalization and trainer semantics, and run a broader inexpensive estimator study before training.

The main obstacles are practical benefit and objective fidelity, rather than an invalid thought-value unbiasedness proof. Reduced advantage error gives a tighter gradient-error bound; it does not guarantee a reduction in actual gradient error or improved training performance.

## Evidence and scope

- Read all 43 pages using page-labeled extraction; visually checked pages 8 and 16 to confirm the variance regularizer and answer-loss equations.
- Inspected `README.md`, `scripts/README.md`, `metadata/README.md`, and relevant generation, reward normalization, and loss code in `trainer/grpo_ma_trainer.py`.
- Checked primary-source arXiv records and relevant full-text sections for GRPO-MA, VIGOR, and EPIG-Tree. This was a targeted comparison, not exhaustive novelty certification.
- Ran exact finite Bernoulli enumeration for six probability vectors at the proposed default, plus a Jacobian finite-difference check. The fresh reviewer independently inspected and reproduced these calculations.
- No model training or benchmark evaluation was performed. The repository supplies a fixed-M reference trainer and ShareRobot trajectory task, not JADA results. Math/code task integrations are absent from this release. The metadata README explicitly calls its companion dataset URL a placeholder, so dataset access still needs verification.

## What is mathematically sound

Condition on the prompt, sampled thoughts, and unchanged policy. Let pilot mean error be `eP_j`, main mean error `eM_j`, and let main counts depend only on the full pilot history. With independent pilot streams, conditionally independent fresh main streams, and every `M_j >= 1`,

\[
 E[e^M_j\mid\mathcal F_P]=0,\qquad
 \widetilde\mu_j=\lambda\bar R^P_j+(1-\lambda)\bar R^M_j,
 \qquad E[\widetilde\mu_j]=\mu_j.
\]

This requires raw pilot means in the final estimator, as the proposal's pseudocode correctly uses; posterior-smoothed means would change the proof. The fixed coefficient must be committed before current pilot outcomes. Allocation must be finalized before main sampling; additional outcome-dependent stopping would require separate analysis. The proposal correctly distinguishes unbiased values from nonlinear normalized advantages.

For this design, shared allocation decisions do **not** automatically induce nonzero cross-thought value-error covariance. Conditional mean-zero main errors eliminate pilot/main cross terms, and independent pilot errors give

\[
 \operatorname{Cov}(\widetilde\mu)
 =\operatorname{diag}\left[\sigma_j^2\left(
 \frac{\lambda^2}{m_0}+(1-\lambda)^2E\left[\frac1{M_j}\right]
 \right)\right].
\]

Writing `q_j = ||J_:j||²` and `c_j = sigma_j² q_j`, the leading-order vector squared error is

\[
 E\|f(\widetilde\mu)-f(\mu)\|^2
 \approx \frac{\lambda^2}{m_0}\sum_jc_j
 +(1-\lambda)^2\sum_jc_jE[1/M_j].
\]

Thus, with a common precommitted lambda, the pilot contribution is allocation-independent at first order and the main allocation retains the `c_j/M_j` structure. **The allocator is not invalid simply because it reuses pilots.** The missing derivation should be added, along with its local approximation limits. Conditional on realized pilot errors, the corresponding first-order risk includes `lambda² ||J eP||²`; treating noisy plug-in J and variance estimates as truth does not establish finite-budget optimality.

For population variance normalization, put `z = mu - mean(mu)`, `v = ||z||²/K`, and `s = sqrt(v + eps)`. Then

\[
 J=\frac{I-\mathbf1\mathbf1^T/K}{s}
 -\frac{zz^T}{Ks^3},\qquad
 q_j=\frac{1-1/K}{s^2}
 -\frac{z_j^2(v+2\epsilon)}{Ks^6}.
\]

For negligible epsilon and positive v, `q_j = [(K-1)/K - z_j²/(K v)]/v`. This reveals the proposed influence factor as a leverage correction. Analytic computation avoids materializing an autograd Jacobian; autograd remains a useful validation reference. At `K=2`, column norms are equal, so this factor cannot distinguish thoughts beyond variance; at `K=1` there is no relative-credit signal. These are useful boundary cases, not claims that `K=4` must fail.

The continuous square-root rule and discrete diminishing-marginal greedy rule are correct for fixed nonnegative coefficients, equal per-answer costs, a feasible positive minimum, and the separable surrogate. They are not proofs of minimum exact nonlinear MSE under noisy adaptive coefficients. An oracle for this surrogate is a diagnostic comparator, not a certified lower bound on actual finite-budget MSE.

## Exact checks at the proposed default

Settings: `K=4`, `m0=2`, `B_main=8`, `B_total=16`, `lambda=0.5`, `eps=1e-6`, Beta(0.5,0.5), one mandatory main answer per thought, and smallest-index greedy tie breaking, matching the proposal's `max` pseudocode. We enumerate all 81 pilot count configurations and every conditional main count configuration; there is no Monte Carlo sampling error. Values below are `E[||A_hat-A_true||²/K]` for this finite Bernoulli model, not LLM results.

| True success probabilities | Uniform / variance-only | JADA fixed reuse | Relative change in MSE |
|---|---:|---:|---:|
| [0.95, 0.55, 0.45, 0.05] | 0.203056515 | 0.201569297 | 0.73% lower |
| [0.90, 0.52, 0.48, 0.10] | 0.280223984 | 0.288822868 | 3.07% higher |
| [0.50, 0.50, 0.50, 0.50] | 0.972346959 | 0.988913831 | 1.70% higher |
| [0.55, 0.52, 0.48, 0.45] | 1.649356650 | 1.671195761 | 1.32% higher |
| [0.90, 0.80, 0.70, 0.10] | 0.201161472 | 0.211476383 | 5.13% higher |
| [0.99, 0.99, 0.50, 0.01] | 0.068028536 | 0.063511898 | 6.64% lower |

At `m0=2`, the posterior expected reward-variance proxy has only two values: `5/48` for agreeing pilots and `3/16` for mixed pilots. Their ratio is 1.8. Greedy variance-only allocation with eight main answers and four mandatory first answers gives `[2,2,2,2]` for every pilot history: even the largest second-sample marginal gain is smaller than the smallest first-extra-sample gain. Thus its equality with uniform in this table is an exact default-budget effect, not a bug in the diagnostic.

For uniform allocation, two pilots plus two main samples with lambda 0.5 exactly equal the ordinary four-answer pooled mean. The table therefore includes a genuine uniform 16-answer baseline. JADA and variance allocation often differ, but different allocations alone do not establish lower error. Tie permutations should be checked in the broader study; the table preserves the proposal's deterministic rule.

Fixed-reuse thought-value biases are zero to floating-point precision (below approximately `5e-16`); nonlinear advantage biases are not zero. Naive pooled adaptive reuse can have nonzero value bias: for `[0.9,0.52,0.48,0.1]`, its first-coordinate bias is approximately `0.01122087`. On the main example, naive reuse has slightly lower advantage MSE than fixed reuse despite biased values. Bias control and minimum MSE are different goals.

The analytic Jacobian agrees with central finite differences at `[0.83,0.50,0.50,0.17]` to maximum absolute error `5.68e-10`, with perturbation `1e-5`.

**Decision:** these checks do not pass the proposal's Gate 1 (section 32, page 23). They justify inexpensive revision and broader diagnostics, not online RL expansion or a general impossibility claim.

Saved diagnostic code and full numerical output: [diagnostic_checks.py](.aris/traces/research-review/2026-10-11_run01/diagnostic_checks.py) and [diagnostic_results.json](.aris/traces/research-review/2026-10-11_run01/diagnostic_results.json). Intermediate extraction and rendered pages are under `tmp/pdfs/`.

## Findings and minimum fixes

### 1. High: the current default has weak or negative finite-budget benefit

Sections 29–32 and 55 propose the exact settings tested above. The core contribution requires beating variance-only with real finite pilots, rather than only oracle coefficients. Half the default answer budget is spent on pilots; the main budget leaves little allocation flexibility. Near-equal values also make a local Taylor approximation poor: at exactly equal true probabilities, true advantage is zero, but tiny epsilon can amplify sampled differences to an advantage vector with roughly unit per-coordinate squared magnitude.

Minimum fix: test multiple predeclared regimes, budgets and feasible pilot fractions; report oracle versus plug-in gaps and approximation error, including ties and sparse/all-equal groups. Retain negative results. A larger budget or different pilot schedule is a hypothesis to test, not an assumed solution.

### 2. High: equal loss weights do not fully preserve answer-objective semantics

Section 21 (page 16) correctly gives each thought equal outer loss mass. However, inherited trainer lines 800–810 normalize answer rewards using the flat sampled mixture. Under ragged sampling its population mixture has weights `M_j / sum_l M_l`, rather than equal thought weights. Reweighting the subsequent loss does not undo this change in baseline and scale. Finite-sample self-normalization also remains biased.

Specify the intended answer normalization explicitly. For equal-thought target moments, a natural empirical definition is

\[
 b=\frac1K\sum_j\frac1{M_j}\sum_mR_{jm},\qquad
 u=\frac1K\sum_j\frac1{M_j}\sum_mR_{jm}^2,\qquad
 s_R=\sqrt{\max(u-b^2,0)+\epsilon_R}.
\]

These restore equal-thought mixture semantics; they do not by themselves prove an exactly unbiased PPO/GRPO gradient. Document main-only versus pilot/main answer training and regularization. The proposal's pseudocode passes only main answers to the answer loss, so pilots contribute to thought estimation but not directly to answer-token learning. If answer KL is retained, give it the same per-thought reduction. Compare all adaptive baselines with identical normalization, reuse, KL and token reduction; separately retain the original uniform implementation as a reference.

### 3. High: novelty is substantially narrower than the three contribution labels suggest

GRPO-MA v4 already derives the normalization derivative and uniform-M delta-method variance in section 3.3.2, Eq. (2). Replacing common M by fixed per-thought counts and summing the coordinate variances yields JADA's trace surrogate by direct extension. Cite this inheritance explicitly. [GRPO-MA full text](https://arxiv.org/html/2509.24494v4)

EPIG-Tree section 5.1, Theorem 1 / Eq. (9), already solves a sensitivity-weighted inverse-sample-count objective using a square-root cost-aware allocation law. Its sensitivity concerns the policy gradient; JADA uses normalization sensitivity. The optimization template is established, while the target quantity differs. [EPIG-Tree full text](https://arxiv.org/html/2609.20004v1)

VIGOR allocates rollouts across prompts using group reward variance. JADA allocates continuations across fixed thoughts within one prompt. This distinction is useful, but pilot-driven adaptive rollout budgets are not new in general. [VIGOR full text](https://arxiv.org/html/2607.22002v1)

The defensible candidate contribution is a finite-budget procedure for normalized thought-credit estimation, with correct pilot reuse and measured benefit after all costs. The inspected sources do not establish that no other work has already done this. Update reference 2 to the latest GRPO-MA title. Retain the proposal's section 25.4 restriction against first-adaptive-branching claims.

### 4. Medium-high: constant lambda introduces a variance floor across growing budgets

With fixed m0 and nonvanishing lambda, variance is at least `lambda² sigma_j²/m0`, even as main samples grow. This limit is acknowledged qualitatively by the proposal's variance-optimality risk, but its budget ablations need the explicit analysis. For uniform allocations, `m0=2`, `lambda=0.5`, and total budget 64 at K=4 give main count 14. The value variance is `sigma²/7`, versus `sigma²/16` for a pooled 16-answer uniform estimate, a ratio `16/7 ≈ 2.286`.

Minimum fix: keep fixed lambda 0.5 as an ablation and include a budget-dependent, precommitted coefficient, such as `lambda = m0/(m0+B_main/K)`. It matches pooling when allocation is uniform, preserves value unbiasedness when main counts are adaptive, and removes the fixed-lambda pilot floor as B grows. It is not generally adaptive-allocation variance-optimal; choose any alternative weights from historical data before the current pilots and evaluate separately.

### 5. Medium-high: advantage MSE does not order gradient MSE

Section 22 (pages 16–17) carefully describes an upper bound, but sections 57/60 and the narrative should consistently treat transfer as an empirical hypothesis. If columns of S are the fixed thought score vectors,

\[
 g=SA/K,\qquad
 E\|\widehat g-g\|^2\approx
 \frac1{K^2}\sum_j\sigma_j^2\|SJ_{:j}\|^2
 \left[\lambda^2/m_0+(1-\lambda)^2E(1/M_j)\right].
\]

The gradient-sensitive coefficients can differ from `sigma_j² ||J_:j||²`. Under `||s_i|| <= G`, the bound `||g_hat-g||² <= G² ||A_hat-A||²/K` is valid, but smaller error in the right-hand side does not imply smaller realized left-hand error. Compare a frozen-model, gradient-sensitive diagnostic allocator against JADA. Label `g*` as a conditional fixed-thought surrogate gradient, not the exact gradient of expected task reward. The actual trainer averages each segment's token loss and clips PPO ratios, so sequence-score gradients in section 35 do not by themselves audit the implemented training update. Add a token-reduction-matched check, then distinguish the clipped/KL update from the simple surrogate.

Do not turn section 23's optional convergence template into a theorem without controlling nonlinear bias, clipping, self-normalization, and the objective being differentiated.

### 6. Medium: trainer integration changes important conventions

The current trainer uses globally gathered thought values, sample standard deviation (`std()`), and a minimum clamp. JADA proposes population variance with epsilon under the square root. These choices differ at small K and near degeneracy. Hold them fixed across the controlled methods or report a separate normalization ablation.

README semantics are K thoughts **per rank**; four ranks with K=4 create 16 global thoughts for a prompt. Explicitly define whether JADA's budget and Jacobian use global K or local K. Computing local Jacobians while training globally normalized advantages targets a different function.

Fixed-M assumptions occur in reward reshaping (line 797), repeated multimodal input expansion, thought index strides (line 1030), and sign-inconsistency reshaping (lines 1222–1224); answer/KL reductions also assume a rectangular layout. Ragged counts need explicit thought ownership and consistent global budget/statistics. The MVP should demonstrate uniform-count equivalence first. These are future integration requirements, not bugs in a JADA implementation that does not yet exist.

### 7. Medium: finite reference pools and feasibility need explicit protocols

Sections 33–35 use noisy reference means as if true. At N=128, a Bernoulli mean near 0.5 has standard error approximately 0.044, before nonlinear normalization. Overlap between reference, pilot and main observations can create optimistic error estimates; sampling main rewards without replacement from a depleted common pool also breaks the conditional independence used in the value proof relative to the original empirical-pool mean.

Use independent reference and replay pools and report reference-size sensitivity. Alternatively, explicitly define a finite empirical-distribution experiment with independent draws with replacement, and limit conclusions to that distribution. Depletion invalidates the displayed conditional mean-zero proof; it does not automatically imply unconditional bias. Under uniformly sampled fixed-size pilots and uniformly drawn positive main counts from the remaining finite population, the fixed-weight estimator can still be unconditionally unbiased for that population mean, but its covariance and finite-population correction need a different derivation. Bootstrap across prompts/thought groups for LLM evidence; repeated replay trials from a single prompt are not independent model examples.

Define total versus main budget throughout: `B_total=K*m0+B_main`, with `B_main >= K` for the estimator. For example, K=4 and m0=4 cannot fit B_total=16 while retaining a main sample per thought. Filter infeasible cells of the proposed pilot/budget grid. Qwen3 is a different generation/architecture within the Qwen family; describe that test precisely rather than as evidence across unrelated model families.

## Minimum staged experiment package

1. **CPU estimator gate.** Finish the exact reuse covariance and local surrogate derivations. Verify analytic Jacobian and allocator invariants. Sweep K in {2,4,8,16}, total budgets {16,32,64}, and m0 in {1,2,4} only where feasible. Include the six regimes above and a predeclared heterogeneous distribution over probability vectors. Cross allocation {uniform, variance-only, JADA, oracle} with reuse {main-only, naive, fixed 0.5, budget-precommitted}. For larger cases use paired Monte Carlo with uncertainty intervals; never choose the benchmark distribution after inspecting winners. Show improvement as a function of pilot fraction, group dispersion and leverage. No GPU cost.
2. **Frozen-model gate, only after useful CPU signal.** Start with 100–200 prompts, **global K=4**, and independent reference/replay pools of 128 continuations each. That is 102,400–204,800 generated continuations plus 400–800 thought prefixes. If online training instead uses four ranks with four thoughts per rank, its global K=16 requires a matching frozen study: 409,600–819,200 continuations and 1,600–3,200 thoughts under the same pool sizes. Choose this scope before generating pools. Diagnose format compliance and reward sparsity first. Report advantage bias/MSE, sign error, allocation mismatch to oracle, and reference-size sensitivity. Reuse identical stored pools for all estimators. Include weighted answer normalization without changing it between adaptive methods.
3. **Gradient gate on the same frozen material.** Measure LoRA gradients for 400–800 fixed thoughts at global K4 (1,600–3,200 at global K16), with token reduction matching the trainer. A K-by-K score Gram matrix can evaluate many replay gradient errors without repeated full backward passes. Compare JADA with variance-only and a diagnostic gradient-sensitive allocation; check the simple frozen surrogate separately from clipped/KL updates. Report prompt-level intervals and clarify the parameter subset.
4. **Online pilot, only after both estimator and gradient gates.** One available or implemented binary-reward task, one 3B model, four methods and three seeds: twelve training runs. Compare native uniform, matched two-stage uniform, variance-only, and JADA. Use the same initial checkpoint, reward, regularization, optimizer and normalization; hold pilot answer-loss usage identical across the three two-stage methods. Define global K, count all pilot and main generation, and track tokens, wall time, GPU-hours, backward passes and target-performance crossings. The two-stage uniform control is necessary to separate allocation from the split estimator and main-only answer training. A nine-run screen is a reduced comparison and cannot resolve that attribution by itself.
5. **Submission expansion.** Add a second task and larger model only after the pilot shows a worthwhile effect. Compare relevant gradient-aware allocation and document the narrower setting if a full EPIG implementation is not directly comparable. Predefine evaluation frequency, target selection and confidence intervals; report failed target crossings as censored rather than deleting them.

Compute estimates are work counts, not measured duration. For a generation throughput q tokens/GPU-second and measured mean completion length L, approximate frozen-pool GPU-hours as `continuations * L / (3600*q)`, plus prefix/prefill and scheduling overhead. Measure q on the actual serving configuration. Online cost is the sum of twelve measured per-run GPU-hour costs, plus evaluation and debugging; method costs may differ. The proposal's 2–10-week calendar estimates cannot be verified without throughput and staffing information; no GPU was rented or job launched for this review.

## Results-to-claims matrix

| Possible outcome | Claim supported | Claim not supported |
|---|---|---|
| Reuse proof and exact bias checks hold | Precommitted fixed-weight values are unbiased under stated independent two-stage sampling | Unbiased normalized advantages or unbiased PPO/GRPO gradients |
| Oracle improves, practical pilots do not | The surrogate identifies potential allocation headroom under known statistics | Useful implementable JADA improvement |
| JADA beats variance-only in broad synthetic/frozen groups | Lower normalized-credit estimation error in the tested regimes and budgets | Universal optimality or online RL gains |
| Advantage MSE falls, gradient MSE does not | An estimator-level result; gradient transfer is limited | Gradient-efficient training |
| Conditional gradient MSE falls, online metrics do not | Better frozen conditional gradient calibration for the measured parameter subset | Better training or task performance per compute |
| Same-answer-count gains, no token/wall-time gains | Advantage per sampled continuation, or accuracy under the stated rollout count | Token/GPU-hour efficiency |
| Target is reached with fewer measured GPU-hours and matched evaluation | Empirical compute-to-target improvement with stated models, tasks and uncertainty | Universality, convergence guarantees, or cost reduction outside measured hardware |
| Variance-only helps but Jacobian has no robust added value | Adaptive allocation may be useful; narrow or pivot the contribution | Necessity or value of the normalization-Jacobian term |
| No consistent benefit after matched baselines | Boundary-condition or negative-result study, if analyzed carefully | The proposed positive JADA efficiency story |

## Prioritized actions

| Priority | Action | Estimated compute | Exit condition |
|---|---|---|---|
| P0 | State precise normalization, global K, budgets and sampling assumptions | None | All theory and compared implementations target the same map |
| P0 | Add exact reuse covariance, pilot floor and precommitted budget-weight analysis | CPU only | Proof matches raw pilot/main estimator |
| P0 | Narrow novelty relative to GRPO-MA Eq. 2 and EPIG Eq. 9 | None | Claims credit existing foundations and name the specific new procedure |
| P0 | Run broader finite-pilot estimator checks | CPU only | Practical benefit is robust and meaningful versus variance-only |
| P1 | Define answer moments, pilot usage, KL reduction and ragged ownership | Small correctness checks; no training required | Uniform case reproduces its selected control |
| P1 | Independent frozen pools and reference sensitivity | 102k–205k continuations at global K4; four times this at global K16 | Benefit persists on real model-generated fixed thoughts |
| P1 | LoRA/token-matched conditional gradient study | 400–800 thought backward passes at global K4; four times this at global K16 | Lower gradient MSE versus variance-only |
| P2 | Twelve-run online pilot and cost accounting | 12 measured training-run costs | Useful token/time gains with uncertainty |
| P3 | Second task/scale and targeted ablations | Conditional on pilot; not estimated in hours | Scope claims match measured breadth |

## Proposed paper structure if the gates pass

1. Problem: finite-budget credit estimation inside a fixed thought group; why raw reward variance alone may miss normalization leverage.
2. Foundations and scope: inherited GRPO-MA normalization analysis and established sensitivity-weighted allocation; distinguish gradient-aware branching.
3. Two-stage estimator: policy snapshot, raw pilot/main values, exact unbiasedness/covariance, precommitted reuse weights and limits.
4. Allocation: local trace-error target, analytic leverage factor, integer budget, pilot estimation and finite-budget failure regimes.
5. Evidence: oracle/practical synthetic gaps; independent frozen-model estimator and conditional-gradient evaluation; real online compute-to-target comparisons.
6. Limitations: near-degenerate normalization, noisy small pilots, advantage/gradient mismatch, ragged generation overhead and scope of tested tasks.

Figure plan: estimator MSE versus total budget with all pilot costs; finite-pilot/oracle gap by dispersion; advantage versus gradient-error comparison; online performance versus tokens and GPU-hours; allocation examples with failures as well as successes.

## Review dialogue and convergence

Round 1: fresh adversarial review of the whole proposal, trainer conventions and primary related work. The executor accepted the answer-normalization, finite-pilot, novelty and compute-fairness concerns. The executor's suspected estimator/allocation mismatch was narrowed: common fixed lambda preserves the allocation-dependent first-order objective, and shared allocations do not force off-diagonal covariance under independent two-stage sampling. Exact finite diagnostics then provided evidence that the current default fails the stated gate.

Round 2: the reviewer checked this synthesis and found no material algebra error. Three clarifications were accepted and incorporated: the practical online design includes both native and matched two-stage uniform (twelve runs total); frozen/gradient work counts explicitly use global K4 and scale fourfold for global K16; and finite-pool depletion invalidates the conditional proof without automatically implying unconditional value bias. The reviewer also endorsed precommitted budget-dependent reuse as a comparator, the staged gates, and the bounded claims matrix.

**Final consensus:** proceed with inexpensive estimator diagnostics and precise objective specification. The current default does not justify expensive online expansion. The mathematical foundations are coherent, while novelty and useful finite-budget effects remain open. The claim requirements, gated experiment package, and narrative have converged; actual proposal rewrites, trainer fixes, and positive experiments have not been performed or certified.

The reviewer's proposal-stage submission-readiness assessment is **2/10**, with confidence **4/5** in the mathematical/integration review. This is an explicit readiness scale, not an observed acceptance probability or a rejection of the broader idea. Same-family `acceptance_status: provisional` remains in force.

Raw transcripts: [round 1](.aris/traces/research-review/2026-10-11_run01/001-round-1-review.response.md), [round 2](.aris/traces/research-review/2026-10-11_run01/002-round-2-convergence.response.md), and [targeted literature consultation](.aris/traces/research-review/2026-10-11_run01/support-novelty-check.response.md). The literature-support agent was `/root/jada_reviewer/novelty_check`, inheriting `gpt-6-astra` / ultra; it also remains same-family and provisional. Project memory was updated in [RESEARCH_NOTES.md](RESEARCH_NOTES.md). No source implementation or proposal revision is implied by this report; the revisions above are recommendations.
