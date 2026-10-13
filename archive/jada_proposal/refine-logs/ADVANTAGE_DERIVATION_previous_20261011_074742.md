# Derivation Package: advantage target and update

## Target

Define what thought advantage estimates, which frozen-policy update it serves,
and what evidence a future normalization method must supply. This contract
reframes the scientific endpoint as update estimation, without selecting a
new normalization or reopening any completed gate.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION. The conditional thought-value
target and local update are well defined. Linking them to expected-reward
policy gradients, full clipped updates, or training gains needs additional
assumptions and evidence. No new estimator is proposed here.

## Invariant Object

For a frozen prompt, K fixed thought prefixes and frozen continuation policy,
let q_j=E[R(answer)|prompt,thought_j]. R is the declared reward sum, not
necessarily binary correctness. The primary research endpoint is the error
in the thought PPO ascent update produced by replacing true conditional
values q with estimates V, with the same sampled thoughts, masks, parameter
state and reduction as the baseline. Conditional value uncertainty and
thought sampling uncertainty are different layers.

## Assumptions

- All K thoughts belong to one declared prompt/normalization group. Cross-rank
  ordering and grouping must be established before a trainer integration.
- Continuation draws estimate q under a fixed policy and fixed reward rule.
  Adaptive positive counts and fixed reuse inherit the earlier covariance
  assumptions; correctness probabilities alone model only binary rewards.
- The thought score is the masked token-mean score, each thought weighted
  equally. The answer branch has a separate token/sequence reduction.
- Rewards, estimated values and normalizers are treated as stopped-gradient
  coefficients in the policy loss. The trainer constructs reward tensors
  from generated outputs; this analysis does not differentiate through them.
- A fixed-score conditional analysis conditions on thoughts, masks and policy
  state. It does not assume answer scores independent of answer rewards.
- K>=2 for sample standard deviation; degenerate/singleton groups require
  explicitly defined handling, not an undefined standard deviation.

## Notation

P=I-11^T/K, q=(q_1,...,q_K), z=Pq. Define the implementation-aligned
conditional-value target

```
h_tau(q) = Pq / max(sqrt(||Pq||^2/(K-1)), tau), tau=1e-4.
```

This is an ideal target obtained by substituting conditional values into the
trainer's thought-normalization rule; it is not E[h_tau(V)].
A*=h_tau(q), Ahat is a future estimator of this same A*, delta=Ahat-A*.
T_j is the set of active thought tokens, L_j=|T_j|. Empty token sets have
zero score contribution, matching the mask reduction.
At the old-policy point define

```
s_j = (1/K) * (1/L_j) * sum_{t in T_j} grad_theta log pi_theta(token_jt),
S = [s_1,...,s_K], g*=S A*, ghat=S Ahat, G=S^T S.
```

For nonempty T_j the expression is literal; set s_j=0 for an empty mask.
g is the reward-surrogate ascent gradient; optimizer preconditioning,
momentum, clipping of gradient norms and parameter steps are separate.

## Derivation Strategy

Map the actual normalization and loss reduction to a conditional target,
differentiate the surrogate at a frozen policy, then define exact estimation
loss. Treat multi-step clipping, the answer branch and reward-gradient claims
as explicit extensions, not implicit consequences.

## Derivation Map

1. Trainer reward aggregation defines q; thought means estimate q.
2. Trainer normalization defines h_tau and A*, distinct from earlier f_epsilon.
3. The token-masked thought PPO loss defines the ascent update g.
4. At ratio1, fixed scores yield the exact Gram-weighted error identity.
5. Away from ratio1, the clipped loss must be differentiated directly.
6. Full-update or reward-gradient claims require separate joint-distribution
   and objective arguments, with scale-matched comparisons.

## Main Derivation

### Step1: target and code mapping (identity / definition)

The trainer sums reward functions, computes each thought's continuation mean,
gathers thought means, and normalizes by `.std().clamp(min=1e-4)` in
[reward processing](trainer/grpo_ma_trainer.py:795). Answer advantage
instead uses all gathered individual rewards. These are different objects.
The local CPU check confirms default `.std()` equals correction1 in the
installed PyTorch2.12.1+cpu. The repository README pins a different training
version, so this check is not an execution certificate for that environment.

Earlier gates used f_epsilon(q)=Pq/sqrt(||Pq||²/K+1e-6). Their target has
population variance and additive smoothing. When the floor is inactive and
smoothing negligible, h_tau(q) is approximately sqrt((K-1)/K)*f_epsilon(q).
Near equality or the floor this relation is insufficient. Existing results
remain valid for their frozen target, but cannot be relabeled trainer-exact.

### Step2: local reward-surrogate update (identity under stated conditions)

The trainer uses one thought sequence per prefix, token-mean reduction per
thought, then mean over thoughts; it separately averages all answer sequences
in [compute_loss](trainer/grpo_ma_trainer.py:1030). Define

```
U_think(theta,a) = (1/K) sum_j (1/L_j) sum_t
                  min(r_jt(theta)*a_j, clip(r_jt(theta),lo,hi)*a_j).
```

The loss is minus U_think plus the separate answer loss and KL penalty.
At the old-policy point r=1 inside clip bounds, grad U_think(theta,a)=S a.
The sign is ascent; the trainer's loss gradient is its negative.
Holding the answer/KL branches fixed isolates thought-credit estimation.

### Step3: primary conditional loss (identity)

With S fixed conditional on the sampled thoughts and policy state,

```
L_update = E[||ghat-g*||² | thoughts, masks, policy]
         = E[delta^T G delta]
         = tr(G Cov(delta)) + E[delta]^T G E[delta].
```

Scalar advantage MSE corresponds to G=I/K and remains a diagnostic. The
previous synthetic-Gram gate already tested objective headroom and failed its
materiality criterion. This contract does not override that result or
reselect geometries. If scores are random and reward-correlated, retain the
joint expectation E[||S delta||²]; replacing it by average G times an
unconditional covariance is generally invalid.

### Step4: clipping and full-update boundary (identity / limitation)

Away from ratio1, positive advantage clips the upper-ratio branch and negative
advantage clips the lower-ratio branch. The coefficient multiplying each
score depends on ratio and the sign of a_j. If estimated and target advantages
have different signs, the active branches can differ. Evaluate
grad U_think(theta,Ahat)-grad U_think(theta,A*) directly; a single fixed G
does not represent all such changes. Declare handling at exact clip boundaries.

Scaling thought advantages by a positive constant scales its reward-surrogate
gradient, including away from clipping kinks. It does not scale the separate
answer branch or the unchanged KL penalty. Thus it changes their relative
weights. Even scaling both reward branches changes their strength relative to
fixed KL. Neither operation is universally equivalent to scaling the learning
rate of the whole optimizer; adaptive optimizers add further differences.

### Step5: relation to expected reward (definition / non-equivalence)

For independently sampled thoughts, the unregularized expected-reward
thought-score gradient involves E[score(thought)*q(thought)]. A baseline
independent of that thought can be subtracted because E[score]=0 under the
usual differentiability/integrability conditions. A finite-group baseline
containing that same thought is dependent: for iid K samples and an
unnormalized group-mean baseline, the expected group update is (1-1/K) times
the ordinary score gradient. Random group standard deviation introduces
additional dependence. Token-mean rather than sequence-sum scores also change
length weighting. Consequently h_tau(q) is an implementation-aligned group
credit target, not an automatically unbiased reward policy-gradient estimator.

If the research goal changes to estimating the true reward gradient, it needs
a separate contract for thought distribution, baseline dependence, token
weighting, reward-score dependence and KL. We retain the implementation-aligned
conditional thought update as the immediate estimation target.

## Remarks and Interpretation

- The next normalization must estimate A* under this contract or explicitly
  declare a changed objective. Changing target is a new research hypothesis.
- Improving direction, scale and branch balance are separate claims. A
  constant advantage multiplier alone provides no new credit information.
- The previous fixed-half probe remains rejected. Its positive scalar-MSE
  mean cannot establish update or reward improvement.

## Acceptance and Rejection Criteria for a Future Design

No method or new confirmation is started by this contract. Before one is run:

1. Freeze a concrete estimator, its runtime information, all sample costs,
   normalizer convention, prompt grouping, policy state and clipping regime.
   It may not use true q or main outcomes unavailable at decision time.
2. Compare equal-budget uniform plug-in and the unchanged target A*. State
   whether the answer branch and KL remain fixed, and report branch gradients
   separately plus full-gradient error if a full-update claim is made.
3. Report absolute update MSE, squared bias and variance, target alignment,
   magnitude and direction error. At g*=0, relative risk/direction are undefined;
   report absolute spurious-update energy, never silently drop null cases.
4. Include a scalar-calibration control chosen on development only and frozen
   before confirmation; it receives the same available information. Require
   any claimed credit improvement to beat that scale control as well as
   uniform. No oracle scalar can serve as a deployable baseline.
5. Keep the original mixture and all six regimes. Predeclare a new untouched
   confirmation only after a materially distinct design survives development.
   The proposed CPU pass rule is lower95 relative update-MSE reduction >5%
   against both uniform and the scale control, plus no >10% exact regression
   in any named stress/budget cell. This is a future contract criterion,
   not a retrospectively applied change to any completed gate.
6. If exact enumeration is unavailable, pair Monte Carlo across estimators
   and cluster intervals by independent probability vector/group. Joint score
   models must be fixed before outcomes and identified as synthetic unless
   measured. No new geometry winner selection.
7. Verify equality, deterministic nonconstant reward, clipping/sign-change,
   std-floor, empty-mask and singleton-group boundaries. Verify runtime
   information and grouping before any trainer change.

## Boundaries and Non-Claims

This is a source-level contract and local algebra audit. No trainer execution,
model download, GPU, new benchmark, candidate tuning, fresh confirmation or
training improvement occurred. The CPU contract is a research slice of the
trainer, not complete behavioral equivalence. Trainer code is unchanged.

## Open Risks

- Real frozen-model score geometry remains unmeasured; earlier synthetic
  geometry does not certify it. No such model run is justified here.
- Actual reward functions need not be Bernoulli; their declared sum and joint
  score distribution need to replace the toy model for broader claims.
- Global grouping, uneven rank counts, all-invalid and singleton handling
  need dedicated integration evidence. The inspected code normalizes all
  gathered thought values when any local thought is valid; it is not a
  valid-only normalization rule. No validity bug fix is attempted here.
- Local PyTorch differs from the pinned training environment. The bridge
  freezes its desired convention explicitly, pending environment validation.
