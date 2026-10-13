# CPU fixed-score gradient objective headroom gate

Status: completed CPU evaluation; NO_OBJECTIVE_SPECIFIC_SIGNAL. User authorized CPU work. No model or GPU calls.

Hypothesis: exact allocation under a predeclared non-isotropic synthetic score Gram reduces gradient-MSE beyond the scalar advantage-optimal oracle. Oracle-versus-uniform headroom alone is insufficient because the earlier scalar oracle already had headroom.

## Frozen design

K4; budget-precommitted reuse; B16/m0=1,2 and B24/m0=4. The primary averages squared-error risks across four fixed nontrivial Grams and all three settings inside each probability-vector cluster. Seven trace-one PSD matrices were saved before outcomes: isotropic scalar anchor, aligned-null boundary, equicorrelated scalar-half anchor, fixed contrast rank-one, heterogeneous diagonal, paired correlated and dense correlated matrices. None is chosen from winners, aligned with current p or fitted to the benchmark.

Nine allocations are retained: uniform, variance-only, original JADA, known-G pilot gradient-JADA, true-p gradient local-surrogate oracle, fixed/adaptive true-p exact gradient oracles, and fixed/adaptive true-p scalar oracles evaluated on the gradient loss. Runtime gradient-JADA uses only pilot counts and the supplied fixed G. This is a known-geometry toy policy, not an estimated real score geometry.

Fresh96 probability vectors seed1732052026,16 per unchanged original mixture component, plus96 consumed development vectors and six named stress regimes. All are saved before full evaluation. Exact reward-state/count enumeration at every setting; no reward Monte Carlo or posterior fitting in this headroom gate. Independent units for fresh inference are96 probability vectors, not Grams or repeated settings.

HEADROOM_FOR_GRADIENT_CPU requires both lower95 relative limits above5%: adaptive gradient oracle versus uniform and adaptive gradient oracle versus adaptive scalar oracle. Report every method and Gram, including failures, and preserve all scalar negatives. Aligned-null has zero loss and no defined relative ratio. Fixed bins for dispersion/leverage are descriptive; pilot fraction is confounded with budget in this scope.

## Verification and interpretation

All23 tests passed, including scalar equivalence, null loss, equicorrelation scaling, score-covariance identity, weighted conditional risk versus direct enumeration, coefficient identity and pilot-only allocation independence from current p. End-to-end sanity passed all three settings before deployment. Fresh review is requested by experiment-bridge; if unavailable, use its stated local-only fallback and record the failure rather than claiming independent clearance.

Saved-table audit verifies complete count spaces, exact weighted oracle minima, scalar-oracle minima, pilot-only known-G allocation, direct weighted-risk cases, original scalar engine agreement, predeclared fresh vector generation and paired intervals. All artifacts are retained and fixed summaries versioned.

A positive oracle outcome supports only further CPU method development under supplied synthetic G. It does not establish a deployable estimator, an LLM gradient benefit or a training GO. A posterior weighted-risk policy, if justified, needs another declared protocol and fresh confirmation. Reward-score dependence and real score estimation remain outside this gate; no GPU work follows automatically. Positive-result method ablations are not appropriate for oracle-only headroom.
