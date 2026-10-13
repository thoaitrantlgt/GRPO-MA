# Proposed objective revision: credit error versus update error

Status: derivation and next-hypothesis memo, not an experiment or training GO. Current allocator gates remain negative. The convergence diagnostic must be assessed before committing another method.

## The mismatch to test

For a fixed normalization group of K thoughts, let the scalar advantage error vector be delta=estimated A minus target A(p). Let s_j be the thought's score-gradient vector after the intended token reduction, answer weighting and group scaling. Stack these vectors as the columns of S. If S is fixed with respect to the reward sampling being evaluated, update error is S delta, and exactly

    E ||S delta||^2 = tr(G Cov(delta)) + E(delta)^T G E(delta),
    G = S^T S, with G positive semidefinite.

The CPU gates minimized E||delta||^2/K. This corresponds to G=I/K, so it ignores how credit errors align with correlated score-gradient directions. A measured reduction in scalar advantage MSE does not itself imply a reduction in gradient MSE. Conversely, the weak scalar allocation results do not prove a gradient-aware objective has useful headroom.

Under a local delta-method approximation delta approximately J(V-p), and the fixed-reuse raw-value covariance theorem, the allocation-dependent part becomes

    (1-lambda)^2 sum_j c_j/M_j,
    c_j = p_j(1-p_j) [J^T G J]_jj.

All c_j are nonnegative because G is positive semidefinite. Existing positive-integer greedy allocation can optimize this separable surrogate. The pilot covariance term lambda^2/m0 is allocation-independent. This derivation inherits the local approximation's failure modes; the previous true-p local-surrogate oracle was worse under the scalar metric. Merely replacing column norms by a Gram weighting is not evidence that the problem is solved.

## What a CPU-only falsification should establish first

1. Declare a score-Gram model independently of method outcomes. Include G=I/K as the exact scalar-regression control, rank-one aligned gradients and correlated positive-semidefinite matrices. Treat synthetic G as a model assumption, not measured LLM gradients.
2. Compare exact nonlinear weighted risk E[delta^T G delta] to the local surrogate and its true-p oracle. This separates objective headroom from posterior estimation noise before writing a deployable method.
3. Cross the same uniform/variance-only/JADA controls with a precisely specified gradient-aware policy; preserve the original six probability-vector components and stress regimes. Any new method requires a newly committed confirmation protocol rather than reusing consumed vectors as fresh evidence.
4. Define whether G can actually be estimated before allocation. G must not be built from unavailable main outcomes or the current group's true p. A true-G method is an information control. A pilot-estimated G needs its own error and sample-cost analysis.
5. Account for reward-score dependence. If S is random and correlated with sampled rewards, the fixed-S identity cannot be evaluated by multiplying unconditional advantage covariance by an average G. Preserve their joint distribution and use matched token reduction. Only after CPU evidence could frozen-model pools supply real score geometry; no model run is authorized by this memo.

## Scope boundary

The memo proposes a distinct objective test, not another parameter sweep of a failed scalar estimator. No GPU, model call, synthetic outcome experiment or gradient-aware implementation was run here. Do not claim novelty, gradient benefit, practical deployability or online gains from this derivation alone.
