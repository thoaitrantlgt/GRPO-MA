# Synthetic fixed-score gradient loss

Let delta=f(V)-f(p), with the unchanged population-variance/epsilon normalization and fixed budget-precommitted raw reuse. If score matrix S is fixed before reward sampling, the update error is S delta. Its squared norm is delta^T G delta for G=S^T S positive semidefinite. Exact loss also equals tr(G Cov(delta))+E(delta)^T G E(delta). Random reward-correlated S violates the fixed-S conditioning used here; no measured model score geometry is claimed.

All declared Grams have trace1. G=I/4 equals the prior scalar advantage-MSE exactly. Centered advantages have sum zero, so with P=I-11^T/4 only PGP is observable. Aligned-null G=11^T/4 therefore gives exactly zero loss; relative improvements are undefined. The equicorrelated control G=.5 I/4+.5 11^T/4 is exactly half the scalar loss and supplies no independent objective geometry. The four nontrivial primary geometries are a fixed contrast rank-one matrix, a heterogeneous diagonal matrix, a paired correlation matrix and a dense PSD matrix. They were defined independently of p and method outcomes.

For each pilot history h and candidate positive main allocation m, exact weighted conditional risk is

    r_G(h,m;p)=E_t|p,m [f(V)^T G f(V)] + f(p)^T G f(p)
               -2 E_t|p,m[f(V)]^T G f(p).

The engine enumerates all main count states, pilot histories and allocations. It projects G through P for stability and explicitly zeros the declared null geometry. Under trace-one G and K4, loss is bounded above by16 times the largest eigenvalue of G. The adaptive weighted oracle minimizes this conditional risk at each p,h; fixed weighted oracle minimizes pilot-averaged risk. Scalar-oracle counts minimize the isotropic risk and are then evaluated under every G. Comparing weighted to scalar oracle isolates objective geometry from access to true p.

The local surrogate is sum_j c_j [lambda^2/m0+(1-lambda)^2 E(1/M_j)], where c_j=p_j(1-p_j)[J(p)^T G J(p)]_jj. The true-p surrogate uses known p. Gradient-JADA substitutes Jeffreys posterior mean for p inside J and posterior E[p_j(1-p_j)] for reward variance, using only pilots and the supplied synthetic G. It is a diagnostic toy policy, not a method that estimates score geometry before real model generation. Scalar-equivalent controls reuse original allocation tie handling. Positive counts and fixed-lambda raw unbiasedness/covariance proofs remain unchanged.

The primary averages risks over four nontrivial fixed geometries and all three settings inside each probability-vector cluster, then equally weights the six population strata. Relative reduction is a ratio of these mean risks, not an average of ratios from near-zero groups. Null ratios are not reported. Fixed synthetic matrices and settings are repeated measurements; they do not enlarge the independent sample count beyond96 fresh vectors.

HEADROOM_FOR_GRADIENT_CPU requires both lower95 relative limits above5%: adaptive weighted oracle versus uniform, and adaptive weighted oracle versus adaptive scalar oracle. It does not declare the pilot policy successful, reverse old scalar gates or authorize model training. A later pilot-only weighted posterior method would need a separately declared protocol and new confirmation. G estimation and reward-score dependence require further evidence.
