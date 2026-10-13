# Exact nonlinear oracle and its information boundary

Fix K=4, Bernoulli probabilities p, pilot size m0, total budget B=16 and main budget b=B-K m0. All main count vectors lie in S={m in positive integers^K: sum(m)=b}. Raw reused values have precommitted lambda=K m0/B and unchanged normalization epsilon=1e-6.

For observed pilot count vector h, let t be a fresh main success-count vector. Define

    value(h,t,m) = lambda h/m0 + (1-lambda) t/m,
    f(x) = (x-mean(x))/sqrt(population_variance(x)+epsilon),
    L(h,t,m;p) = mean((f(value(h,t,m))-f(p))^2).

Coordinate division is componentwise. The exact conditional nonlinear risk is

    r(h,m;p) = sum_t [product_j BinomialPMF(t_j;m_j,p_j)] L(h,t,m;p).

The pilot probability is w(h;p)=product_j BinomialPMF(h_j;m0,p_j). Fresh main sampling remains independent conditionally on h, even when m depends on h and true p.

## Fixed and adaptive optima

The fixed-count true-p oracle chooses

    m_fixed(p) = argmin_{m in S} sum_h w(h;p) r(h,m;p).

It knows p but chooses one allocation before pilots. Its risk cannot exceed that of any fixed-count policy, including uniform and the true-p local-surrogate allocator. It need not beat pilot-adaptive variance-only or JADA.

The true-p adaptive oracle chooses

    m_adaptive(h,p) = argmin_{m in S} r(h,m;p),
    R_adaptive(p) = sum_h w(h;p) min_{m in S} r(h,m;p).

For every pilot-only policy a(h) in S,

    R_adaptive(p) <= sum_h w(h;p) r(h,a(h);p).

This holds pilot state by pilot state. Randomized allocations are convex combinations of conditional risks and cannot beat their minimum either. Consequently the adaptive oracle gives a global lower bound on nonlinear MSE for this count/reuse sampling design, while the old local-surrogate oracle does not. This is an exact finite-budget allocation statement, not a gradient/training theorem.

The information advantage is explicit: knowing true p reveals target advantages and realized pilot errors. That information is not available to a deployable pilot-only allocator. Conditional raw-value expectation remains p+lambda(h/m0-p), independent of m; the oracle shapes the main-noise distribution and nonlinear normalization rather than directly changing this conditional raw mean. A large oracle gap demonstrates headroom but not that a pilot-only method can learn the oracle decisions.

## Exhaustiveness and exact evaluation

The number of positive K-way compositions of b is choose(b-1,K-1). At m0=1, b=12 gives 165 allocations and 16 pilot states. At m0=2, b=8 gives 35 allocations and 81 pilot states. Every main success-count state is included for each allocation. There is no noisy search objective, held-out reward selection error or Monte Carlo approximation; finite floating-point rounding remains.

The implementation accelerates finite summation using

    r = E[mean(f(value)^2)] + mean(f(p)^2)
        - 2 mean(E[f(value)] .* f(p)).

For each allocation, all candidate normalized estimates depend only on h,t,m and the committed reuse coefficient. Their exact main-count PMF weights vary with p. Matrix operations therefore evaluate the same finite sum for many p vectors at once. Tests compare this identity against direct squared-error enumeration and the previous exact default anchors. Full conditional risk tables are saved for independent verification. Numerically negative values within roundoff are clamped to zero; values outside [-1e-12,4+1e-12] raise an error. Equal computed risks take the lexicographically first composition.

## Raw-value unbiasedness is retained

With m>=1 fixed before main outcomes, E[t_j/m_j | h,p]=p_j. Therefore

    E[value_j | p] = lambda E[h_j/m0 | p] + (1-lambda)p_j = p_j.

The earlier exact diagonal covariance proof applies to both true-p oracles:

    Cov(value | p) = diag(p_j(1-p_j)[lambda^2/m0
                                   +(1-lambda)^2 E_h(1/m_j(h,p))]).

Only the allocation policy changes; reuse, total sampling cost and normalization are held fixed. Unbiased raw values still need not yield unbiased normalized advantages.

## Population inference and decision

Primary evidence uses 96 fresh probability vectors, 16 in each of six predeclared distribution strata. Both m0 settings are averaged inside each vector cluster, and strata are equally weighted. Paired baseline/oracle covariance across vectors gives delta-method relative-reduction intervals, with contrast-specific Welch-Satterthwaite degrees of freedom. These nominal 95% intervals concern variation over the declared vector distribution, not sampling uncertainty in exact count summation or LLM task generalization.

HEADROOM requires both adaptive-oracle-versus-uniform and adaptive-oracle-versus-variance-only lower limits to exceed 5%. This diagnostic permits a separate CPU method-development step, never an online-training GO. The original JADA NO-GO remains unchanged. Budgets 32/64, other K and other reuse are outside this gate. New deployable methods need another predeclared confirmation set after development.
