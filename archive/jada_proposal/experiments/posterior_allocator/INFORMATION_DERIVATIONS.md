# Pilot information gate: fixed reuse and cost controls

Let m0 pilot answers per thought and positive integer M_j main counts sum to main budget N. Total cost B=4m0+N. Reuse lambda is selected before the current pilot observations; it may be independent of 4m0/B in the fixed-main diagnostic arm.

Raw values are V_j=lambda H_j/m0+(1-lambda)T_j/M_j. Condition on pilots and auxiliary tie randomness. Independent main samples have E[T_j/M_j | H,M,p]=p_j. Therefore E[V_j|p]=p_j and

    Cov(V|p) = diag(p_j(1-p_j) [lambda^2/m0 + (1-lambda)^2 E(1/M_j|p)]).

Off-diagonal covariance is zero: independent pilot coordinates have zero covariance, centered main means have zero conditional cross-covariance, and pilot/main cross terms vanish by the conditional mean identity. Pilot-dependent allocation and independent tie randomization preserve this argument. Normalization f(V) remains nonlinear, so unbiased raw values do not imply unbiased advantages.

The exact conditional risk engine accepts a separately precommitted lambda. When lambda is omitted it retains the prior budget-precommitted behavior. Counts still use total-4m0 as main budget. Targets remain f(true p). Exact main-state risk is computed using E[f(V)^2]+f(p)^2-2 E[f(V)]f(p), averaged across thoughts.

The posterior objective remains E_p|H[r(H,M;p)]. Offline scrambled-Sobol quadrature integrates joint pilot likelihood times exact main risk; numerator and mass are pooled then jointly permutation-symmetrized. True p appears only at offline prior points or in the evaluator and information-control oracle; runtime allocate accepts only pilot counts and independent tie RNG. Frozen policies may be approximate posterior minimizers, never certified continuous Bayes optima.

At fixed N=8 and lambda=.5, m0={1,2,4} isolates changing pilot size within that estimator family while changing total costs {12,16,24}. A gain here is an information diagnostic rather than an equal-cost gain. At fixed B=24, m0={1,2,4} also changes N={20,16,8} and lambda={1/6,1/3,2/3}. This arm supplies matched-total-cost comparisons. Uniform uses six total answers per thought with equal pooling in the latter arm; its exact baseline therefore agrees across pilot splits.

Population comparisons equally weight six strata of 16 independent vectors. Each setting has paired method risks per vector, exact over reward states and allocation randomization. Student-t delta intervals describe probability-vector sampling uncertainty conditional on the frozen policies. Quadrature variability is diagnosed separately with independent scrambles and half-policy disagreement; confirmation does not trigger more quadrature or a primary-method switch.

For the fixed-main trend, let g_m(p)=R_uniform,m(p)-R_Jeffreys,m(p). Report the equal-stratum mean of g_4(p)-g_1(p) and its paired stratified t interval. The implementation passes offset risks 4+g_4 and 4+g_1 to the existing paired difference routine; the offset cancels exactly, and only the absolute difference/interval is retained. No relative effect based on that arbitrary offset is reported.

All settings keep complete positive allocation spaces. Full-information adaptive oracle risk sums min_M r(H,M;p) over pilot probabilities, so it is a lower comparator for every frozen pilot-only policy. Its benefit does not establish pilot-only learnability or training gains. Stress and fixed-bin subgroup analyses are descriptive; all six benchmark components are retained.
