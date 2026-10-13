# Pilot-only posterior nonlinear allocation

Keep K=4, B_total=16, m0 in {1,2}, epsilon=1e-6 and lambda=K m0/B_total. The count space consists of every positive integer vector m summing to B_main=B_total-K m0. Raw final values and normalized advantages are unchanged from the exact nonlinear-oracle gate.

## Posterior objective

Let h be observed pilot successes, p the unknown true vector and pi(p) a committed population prior. The pilot likelihood is L_h(p)=product_j BinomialPMF(h_j;m0,p_j). For candidate m, exact conditional fresh-main risk is

    r(h,m;p)=E_t|p,m mean((f(lambda h/m0+(1-lambda)t/m)-f(p))^2).

The deployable Bayes objective is

    B(h,m) = integral pi(p)L_h(p)r(h,m;p)dp / integral pi(p)L_h(p)dp.

It integrates squared error to the true f(p); replacing this target by f(E[p|h]) changes the objective. The main outcome expectation is summed exactly. Continuous prior integration is approximated by predeclared randomized quasi-Monte Carlo points and is not an exact global Bayes-optimality certificate.

With a finite weighted prior support p_v, the joint numerator and mass are

    N_hm=sum_v w_v L_h(p_v)r(h,m;p_v), D_h=sum_v w_v L_h(p_v).

Choose counts minimizing N_hm/D_h. Averaging replicate posterior ratios is generally different from pooling joint numerators and masses; the implementation pools first, then divides. Under a perfectly integrated, correctly specified prior, a posterior-optimal policy cannot exceed a uniform policy's prior-average risk. This does not imply dominance at each p or under a different population distribution. Tests verify the finite-support Bayes inequality and objective against direct integration.

## Priors and information access

Primary: iid Beta(.5,.5). Secondary: the previously declared synthetic population prior, equally mixing five iid-coordinate Beta distributions and a logistic-normal group with shared location and scale. The secondary knows the population prior, not the current group's true p; its apparent benefit is conditional on that prior knowledge. It is reported as benchmark-informed sensitivity, never as distribution-free evidence or a replacement primary.

Offline prior points are generated from committed seeds; no old or confirmation vector is accepted as a fitting argument. The runtime table has one row per possible pilot count history. allocate(pilot_counts,rng) accepts only that history and independent auxiliary tie randomness. Counts are locked before main observations. True p enters the separate evaluator and diagnostic oracles only.

## Exchangeability and ties

Both priors are invariant to coordinate permutation. Therefore N_(g h,g m)=N_(h,m) and D_(g h)=D_h for any permutation g. Numerical integration noise may violate these identities. Group-average the joint numerator and mass over all 24 permutations before dividing; this is a valid symmetry-preserving quadrature improvement for the declared exchangeable priors.

For each h, give equal probability to every candidate within 1e-12 of the minimum approximate posterior risk. This includes tied permutations within identical-pilot blocks and preserves permutation equivariance without relying on thought index. Randomization is fully integrated in exact evaluation:

    R_policy(p)=sum_h L_h(p) sum_m q(m|h) r(h,m;p).

A policy can be used directly by sampling m from its saved row; the evaluator instead sums the row probabilities to avoid Monte Carlo reward/tie error. The true-p adaptive nonlinear oracle remains a lower bound on this policy risk.

## Reuse proof remains unchanged

Extend the pilot sigma-field to include auxiliary tie randomness. Conditional main means still have E[t_j/m_j|h,m,p]=p_j. Consequently E[value_j|p]=p_j, and

    Var(value_j|p)=p_j(1-p_j)[lambda^2/m0
                            +(1-lambda)^2 E_h,m(1/m_j)].

Independent pilot streams and conditional fresh main streams give zero off-diagonal value-error covariance. Shared randomized allocation does not alter this proof. Normalized advantages need not be unbiased.

## Numerical and population uncertainty

Eight independent scrambled Sobol replicates use 1,024 Jeffreys points each, and 256 points per mixture component per replicate. Component strata have equal weights. The same prior points are reused for both pilot sizes. Variability across scrambles and first-four/last-four policy disagreement are recorded; no sample increase or prior switch occurs after confirmation outcomes.

Fresh-vector evaluation integrates all count states and tie probabilities exactly. Its population interval averages both pilot settings inside each vector cluster, then equally weights six strata of 16 vectors. Paired covariance and Welch-Satterthwaite delta-method intervals are conditional on the frozen policy. They do not include construction-seed variation; that separate diagnostic must accompany the outcome. Neither interval supports LLM or training generalization.

The primary passes only if its lower relative-reduction limit exceeds 5% and both m0 point estimates are positive. The secondary is diagnostic. This scope supplies a pilot-only CPU signal test, not a GPU/online-training GO. Previously examined confirmation vectors are development now; new method changes consume the new holdout and require another declared confirmation.
