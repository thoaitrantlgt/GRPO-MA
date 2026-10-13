# Reuse covariance and local surrogate

Condition throughout on the prompt, sampled thoughts and unchanged policy. Reward streams are independent across thoughts, with means mu_j and variances sigma_j^2. Let F_P contain all pilot observations, each pilot size be m0, and main counts M_j >= 1 be F_P-measurable. Finalize counts before generating fresh independent main streams. No adaptive main stopping or finite-pool depletion is covered.

## Exact moments for precommitted reuse

Put eP_j = RbarP_j - mu_j, eM_j = RbarM_j - mu_j. Then

    E[eM | F_P] = 0,
    Cov(eM | F_P) = diag(sigma_j^2 / M_j),
    Cov(eP) = diag(sigma_j^2 / m0).

For a deterministic scalar lambda, committed before current pilot rewards,

    e = lambda eP + (1-lambda) eM,
    E[e] = 0,
    Cov(e) = diag(sigma_j^2 [lambda^2/m0 + (1-lambda)^2 E(1/M_j)]).

Proof: E[eP_i eM_j] = E[eP_i E(eM_j|F_P)] = 0 for all i,j. For i != j, the conditional main cross moment is zero and independent pilot cross moments are zero. On the diagonal use total expectation of the conditional main variance. Shared adaptive counts alone do not create cross-thought covariance. Unequal deterministic coefficients lambda_j admit the same diagonal formula coordinate by coordinate. Raw pilot means are required in the estimator; posterior means are only used for allocation.

Naive pooling is a different estimator. Write w_j(F_P)=m0/(m0+M_j). Then

    bias_j = E[w_j eP_j],
    E[e e^T] = E[(w .* eP)(w .* eP)^T]
                 + diag(sigma_j^2 E[(1-w_j)^2/M_j]),
    Cov(e) = E[e e^T] - bias bias^T.

Thus naive reuse may be biased and have off-diagonal covariance. It is evaluated, not treated as an unbiased substitute.

## Analytic normalization derivative

Use population variance, not the trainer's sample-standard-deviation convention. Define P=I-11^T/K, z=P mu, v=z^T z/K, s=sqrt(v+epsilon), f(mu)=z/s. Differentiating v gives dv=2 z^T dmu/K. Hence

    J = P/s - zz^T/(K s^3),
    q_j = ||J_:j||^2
        = (1-1/K)/s^2 - z_j^2 (v+2 epsilon)/(K s^6).

The identity follows by expanding the square, using Pz=z, P^2=P, and z^T z=K v. With v>0 define h_j=z_j^2/(K v); sum(h)=1. For negligible epsilon, q_j=[(K-1)/K-h_j]/v. For K=2 the q_j are equal; the stable exact expression is q_1=q_2=epsilon^2/[2(v+epsilon)^3]. When epsilon=0 and v>0 both are zero because the normalized two-coordinate vector is locally constant. At equal means v=0, J=P/sqrt(epsilon), q_j=(1-1/K)/epsilon, and the local surrogate can greatly overstate bounded nonlinear error. K2 JADA directly uses variance-only counts because the common positive q cannot change the ideal allocation; this avoids floating-point scaling changing tied marginal gains.

## Local quadratic risk and its limits

Taylor expansion gives f(mu+e)-f(mu)=J e+r(e). For unbiased fixed reuse,

    R_local = E||J e||^2
            = trace(J Cov(e) J^T)
            = lambda^2/m0 sum(c_j)
              + (1-lambda)^2 sum(c_j E(1/M_j)),
    c_j = sigma_j^2 q_j.

Divide by K for the per-coordinate MSE reported by this experiment. Conditionally on pilots,

    E[||J e||^2 | F_P]
      = lambda^2 ||J eP||^2 + (1-lambda)^2 sum(c_j/M_j).

The pilot term is independent of main allocation. Plug-in J and posterior variance are noisy estimates; this statement proves neither plug-in finite-budget optimality nor minimum exact nonlinear MSE. For naive reuse replace the pilot term by ||J(w .* eP)||^2 and the main term by sum(q_j sigma_j^2 (1-w_j)^2/M_j); w depends on counts, so the same c/M allocator has no naive-risk optimality claim.

If the Hessian norm is at most H along the segment from mu to mu+e, ||r(e)|| <= H||e||^2/2. Therefore

    |R_exact-R_local| <= H sqrt(E||J e||^2 E||e||^4)
                         + H^2 E||e||^4/4.

This bound uses Cauchy-Schwarz; it is useful only when errors are small compared to the normalization scale. Epsilon>0 ensures smoothness, but small epsilon and near-equal groups make global derivative bounds large. The experiment stores actual nonlinear MSE and local linear MSE separately, including equal and near-equal groups. Unbiased values do not imply unbiased normalized advantages.

## Allocator invariants

For fixed c_j>=0, equal unit cost and sum M_j=B_main, minimize sum c_j/M_j. Without active lower bounds, stationarity gives M_j=B_main sqrt(c_j)/sum sqrt(c). With M_j>=1, the continuous solution is M_j=max(1,sqrt(c_j/tau)), with tau chosen to meet the budget; all-zero coefficients allow any feasible allocation.

For integer counts start each M_j at one. Each extra answer decreases the objective by c_j/[M_j(M_j+1)], a nonincreasing sequence. Greedily take the largest remaining decrease. The prefix constraint on each sequence is automatically respected; choosing the largest B_main-K marginal gains proves global integer optimality for this separable objective. The code verifies positivity, integrality, conservation and brute-force optimality on small cases, including zero coefficients and ties. Ties use smallest index. Analytic and historical materialized-J norms can differ at machine-precision ties; the default regression records both, and reverse-coordinate sensitivity is predeclared.

Budget accounting is B_total=K m0+B_main with B_main>=K. Fixed lambda=.5 has an asymptotic pilot variance floor when m0 is fixed. Budget-precommitted lambda=K m0/B_total decays as the total budget grows, is fixed before pilots and equals naive pooling under uniform main counts. It need not be variance-optimal for adaptive allocation. A common variance-optimal coefficient for a known allocation distribution would be T/(S+T), where S=sum c_j/m0 and T=sum c_j E(1/M_j); this is not selected using current outcomes in this benchmark.

The oracle uses true c_j to optimize only the local surrogate. It is a diagnostic comparator, not a lower bound on nonlinear MSE. No training, gradient or LLM performance claim follows from these Bernoulli calculations.
