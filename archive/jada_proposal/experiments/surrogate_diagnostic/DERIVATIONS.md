# Exact moments and the limits of the local allocation surrogate

For K independent Bernoulli groups, take m0 pilot samples per group and
positive main counts M_j chosen from the entire pilot history H. Conditional
on H, fresh main counts are independent Binomial(M_j,p_j). For a constant,
precommitted lambda, define V_j=lambda H_j/m0+(1-lambda)Y_j/M_j.

The fresh residual N_j=Y_j/M_j-p_j has E[N_j|H]=0 and
E[N_j N_l|H]=0 for j != l. Therefore all pilot/main cross terms vanish,
including when allocation depends on other groups' pilots. Independent
pilots then give the exact unconditional identity

```
E[V]=p
Cov(V)=diag{p_j(1-p_j)[lambda^2/m0+(1-lambda)^2 E(1/M_j)]}.
```

This does not extend automatically to naive pooled reuse: its pilot weight
m0/(m0+M_j(H)) is random and correlated with the pilot error. The exact
covariance identity concerns raw estimates, not normalized advantages.

Let P=I-11^T/K, z=Pp, v=||z||^2/K, s=sqrt(v+epsilon), and
f(p)=z/s. Direct differentiation gives

```
J(p)=P/s - zz^T/(K s^3).
```

With e=V-p, delta=f(V)-f(p), L=J(p)e, R=delta-L, the following are
identities, rather than asymptotic approximations:

```
E[delta]=E[R]                 (because E[L]=0)
MSE=||E[delta]||^2/K + tr(Cov(delta))/K
MSE=E||L||^2/K + 2E[L^T R]/K + E||R||^2/K
E||L||^2/K=tr(J Cov(V) J^T)/K.
```

Thus unbiased V does not imply unbiased f(V); squared bias is only one
part of nonlinear risk. A large negative cross term can cancel a large
linear risk and remainder energy. Reporting only the squared bias misses
this mechanism.

For fixed lambda, the pilot contribution to the linear risk is allocation
independent. Its main contribution is a positive constant times
sum_j c_j/M_j, where c_j=p_j(1-p_j)||J[:,j]||^2. The positive-integer
allocator starts each M_j at1, then picks the largest marginal decrease
c_j/[M_j(M_j+1)]. Marginal decreases within each group are nonincreasing,
so selecting the largest available decrease at every step selects the
largest feasible prefix of these sequences. This proves global optimality
for the separable surrogate. It does not prove optimality for exact risk.
This diagnostic independently checks all35 positive allocations at main8.

For direction vectors u,w the Hessian bilinear form is

```
D^2 f(p)[u,w]
 = -(Pu*(z^T w)+Pw*(z^T u)+z*(u^T P w))/(K s^3)
   + 3z*(z^T u)*(z^T w)/(K^2 s^5).
```

Its norm is at most 6||Pu||||Pw||/(sqrt(K)*s^2), using
||z||<=sqrt(K)*s. Along the line from p to V, Taylor's integral remainder
therefore obeys

```
||R|| <= 3||Pe||^2/(sqrt(K)*s_min^2),
s_min = min_{0<=t<=1} sqrt(||P(p+t e)||^2/K+epsilon).
```

Even large dispersion at p does not by itself bound s_min away from zero
on every sampled segment. A small average noise-to-dispersion ratio is a
scale diagnostic, not a tail or global remainder guarantee.

Write Q=E||L||^2/K, T=E||R||^2/K. Cauchy-Schwarz gives the exact bound
abs(MSE-Q)<=2 sqrt(Q*T)+T. If Q>0 and sqrt(T/Q)<=sqrt(1.1)-1,
the discrepancy is at most10% of Q. This sufficient condition requires
control of remainder energy; it is not certified merely by dispersion.

At p=a*1, f(p)=0 and J=P/sqrt(epsilon). Uniform allocation has a
permutation-symmetric advantage distribution with zero mean, yet
MSE<=1 while Q grows as1/epsilon for nonzero sampling variance. This
is an explicit failure of local calibration with zero normalized bias.

The run uses all consumed vectors and all frozen paths. Its thresholds
are descriptive, its reward-state moments are exact up to floating-point
error, and it creates no new population confirmation or training evidence.
