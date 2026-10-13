# Derivation Package

## Target

Explain the finite-budget tradeoff of reducing normalized-advantage amplitude,
and falsify a fixed half-amplitude estimator under an unchanged target.
The immediate goal is exact risk algebra and a descriptive CPU probe.

## Status

COHERENT AS STATED. This is a normalization mechanism probe, not a claim
that amplitude shrinkage preserves a policy-gradient objective.

## Invariant Object

R(c,p)=E[||c*f_epsilon(V)-f_epsilon(p)||^2]/K. Every estimator is scored
against the same original target; changing the target to match the candidate
would answer a different question.

## Assumptions

- K=4 independent Bernoulli groups with fixed true probabilities p.
- Uniform positive sample counts n=total/K, total in {8,16,32}.
- V_j is the pooled group sample mean; no adaptive allocator or pilot tuning.
- epsilon=1e-6 is fixed. Scalar c is deterministic; the candidate uses c=1/2.
- The oracle control knows p and exact moments; it is unavailable at runtime.
- Consumed vectors and predetermined dispersion paths give descriptive
  evidence, not a new confirmation or new population intervals.

## Notation

P=I-11^T/K; f_epsilon(x)=Px/sqrt(||Px||^2/K+epsilon).
A=f_epsilon(V), t=f_epsilon(p), u=E[A].
a=E[||A||^2]/K, b=u^T*t/K, d=||t||^2/K.
All expectations are over count states at fixed p.

## Derivation Strategy

Expand the unchanged exact quadratic loss in c, separate bias and variance,
then use equality and noiseless boundaries to establish the tradeoff before
enumerating all reward states. No delta-method approximation is needed.

## Derivation Map

1. The invariant risk depends on a,b,d, obtained by exact finite count sums.
2. Completing the square gives the scalar oracle and excess risk.
3. Substitution c=1/2 gives a precise improvement condition.
4. Boundary cases show that a universal fixed shrinkage improvement is false.
5. A frozen empirical rejection rule tests the weaker benchmark hypothesis.

## Main Derivation

Step1 (**identity**): expanding the square yields R(c,p)=a*c^2-2*b*c+d.
The bias/variance identity is
R(c,p)=||c*u-t||^2/K+c^2*(a-||u||^2/K).
Shrinkage lowers variance by c^2; its change in squared bias is not generally
of a fixed sign.

Step2 (**proposition**): if a>0, the unrestricted minimizer is c*=b/a and
R(c,p)-R(c*,p)=a*(c-c*)^2. For c in [0,1], convexity gives the clipped
minimizer clip(b/a,0,1). If a=0, A=0 almost surely and all c have risk d.
The oracle is a control, not a pilot-feasible estimator.

Step3 (**identity**): R(1,p)-R(1/2,p)=3*a/4-b. Thus fixed half-amplitude
improves precisely when b<3*a/4. This is an alignment-to-energy condition,
not simply a small-dispersion condition.

Step4 (**proposition: boundaries**): at equal probabilities, t=0 and symmetry
gives u=0 under uniform sampling. Hence b=d=0 and R(1/2,p)=R(1,p)/4.
At a deterministic nonconstant Bernoulli vector p in {0,1}^K, V=p almost
surely, so a=b=d>0, R(1,p)=0, R(1/2,p)=d/4>0. More generally, as n grows
with nonconstant p fixed, V converges to p, f is bounded and continuous,
and dominated convergence gives the same limiting bias penalty. Therefore
fixed c<1 cannot uniformly dominate c=1 across signal/noise and budgets.

Step5 (**exact computation**): enumerate every count vector in {0,...,n}^4,
weight it by the product Binomial PMF, and compute a,b,u. The target, c,
distribution, budgets and rejection rule are frozen before these outcomes.

## Remarks and Interpretation

- This changes amplitude, not the normalized direction. It isolates whether
  noisy normalized amplitudes can explain apparent risk headroom.
- Alternative denominator regularization, independent scale estimation and
  data-dependent shrinkage are different designs. None is selected or tuned
  in this probe. The fixed half coefficient is a simple falsifiable device.
- Finite-budget benefit can coexist with high-signal regression and
  asymptotic inconsistency. Average benefit alone is insufficient.
- We retain a zero estimator as an explicit degenerate control: beating
  plug-in MSE near equality is not evidence of useful credit information.

## Boundaries and Non-Claims

- Normalized-advantage MSE is an estimation objective; it is not training
  progress, reward improvement, measured gradient error or novelty.
- A globally constant multiplier also scales a fixed-score gradient; its
  effect can overlap a learning-rate change. No distinct learning algorithm
  or empirical training advantage follows from this probe.
- The original allocator NO-GO and gradient objective gate are unchanged.
- No oracle coefficient or development winner is deployed or confirmed.

## Open Risks

- Data-dependent coefficients need additional risk terms because c and A
  can be dependent; the deterministic-c oracle formula cannot certify them.
- Reward-correlated scores and estimated normalization scales are absent.
- Consumed development data can guide mechanisms but cannot certify a
  population gain. A new confirmation would require an independently fixed
  future design, if there is a sufficient reason to pursue it.
