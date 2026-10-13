# Posterior conditional-target credit: concrete research design

## Target

Estimate the unchanged conditional-value normalized advantage using the
posterior expectation of that target, with uniform continuation counts.

## Status

COHERENT AS STATED for fixed conditional Bernoulli values and fixed score
geometry; approximate for the continuous prior and unvalidated for real
trainer updates. The target is the sample-std/floor h_tau(p) defined in the
[target contract](../../idea-stage/docs/ADVANTAGE_TARGET_CONTRACT.md).
This design estimates credits; it does not optimize sample allocations.

## Invariant Object

The loss is E[||S(a(D)-T)||²], the conditional thought-update estimation
loss from the target contract.

## Assumptions

Uniform independent Bernoulli rewards; fixed counts and prior; stopped
credit coefficients; fixed score geometry independent of unknown values.

## Notation

Uniform n=4 rewards per K=4 thought, total16. Counts C_j are independent
Binomial(4,p_j). Prior p_j iid Beta(1/2,1/2), fixed before outcomes.
Let T=h_tau(p), D=C, a(D)=E[T|D] and S a fixed thought-score matrix.
The loss is E[||S(a(D)-T)||²]. G=S^T S is positive semidefinite, fixed
independently of unknown p and continuation outcomes. Conditional scores
that reveal information about p require conditioning the prior on that
information; this prototype does not model that dependence.

## Derivation Strategy

Use conditional centering to prove Bayes risk optimality, then approximate
the prior offline and evaluate the frozen table by complete count enumeration.

## Derivation Map

Posterior law -> conditional target mean -> update-risk decomposition ->
finite-support integration -> matched-information scale control.

## Main Derivation

1. **Identity:** under the continuous prior p_j|C_j follows
   Beta(C_j+1/2,4-C_j+1/2). The estimator integrates h_tau of p itself;
   it is generally different from h_tau(E[p|C]).
2. **Identity:** for any D-measurable credit b(D), conditional centering gives
   E[(b-T)^T G(b-T)|D]
   = E[(a-T)^T G(a-T)|D]+(b-a)^T G(b-a).
   The cross term vanishes because E[T-a|D]=0.
3. **Proposition:** a(D) minimizes posterior expected update error for every
   such fixed PSD G; uniqueness in credit space is unnecessary when G has
   a nullspace. It does not require G at runtime. This is an integrated
   Bayes statement, not pointwise frequentist dominance at every p.
4. **Approximation:** four512-point scrambled Sobol supports, augmented by
   all24 coordinate permutations, approximate the continuous prior. Pool
   joint numerator and predictive mass, then divide. Never average separately
   normalized tables. Bayes optimality is exact for this finite support;
   continuous-prior optimality is not certified by the numerical table.
5. **Identity:** prior-scale baseline minimizes frozen prior predictive loss
   for c*h_tau(C/4), c in [0,1], under the same primary fixed-Gram average.
   It is fitted from prior/support only, before development vectors are read.
   Runtime information and sampling budget match the candidate.

## Runtime and prototype

Offline625-entry table; at runtime index the four observed counts and return
four credits. The table is centered, coordinate-exchangeable by construction,
and bounded as a conditional mean of bounded targets. It does not access
true p, measured gradients, main outcomes before sampling, or an adaptive
allocation policy. Uniform is used throughout. Offline CPU integration
cost and exact evaluation time are reported separately.

## Frozen falsification

See [protocol](protocol.json). Use only consumed96 mixture vectors, six
stress regimes and27 fixed dispersion probes. Preserve all seven previous
synthetic Grams; average the same four nontrivial ones for the primary.
Require >5% development mean reduction against both plug-in and prior-scale,
no >10% regression in any stress/primary-Gram cell, and maximum half-support
table difference<=0.02. Failure stops this prototype; a favorable mean cannot
rescue a stress or stability failure. No confidence interval on consumed data.

## Remarks and Interpretation

- This is nonlinear posterior inference of the unchanged credit target,
  rather than a constant scaling of plug-in credits or a new normalizer.
- Bayes optimality under a chosen prior says nothing about a mismatched
  benchmark population or every fixed p. Report all stress regressions.

## Boundaries and Non-Claims

- At equal-count histories symmetry requires zero credits, while the real
  target can be nonzero. Samples cannot identify every true target exactly.
- As counts grow, an appropriately convergent continuous posterior can
  approach the true target; this fixed-budget table proves no asymptotic rate.
- Only ratio1/fixed-score update loss is evaluated. Clipping, reward-score
  dependence, non-Bernoulli reward sums, real score geometry and full training
  gains remain open. The answer and KL branches remain outside the probe.
- No trainer integration, fresh confirmation, geometry selection, prior
  replacement, parameter sweep, GPU or novel-method claim is authorized by
  a positive development mean alone.

## Open Risks

Continuous-prior quadrature may be unstable even though evaluation of the
frozen table is exact. Failure of the half-support diagnostic blocks a
continuous-posterior performance claim. Stress failures apply to the actual
frozen prototype, not necessarily the exact continuous-prior estimator.

## Subsequent numerical diagnosis

The separately declared eight-scramble nested convergence schedule reached
32768 base points per scramble. Half-scramble max difference0.000257,
adjacent max credit change0.001278 and normalized maximum stress-risk
change0.000434 pass all three numerical tolerances. The original failed
sparse-extreme/paired-correlated cell still worsens17.537%, so the numerical
recommendation is STOP_POSTERIOR_CREDIT_AFTER_CONVERGENCE. Original prototype
outcomes and gates are preserved. This diagnoses stability without proving
zero quadrature error or pointwise inferiority of every posterior estimator.
