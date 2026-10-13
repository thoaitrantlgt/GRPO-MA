# Posterior-credit research prototype

Run `python -m experiments.posterior_credit.run` from the repository root.
Existing output directories cannot be overwritten; choose a new identifier
if a separately authorized run is needed. This is a CPU research prototype,
not a trainer integration or a deployable method.

Uniform sampling uses four Bernoulli observations per thought, K4/total16.
An offline625-entry table approximates E[h_tau(p)|counts] under fixed Jeffreys
prior. The same frozen prior determines a scalar-calibration control before
development risk access. No current true probability or Gram is used at runtime.

The development run failed its complete stress/stability rule despite mean
improvement. Continuous-prior optimality is not certified. Raw metrics retain
all consumed vectors, stress regimes, paths and fixed synthetic geometries.
No fresh confidence interval or training improvement is claimed.

See [design and derivation](DESIGN.md) and [protocol](protocol.json).
Tests verify direct finite-prior posterior sums, integrated Bayes dominance,
centering, equal-count symmetry and coordinate exchangeability.

The separate fixed-schedule convergence diagnosis runs with
`python -m experiments.posterior_credit.run_convergence`. It preserves prior,
target, budget, geometry and the original scalar control. Eight nested Sobol
streams reach32768 points each. Highest-level numerical criteria pass, but
the same stress cell still worsens17.54%, yielding
`STOP_POSTERIOR_CREDIT_AFTER_CONVERGENCE`. This is numerical evidence rather
than a certified continuous-integration error bound.

`--resume-frozen` permits recovery of an incomplete evaluation using saved
tables, never a refit or overwrite of completed results. Source vectors are
checked against the consumed original run. The missing-source-snapshot
interruption and unchanged frozen tables are documented in RESUME_NOTE.json.
