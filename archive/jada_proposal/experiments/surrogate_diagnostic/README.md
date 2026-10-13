# CPU surrogate diagnosis

Run from the repository root: `python -m experiments.surrogate_diagnostic.run`.
Output directories are protected from overwrite; select a new run identifier
before rerunning. The completed run is `20261011_run02`; run01 is preserved
as an unsuccessful CSV export with completed raw JSON moments.

The frozen source is96 consumed vectors and six stress regimes. All27
deterministic dispersion probes are retained. Exact enumeration covers K4,
total16, m0=2, all35 positive main allocations, and all pilot/main states.
This is a descriptive mechanism diagnosis, with no new confidence intervals,
benchmark selection, GPU dependency, or model/training implication.

`python -m unittest tests.test_surrogate_diagnostic` checks moment identities,
an independent existing exact estimator, and the equal-probability boundary.
See [derivations](DERIVATIONS.md) for covariance, Jacobian, allocator proof,
and an exact remainder bound.
