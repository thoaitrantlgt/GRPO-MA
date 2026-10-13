# Sequential sampling: CPU feasibility results

**Status: consumed-data development only. No fresh confirmation or GPU.**

Verdict: **NO_DEVELOPMENT_SIGNAL**. Aggregate gated reduction: 0.0000%; descriptive development 95% interval [0.0000%, 0.0000%].

The intervals resample already-consumed vectors; they cannot establish a fresh generalization claim. Reward-state risks are exact for each frozen implemented policy; posterior integration remains approximate.

| K | B | m0 | Pilot fraction | Method | Reduction vs uniform | Development interval | Maximum stress regression |
|---:|---:|---:|---:|---|---:|---|---:|
| 2 | 16 | 1 | 0.125 | uniform | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 1 | 0.125 | variance-only | 0.0000% | [-0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 1 | 0.125 | JADA | 0.0000% | [-0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 1 | 0.125 | ungated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 1 | 0.125 | gated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 2 | 0.250 | uniform | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 2 | 0.250 | variance-only | -5.8509% | [-6.3043%, -5.4523%] | 47.5122% |
| 2 | 16 | 2 | 0.250 | JADA | 0.0000% | [-0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 2 | 0.250 | ungated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 2 | 0.250 | gated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 1 | 0.250 | uniform | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 1 | 0.250 | variance-only | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 1 | 0.250 | JADA | -1.2857% | [-1.4333%, -1.1569%] | 1.9799% |
| 4 | 16 | 1 | 0.250 | ungated | 0.0218% | [0.0085%, 0.0370%] | 0.0006% |
| 4 | 16 | 1 | 0.250 | gated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 2 | 0.500 | uniform | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 2 | 0.500 | variance-only | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 2 | 0.500 | JADA | -0.1931% | [-0.2945%, -0.0818%] | 1.7199% |
| 4 | 16 | 2 | 0.500 | ungated | 0.0218% | [0.0085%, 0.0370%] | 0.0006% |
| 4 | 16 | 2 | 0.500 | gated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |

K2 stress vectors are coordinate projections. The original six-regime stress gate refers to K4.

## Acquisition and runtime

| K | m0 | Method | Preferred observations / remaining budget | Numerical fallback | Reliability fallback | Exact-evaluation seconds |
|---:|---:|---|---:|---:|---:|---:|
| 2 | 1 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.012 |
| 2 | 1 | variance-only | 0.0000% | 0.0000% | 0.0000% | 0.013 |
| 2 | 1 | JADA | 0.0000% | 0.0000% | 0.0000% | 0.013 |
| 2 | 1 | ungated | 0.0000% | 90.5126% | 0.0000% | 0.212 |
| 2 | 1 | gated | 0.0000% | 90.5126% | 0.0000% | 0.012 |
| 2 | 2 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.010 |
| 2 | 2 | variance-only | 0.0000% | 0.0000% | 0.0000% | 0.022 |
| 2 | 2 | JADA | 0.0000% | 0.0000% | 0.0000% | 0.023 |
| 2 | 2 | ungated | 0.0000% | 88.9313% | 0.0000% | 0.157 |
| 2 | 2 | gated | 0.0000% | 88.9313% | 0.0000% | 0.011 |
| 4 | 1 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.296 |
| 4 | 1 | variance-only | 0.0000% | 0.0000% | 0.0000% | 3.563 |
| 4 | 1 | JADA | 0.0000% | 0.0000% | 0.0000% | 3.092 |
| 4 | 1 | ungated | 0.1020% | 98.7192% | 0.0000% | 1.583 |
| 4 | 1 | gated | 0.0000% | 98.8130% | 0.1020% | 0.303 |
| 4 | 2 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.359 |
| 4 | 2 | variance-only | 0.0000% | 0.0000% | 0.0000% | 4.789 |
| 4 | 2 | JADA | 0.0000% | 0.0000% | 0.0000% | 4.198 |
| 4 | 2 | ungated | 0.1529% | 98.7712% | 0.0000% | 1.315 |
| 4 | 2 | gated | 0.0000% | 98.9120% | 0.1529% | 0.315 |

See [metrics.csv](metrics.csv), [effects.csv](effects.csv), [acquisition.json](acquisition.json), [paired_mc.csv](paired_mc.csv), [protocol.json](protocol.json), and [summary.json](summary.json).

## Decision limits

A fallback-dominated policy can reproduce uniform without demonstrating that sequential acquisition has no potential. The two small nested quadrature levels are a feasibility test; disagreement must be reported, not hidden by fallback. No prior, reliability threshold, benchmark distribution or terminal estimator was tuned on outcomes.

The trial excludes the full 26-configuration sweep, true-p nonlinear oracle, historical reuse factorial, synthetic gradient validation and training. Those stages are not counted as completed.


## Independent reconstruction and mechanism diagnosis

Ten implementation tests PASS. Independent fixed-count outcome summation reproduces uniform risks with maximum absolute error 1.02e-14; gated risk is identical to uniform in all four configurations. [Numerical audit](AUDIT.json).

Paired MC uses 512 episodes per vector/configuration. There are 120 zero-empirical-SE discrepancies that remain unresolved; studentized checks do not validate these rare effects. All exact effects lie inside the reported pointwise Hoeffding intervals, which are too wide to resolve small gains. Exact reward-state propagation supplies the primary risks; no additional MC trials were selected after outcomes.

The one-action terminal-count identity is independently verified. See [mechanism report](mechanism-diagnostic/RESULTS.md) for the derivation and exploratory K2 Bayes control. This explains why early information acquisition is absent; it is not a universal impossibility result for sequential allocation.

[Pilot fraction and fixed-dispersion plot](feasibility.png) and [subgroup table](dispersion_bins.csv) use consumed data. Leverage/gradient evaluation is deferred because the credit prototype has no development signal. No geometry was selected to rescue it.

K=2, m0=1: configuration wall time 12.75s; MC 12.47s; cached risk-vector payload 4.04 MiB. Payload excludes Python dictionaries, score cache, outcome arrays and other process memory; peak RSS was not measured.

K=2, m0=2: configuration wall time 11.83s; MC 11.59s; cached risk-vector payload 3.94 MiB. Payload excludes Python dictionaries, score cache, outcome arrays and other process memory; peak RSS was not measured.

K=4, m0=1: configuration wall time 22.57s; MC 13.71s; cached risk-vector payload 63.00 MiB. Payload excludes Python dictionaries, score cache, outcome arrays and other process memory; peak RSS was not measured.

K=4, m0=2: configuration wall time 22.49s; MC 11.49s; cached risk-vector payload 59.06 MiB. Payload excludes Python dictionaries, score cache, outcome arrays and other process memory; peak RSS was not measured.

Decision: STOP this one-action/balanced-completion prototype before the full sweep. A new acquisition design must account for the future adaptive allocation decisions affected by the observation, rather than only reordering samples under the same terminal counts. No GPU escalation.
