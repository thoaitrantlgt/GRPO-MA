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
| 2 | 16 | 2 | 0.250 | JADA | -6.7437% | [-7.3627%, -6.1908%] | 33233.7565% |
| 2 | 16 | 2 | 0.250 | ungated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 2 | 16 | 2 | 0.250 | gated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 1 | 0.250 | uniform | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 1 | 0.250 | variance-only | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 1 | 0.250 | JADA | -1.2819% | [-1.4522%, -1.1415%] | 2.9541% |
| 4 | 16 | 1 | 0.250 | ungated | -0.0199% | [-0.0489%, 0.0096%] | 0.0000% |
| 4 | 16 | 1 | 0.250 | gated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 2 | 0.500 | uniform | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 2 | 0.500 | variance-only | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |
| 4 | 16 | 2 | 0.500 | JADA | -0.2507% | [-0.4183%, -0.0774%] | 1.7199% |
| 4 | 16 | 2 | 0.500 | ungated | -0.0199% | [-0.0489%, 0.0096%] | 0.0000% |
| 4 | 16 | 2 | 0.500 | gated | 0.0000% | [0.0000%, 0.0000%] | 0.0000% |

K2 stress vectors are coordinate projections. The original six-regime stress gate refers to K4.

## Acquisition and runtime

| K | m0 | Method | Preferred observations / remaining budget | Numerical fallback | Reliability fallback | Exact-evaluation seconds |
|---:|---:|---|---:|---:|---:|---:|
| 2 | 1 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.017 |
| 2 | 1 | variance-only | 0.0000% | 0.0000% | 0.0000% | 0.007 |
| 2 | 1 | JADA | 0.0000% | 0.0000% | 0.0000% | 0.006 |
| 2 | 1 | ungated | 0.0000% | 90.4980% | 0.0000% | 0.255 |
| 2 | 1 | gated | 0.0000% | 90.4980% | 9.5020% | 0.016 |
| 2 | 2 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.011 |
| 2 | 2 | variance-only | 0.0000% | 0.0000% | 0.0000% | 0.010 |
| 2 | 2 | JADA | 0.0000% | 0.0000% | 0.0000% | 0.009 |
| 2 | 2 | ungated | 0.0000% | 88.9143% | 0.0000% | 0.226 |
| 2 | 2 | gated | 0.0000% | 88.9143% | 11.0857% | 0.014 |
| 4 | 1 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.078 |
| 4 | 1 | variance-only | 0.0000% | 0.0000% | 0.0000% | 0.185 |
| 4 | 1 | JADA | 0.0000% | 0.0000% | 0.0000% | 0.159 |
| 4 | 1 | ungated | 0.0840% | 98.6880% | 0.0000% | 1.762 |
| 4 | 1 | gated | 0.0000% | 98.7663% | 1.2337% | 0.080 |
| 4 | 2 | uniform | 0.0000% | 0.0000% | 0.0000% | 0.082 |
| 4 | 2 | variance-only | 0.0000% | 0.0000% | 0.0000% | 0.362 |
| 4 | 2 | JADA | 0.0000% | 0.0000% | 0.0000% | 0.332 |
| 4 | 2 | ungated | 0.1260% | 98.7356% | 0.0000% | 1.723 |
| 4 | 2 | gated | 0.0000% | 98.8531% | 1.1469% | 0.086 |

See [metrics.csv](metrics.csv), [effects.csv](effects.csv), [acquisition.json](acquisition.json), [paired_mc.csv](paired_mc.csv), [protocol.json](protocol.json), and [summary.json](summary.json).

## Decision limits

A fallback-dominated policy can reproduce uniform without demonstrating that sequential acquisition has no potential. The two small nested quadrature levels are a feasibility test; disagreement must be reported, not hidden by fallback. No prior, reliability threshold, benchmark distribution or terminal estimator was tuned on outcomes.

The trial excludes the full 26-configuration sweep, true-p nonlinear oracle, historical reuse factorial, synthetic gradient validation and training. Those stages are not counted as completed.
