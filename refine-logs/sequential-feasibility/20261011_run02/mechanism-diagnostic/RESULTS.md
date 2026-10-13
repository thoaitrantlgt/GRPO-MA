# One-action mechanism diagnosis

**Exploratory diagnosis; primary candidate verdict remains unchanged.**

## Terminal-count identity

Suppose B=K*t, every current count n_i<=t, and the contemplated action satisfies n_j+1<=t. Balanced completion fills all thoughts to t: exactly K*t observations are available and no thought starts above t. Independent Bernoulli reward streams therefore produce the same conditional distribution of final success counts regardless of which admissible thought is sampled first. The terminal estimator depends only on final counts and successes, so Q_j(p,D,q)=Q_U(p,D,q) for every p. This is an identity, independent of the prior or quadrature.

Starting from balanced coverage, this one-action score cannot prefer a deviation before a thought reaches terminal quota. For B=16, preferences can first arise during the last one observation at K2 or the last three observations at K4. It cannot value early information that would support later adaptive allocation. Simply tightening posterior integration or raising pilot counts cannot remove this structural limitation.

| K | Current count per thought | Histories checked | Maximum absolute pointwise gain |
|---:|---:|---:|---:|
| 2 | 1 | 4 | 8.88e-16 |
| 2 | 7 | 64 | 8.88e-16 |
| 4 | 1 | 16 | 3.55e-15 |
| 4 | 3 | 256 | 5.33e-15 |

## Full Bayes information control

This control uses only observed counts and the frozen finite prior, but solves the entire adaptive Bellman problem without the proposed reliability gate. It is an exploratory small-case diagnostic, not a successful replacement method or a true-p oracle.

| K | m0 | Development reduction | Maximum projected stress regression | Bellman states | CPU seconds |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.1204% | 2387.4868% | 4540 | 0.781 |
| 2 | 2 | 0.1204% | 2387.4868% | 4004 | 0.868 |

The finite-prior optimization bound applies under its own support distribution; gains on the consumed heterogeneous population are descriptive. No thresholds, prior or evaluation vectors were changed. The one-action acquisition must be redesigned to account for future adaptive allocation before spending on a larger sweep.
