# Fixed normalization-amplitude CPU probe

**REJECT_FIXED_HALF_AS_GENERAL_FOLLOWUP**. Consumed-data feasibility diagnosis only; no new confirmation interval. GPU cost0.

The unchanged target is f_epsilon(p). Uniform pooled sampling spends all total8/16/32 samples. The only candidate is0.5*f_epsilon(V), fixed before these outcomes. Plug-in, zero, and true-p scalar oracle are controls. All96 consumed mixture vectors, six stress regimes and27 fixed dispersion paths are retained.

## Exact aggregate risks: consumed mixture vectors

| Total | Estimator | MSE | Squared bias | Variance | Reduction vs plug-in |
|---|---|---:|---:|---:|---:|
| 8 | plugin | 0.901805 | 0.349412 | 0.552392 | 0.00% |
| 8 | fixed-half | 0.743923 | 0.605825 | 0.138098 | 17.51% |
| 8 | zero | 0.999690 | 0.999690 | 0.000000 | -10.85% |
| 8 | oracle-scalar | 0.692250 | 0.538048 | 0.154202 | 23.24% |
| 16 | plugin | 0.742004 | 0.235013 | 0.506991 | 0.00% |
| 16 | fixed-half | 0.641874 | 0.515126 | 0.126748 | 13.49% |
| 16 | zero | 0.999690 | 0.999690 | 0.000000 | -34.73% |
| 16 | oracle-scalar | 0.569363 | 0.402951 | 0.166412 | 23.27% |
| 32 | plugin | 0.565475 | 0.147280 | 0.418196 | 0.00% |
| 32 | fixed-half | 0.540500 | 0.435951 | 0.104549 | 4.42% |
| 32 | zero | 0.999690 | 0.999690 | 0.000000 | -76.79% |
| 32 | oracle-scalar | 0.439304 | 0.279257 | 0.160047 | 22.31% |

## Every stress result for the fixed-half candidate

| Regime | Total | Plug-in MSE | Half MSE | Relative reduction |
|---|---:|---:|---:|---:|
| symmetric-wide | 8 | 0.362915 | 0.432076 | -19.06% |
| symmetric-mid | 8 | 0.485727 | 0.495130 | -1.94% |
| all-equal | 8 | 0.929675 | 0.232419 | 75.00% |
| near-equal | 8 | 1.700718 | 1.117191 | 34.31% |
| asymmetric | 8 | 0.374997 | 0.439227 | -17.13% |
| sparse-extreme | 8 | 0.129863 | 0.314937 | -142.51% |
| symmetric-wide | 16 | 0.203057 | 0.351534 | -73.12% |
| symmetric-mid | 16 | 0.280224 | 0.390224 | -39.25% |
| all-equal | 16 | 0.972347 | 0.243087 | 75.00% |
| near-equal | 16 | 1.649357 | 1.080932 | 34.46% |
| asymmetric | 16 | 0.201161 | 0.350649 | -74.31% |
| sparse-extreme | 16 | 0.068029 | 0.284013 | -317.49% |
| symmetric-wide | 32 | 0.108745 | 0.304370 | -179.89% |
| symmetric-mid | 32 | 0.149545 | 0.324770 | -117.17% |
| all-equal | 32 | 0.989457 | 0.247364 | 75.00% |
| near-equal | 32 | 1.537332 | 1.020724 | 33.60% |
| asymmetric | 32 | 0.101419 | 0.300707 | -196.50% |
| sparse-extreme | 32 | 0.035173 | 0.267585 | -660.76% |

## Findings

1. Observation: the frozen mean criteria pass, but 11/18 stress/budget cells regress by more than10%. Interpretation: finite-budget average benefit does not establish robust normalization. Implication: the complete frozen rule gives **REJECT_FIXED_HALF_AS_GENERAL_FOLLOWUP**. Next step: do not deploy or confirm fixed-half based on a favorable mixture mean.

2. Observation: half amplitude quarters variance at every fixed probability vector; bias changes as ||0.5*E[A]-target||²/K. Interpretation: this is an amplitude/bias tradeoff, not a correction of noisy advantage direction. Implication: improved scalar MSE alone can come from attenuating credits. Next step: a future design must specify whether the scientific objective is estimating the current normalized target or changing the target itself.

3. Observation: at equality, half amplitude cuts risk75%, but the zero control has zero risk; at nonconstant deterministic rewards, plug-in is exact while half amplitude has positive error. Interpretation: a universal fixed-shrinkage benefit is false. Implication: no training gain or distinct algorithm follows; multiplying a fixed-score update by a constant overlaps learning-rate scaling. Next step: document the required credit/update objective before choosing any new normalization mechanism.

## Frozen rejection rule

Reject fixed-half as a general follow-up if reduction <=5% at total16 OR nonpositive mean reduction at total8 or32 OR any of six stress regimes at any budget worsens >10% relative to plugin. No winner-based rescue, coefficient sweep or replacement candidate. The primary mean equally weights the unchanged six mixture components (16 vectors each); paths/stress cases are outside this mean. Exact reward enumeration removes outcome Monte Carlo error, not uncertainty over unseen probability vectors.

Exact identity: R(c)=a*c²-2*b*c+d; R(1)-R(0.5)=0.75*a-b. Oracle scalar clip(b/a,0,1) requires true-p moments and is unavailable to a runtime estimator. See [derivation package](../../../DERIVATION_PACKAGE.md).

[Raw metrics](metrics.csv) · [Protocol frozen before outcomes](protocol.json) · [Tradeoff plot](normalization_tradeoff.png) · [Numerical audit](AUDIT.json)
