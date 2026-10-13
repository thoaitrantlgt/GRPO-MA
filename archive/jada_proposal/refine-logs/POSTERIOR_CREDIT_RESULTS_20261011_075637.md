# Concrete posterior-credit prototype completed

**STOP_POSTERIOR_CREDIT_PROTOTYPE**. Development data consumed; no new confirmation. GPU cost0.

The estimator is E[h_tau(p)|counts] under fixed iid Jeffreys prior; allocation stays uniform, K4/total16. Target uses sample standard deviation with floor1e-4. Runtime uses observed counts and a frozen625-entry table only. It does not know true p or score geometry.

| Four-Gram consumed-data comparison | Update-MSE reduction |
|---|---:|
| posterior-credit vs plugin | 16.421% |
| posterior-credit vs prior-scale | 7.925% |

The mean criteria pass, but 1/24 primary stress cells worsen >10%; sparse-extreme/paired-correlated worsens17.815%. Half-support maximum credit difference is0.042499, exceeding the frozen0.02 stability tolerance. Both failures block continuation under the complete declared rule. Do not override them with favorable averages.

The scalar control coefficient0.862610 was fitted from prior predictive risk only, before development risk access. All96 consumed mixture vectors, six stress regimes,27 fixed paths and seven existing Grams are retained. The primary average uses the same four nontrivial synthetic geometries. No prior, geometry, distribution, coefficient or method switch was selected after outcomes.

Prior fit uses2048 base Sobol points across four independent scrambles, each augmented with24 coordinate permutations (49,152 weighted support entries). Finite-support Bayes optimality, coordinate exchangeability, centering, null updates and bias/variance identities pass local audit. The continuous-prior integration is approximate and fails its stability check; exact outcome risk refers to the frozen numerical estimator, not the exact continuous posterior.

Completed 2709 exact method/vector/Gram evaluations; prior fit5.68s CPU, evaluation0.67s. All32 repository tests PASS. Local verification only; no fresh independent reviewer clearance.

[Full results](posterior-credit/20261011_run01/RESULTS.md) · [Prototype design](../experiments/posterior_credit/DESIGN.md) · [Raw metrics](posterior-credit/20261011_run01/metrics.csv) · [Plot](posterior-credit/20261011_run01/posterior_credit.png)

No trainer integration, GPU, model download, new confirmation, numerical rescue run or alternative prior was launched. Previous allocator and gradient gates remain unchanged. The prototype demonstrates a mathematically coherent conditional credit estimator but does not pass its development gate.
