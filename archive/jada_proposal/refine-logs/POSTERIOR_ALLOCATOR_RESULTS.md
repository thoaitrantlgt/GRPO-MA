# Pilot-only posterior allocator CPU results

**NO_CLEAR_SIGNAL** under the predeclared primary criterion. K4/B16/m0=1,2, budget-precommitted reuse; 96 new confirmation vectors from the unchanged six-component mixture, seed 3141592026.

| Method vs uniform/variance-only | MSE reduction | 95% interval |
|---|---:|---:|
| posterior-Jeffreys | 0.0125% | [0.0085%, 0.0165%] |
| posterior-mixture | 0.0125% | [0.0085%, 0.0165%] |
| JADA | -2.3348% | [-2.6506%, -2.0189%] |
| adaptive-nonlinear-oracle | 9.8172% | [8.6843%, 10.9502%] |

Jeffreys is primary. The known benchmark-mixture prior is secondary and cannot replace the failed primary. Both priors produce the same allocation probabilities. Uniform and variance-only coincide here. The 95% intervals describe uncertainty over probability vectors conditional on frozen policies; reward states are evaluated exactly, while continuous prior integration is approximate.

Policy structure: {"posterior-Jeffreys": {"1": {"histories": 16, "histories_nonuniform": 0}, "2": {"histories": 81, "histories_nonuniform": 8}}, "posterior-mixture": {"1": {"histories": 16, "histories_nonuniform": 0}, "2": {"histories": 81, "histories_nonuniform": 8}}}. First-half versus last-half Sobol policy disagreement is zero for all four tables; this is an integration stability diagnostic, not proof of exact continuous Bayes optimality.

The declared criterion requires the primary lower 95% reduction limit above 5% and positive gains at both pilot sizes. m0=1 equals uniform exactly; m0=2 yields a tiny improvement. Oracle headroom remains substantially larger than the recovered pilot-only gain.

Completed 588 exact cells / 4,704 method evaluations. All 17 tests and saved-policy/result audit pass. Fresh implementation review is same-family provisional. GPU cost 0.

[Full report](posterior-allocator-gate/20261011_run01/RESULTS.md) · [Figure](posterior-allocator-gate/20261011_run01/posterior_results.png) · [Derivations](../experiments/posterior_allocator/DERIVATIONS.md)

Preserve the original JADA NO-GO. Positive-result ablation planning is skipped because the primary gate failed. Before any new experiment, specify a distinct hypothesis about information available to a pilot-only allocator and a new confirmation protocol. The current confirmation set is consumed. No frozen-model or online training was launched.
