# Estimator research results

Original JADA CPU gate: **NO-GO**. On the original predeclared mixture, budget-precommitted JADA increases MSE by 3.24% [2.89%,3.59%]. Full results: [CPU gate](cpu-estimator-gate/20261011_run01/RESULTS.md).

New exact nonlinear-oracle gate: **HEADROOM for CPU method development**, scoped to K4/B16/m0=1,2. On 96 fresh confirmation vectors, fixed true-p nonlinear allocation reduces MSE by 6.70% [5.24%,8.16%]; true-p-plus-pilot adaptive nonlinear allocation reduces it by 10.22% [8.87%,11.57%]. The true-p local-surrogate oracle makes MSE worse by 4.78%, indicating objective mismatch even without plug-in estimation noise. Full results: [nonlinear gate](nonlinear-oracle-gate/20261011_run01/RESULTS.md).

No deployable pilot-only nonlinear allocator has been evaluated yet. Both oracle methods know true probabilities, unlike a practical method. The two new baseline comparisons coincide because variance-only equals uniform in this small-budget design. Oracle headroom does not reverse the JADA NO-GO or establish training gains.

All 12 tests pass; exact saved-objective audit verifies global optima, coverage and reported risks. Both experiments used only CPU; GPU cost 0. Next justified work: pilot-only posterior nonlinear-risk allocation on CPU, followed by a newly declared confirmation set. No frozen-model or online training was launched.
