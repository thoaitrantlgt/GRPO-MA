# CPU estimator gate completed

**Decision: NO-GO for frozen-model/online expansion.** On the probability-vector mixture declared before new sampling, JADA with budget-precommitted reuse has normalized-advantage MSE 0.656480 versus variance-only 0.635887: **3.24% higher**, 95% interval **[2.89%,3.59%]**. Matched uniform MSE is 0.630517.

Full report: [RESULTS.md](cpu-estimator-gate/20261011_run01/RESULTS.md). Derivations: [DERIVATIONS.md](../experiments/cpu_gate/DERIVATIONS.md). Figures: [diagnostic curves](cpu-estimator-gate/20261011_run01/improvement_curves.png), also PDF and SVG in that directory.

- Completed all 26 feasible K/budget/pilot settings and all 16 allocation/reuse combinations, with six fixed regimes plus 96 independent stratified heterogeneous vectors per K. Distribution and actual vectors were saved before sampling.
- 2,652 probability-vector/settings cells, 42,432 method evaluations: 924 exact cells and 1,728 paired Monte Carlo cells, totaling 5,184,000 paired trials. Main sweep CPU wall time 119.4 seconds; plots, auditing and tie sensitivity add CPU work. GPU hours/cost zero.
- Seven tests pass, including analytic Jacobian, exact covariance, integer allocator optimum, K2 boundary, paired-MC agreement and stratified intervals. Fresh code review: same-family, provisional; both blockers fixed before sampling.
- K2 matches variance-only exactly. JADA increases MSE by 13.23% at K4, 1.37% at K8 and 0.18% at K16. Across K>=4 settings: 15 negative pointwise intervals and two inconclusive; none positive.
- The fixed-.5 ablation's tiny matched gain does not pass the primary gate and has worse absolute MSE than budget-precommitted reuse. Oracle local-risk improvement does not establish nonlinear-MSE benefit. Pilot fraction/dispersion/leverage charts are descriptive, with predeclared bins and uncertainty intervals.

All negative outcomes and the complete crossed table are saved in metrics.csv/results.jsonl. configuration_summary.csv and aggregate_risks.csv show setting-level comparisons and absolute risks; tie_sensitivity_summary.csv records order sensitivity. These are Bernoulli estimator results, not LLM, gradient or training evidence.
