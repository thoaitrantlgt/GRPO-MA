# CPU estimator research: consolidated evidence

**Decision: stop the current allocator family; no GPU or training escalation.** Raw-estimator covariance and separable allocation are correct. The local surrogate does not reliably represent finite-budget normalized-advantage risk, and pilot-only policies did not recover material nonlinear-oracle headroom.

## Evidence across completed gates

Relative reduction is (baseline risk - candidate risk)/baseline risk. Negative values mean worse error. Each interval belongs to its own originally frozen confirmation protocol; these rows are not a pooled meta-analysis.

| Stage / comparison | Reduction | Original paired CI95 | Decision / source |
|---|---:|---:|---|
| Broad JADA vs variance-only, fixed budget reuse | -3.2385% | [-3.5860%, -2.8910%] | NO-GO; [cpu-estimator-gate](./cpu-estimator-gate/20261011_run01/RESULTS.md) |
| Nonlinear adaptive-nonlinear-oracle-vs-uniform | 10.2214% | [8.8747%, 11.5681%] | HEADROOM; [nonlinear-oracle-gate](./nonlinear-oracle-gate/20261011_run01/RESULTS.md) |
| Nonlinear fixed-nonlinear-oracle-vs-uniform | 6.6990% | [5.2352%, 8.1628%] | HEADROOM; [nonlinear-oracle-gate](./nonlinear-oracle-gate/20261011_run01/RESULTS.md) |
| Nonlinear local-surrogate-oracle-vs-uniform | -4.7779% | [-6.7298%, -2.8261%] | HEADROOM; [nonlinear-oracle-gate](./nonlinear-oracle-gate/20261011_run01/RESULTS.md) |
| Pilot posterior Jeffreys vs uniform | 0.0125% | [0.0085%, 0.0165%] | NO_CLEAR_SIGNAL; [posterior-allocator-gate](./posterior-allocator-gate/20261011_run01/RESULTS.md) |
| Pilot information fixed-main-m0-4 | -0.0660% | [-0.1103%, -0.0218%] | NO_CLEAR_SIGNAL; [pilot-information-gate](./pilot-information-gate/20261011_run01/RESULTS.md) |
| Pilot information fixed-total-m0-4 | 0.0832% | [0.0304%, 0.1361%] | NO_CLEAR_SIGNAL; [pilot-information-gate](./pilot-information-gate/20261011_run01/RESULTS.md) |
| Fixed synthetic G adaptive-gradient-oracle-vs-uniform | 11.3580% | [10.3629%, 12.3530%] | NO_OBJECTIVE_SPECIFIC_SIGNAL; [gradient-objective-gate](./gradient-objective-gate/20261011_run01/RESULTS.md) |
| Fixed synthetic G adaptive-gradient-oracle-vs-adaptive-scalar-oracle | 3.0905% | [2.3924%, 3.7887%] | NO_OBJECTIVE_SPECIFIC_SIGNAL; [gradient-objective-gate](./gradient-objective-gate/20261011_run01/RESULTS.md) |
| Fixed synthetic G gradient-surrogate-oracle-vs-uniform | -5.1019% | [-6.6444%, -3.5593%] | NO_OBJECTIVE_SPECIFIC_SIGNAL; [gradient-objective-gate](./gradient-objective-gate/20261011_run01/RESULTS.md) |
| Fixed synthetic G gradient-JADA-vs-uniform | -1.6227% | [-1.9770%, -1.2683%] | NO_OBJECTIVE_SPECIFIC_SIGNAL; [gradient-objective-gate](./gradient-objective-gate/20261011_run01/RESULTS.md) |

The broad gate covered K={2,4,8,16}, total={16,32,64}, m0={1,2,4} for all26 feasible settings, all four allocation strategies crossed with all four reuse strategies. It included all six original regimes and a distribution fixed before winners: equal-weight iid Beta(.5,.5), Beta(2,2), Beta(8,8), Beta(.5,3), Beta(3,.5), and shared-location logistic-normal groups. It used924 exact cells and1,728 paired Monte Carlo cells, with5,184,000 paired trials. The broad primary interval clusters384 independent probability vectors; repeated budgets/pilots are not independent replications.

Subsequent exact gates used separate fresh96-vector confirmations from that same mixture when declared. Reward-state enumeration removes outcome-simulation error, but the paired vector-cluster intervals still describe uncertainty over probability vectors. All of these confirmation sets are now consumed. Deterministic stress regimes are separate from the population estimand.

The nonlinear oracle knows true probabilities; it demonstrates information headroom rather than deployability. Fixed-oracle and adaptive-oracle gains are distinct. The pilot-only posterior integrates a frozen prior using approximate quadrature and sees only pilot counts at runtime. Its tiny positive effect does not meet the declared5% materiality criterion.

Increasing pilot fraction did not recover material primary benefit, either holding main samples/reuse fixed or holding total cost fixed. The separate [convergence diagnosis](posterior-convergence/20261011_run01/RESULTS.md) stabilized the highest-level policies (half-policy TV0) while leaving consumed-data Jeffreys gains at -0.0264% fixed-main and0.1088% fixed-total. This is numerical evidence against this policy family, not an impossibility theorem or an exact continuous Bayes-optimality certificate.

The gradient gate supplies seven predeclared fixed synthetic PSD geometries; four nontrivial Grams form its primary average. Additional oracle benefit versus the scalar oracle is positive, but its lower95 limit does not exceed5%. Known-G pilot gradient-JADA and the true-p weighted local surrogate worsen risk. There is no measured LLM geometry, reward-score-dependence model, online training, or estimated-score cost claim.

## New exact mechanism diagnosis

[Full diagnosis](surrogate-diagnostic/20261011_run02/RESULTS.md), [raw moments](surrogate-diagnostic/20261011_run02/moments.csv), [allocation table](surrogate-diagnostic/20261011_run02/allocation_diagnosis.csv), and [figure](surrogate-diagnostic/20261011_run02/surrogate_diagnosis.png). This is posthoc analysis of96 consumed vectors, six stress cases, and27 frozen deterministic dispersion probes; it has no fresh population intervals.

| K4 / total16 / m0=2 | Exact MSE | Squared bias | Variance | Local predicted MSE | Descriptive reduction vs uniform |
|---|---:|---:|---:|---:|---:|
| uniform | 0.742004 | 0.235013 | 0.506991 | 5.275466 | 0.000% |
| JADA | 0.753401 | 0.237696 | 0.515704 | 5.589635 | -1.536% |
| local-surrogate-oracle | 0.771011 | 0.244651 | 0.526360 | 5.172296 | -3.909% |
| fixed-nonlinear-oracle | 0.704758 | 0.206951 | 0.497807 | 5.867546 | 5.020% |
| adaptive-nonlinear-oracle | 0.680017 | 0.192899 | 0.487118 | 6.066346 | 8.354% |

1. **Observation:** uniform squared bias accounts for31.67% of risk, while the surrogate predicts5.275466 versus exact0.742004. **Interpretation:** curvature/remainder terms are substantial; bias alone does not explain surrogate failure. **Implication:** reducing a correct linear covariance objective can increase exact error. **Next step:** retain exact nonlinear objectives for any future independently motivated CPU method; do not retune this benchmark.

2. **Observation:** at equal probabilities uniform bias is zero, exact MSE0.972347, surrogate46875. **Interpretation:** normalization is bounded while the derivative scales as epsilon^(-1/2). **Implication:** unbiasedness does not certify linearization. **Next step:** use the derived remainder bound and the equality case as explicit checks in future estimator work.

3. **Observation:** median fixed local-allocation regret is 7.261%; only10/96 vectors pass both descriptive tolerances (10% calibration, 1% regret). **Interpretation:** calibration and allocation ranking are separate. Dispersion helps calibration on the deterministic paths, but allocation regret need not decrease monotonically; leverage/direction matter. **Implication:** dispersion alone cannot gate correctness. **Next step:** require a separate protocol and untouched confirmation only if a materially different estimator objective is later justified.

## Mathematical and numerical checks

[Complete diagnostic derivations](../experiments/surrogate_diagnostic/DERIVATIONS.md) give exact covariance under pilot-adaptive allocation and fixed reuse, the analytic Jacobian, the greedy positive-integer surrogate optimum, the bias/variance decomposition, and Taylor remainder bounds. Earlier [CPU derivations](../experiments/cpu_gate/DERIVATIONS.md) and [gradient derivations](../experiments/gradient_gate/DERIVATIONS.md) retain their original scope.

The exact identity MSE=linear+cross+remainder was checked for every new method/vector cell. Linear risk agrees with the analytic covariance formula; raw estimates remain unbiased. All35 positive main allocations were enumerated to check local surrogate optimality and fixed nonlinear optimality. This distinguishes a valid allocator implementation from an invalid finite-budget surrogate approximation.

New run: 645 evaluations, 8.20s enumeration, audit PASS, Taylor residual1.46e-11, covariance/Jacobian risk residual1.46e-11, raw bias1.19e-16; 26 repository tests PASS. The first export failure is preserved in run01; it was a CSV schema mismatch, corrected in run02.

## Integrity and boundaries

No distribution, geometry, direction, prior, or method was chosen after inspecting winners. Negative outcomes and consumed confirmation labels are retained. Existing paired intervals cannot be recycled as fresh evidence for the posthoc diagnosis. Pilot-fraction/dispersion/leverage breakdowns remain available in the source reports; these subgroup plots do not override primary gates.

Some prior gates have same-family provisional fresh-agent review. Pilot-information and convergence diagnoses have local-only review; the unavailable pilot-information reviewer must not be described as independent clearance. This consolidated report and new diagnosis have local mathematical/numerical checks, not fresh independent reviewer clearance.

**GPU cost:0.** No additional allocator variant, model call, rental, or training experiment was launched. The evidence supports stopping this allocator branch. A future direction needs a concrete new objective or information source and its own evaluation contract; no further experiment is prescribed automatically.
