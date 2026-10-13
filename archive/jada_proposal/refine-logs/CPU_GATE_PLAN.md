# CPU estimator gate plan

Source scope: user request on 2026-10-11 and JADA_GRPO_REVIEW.md, Minimum staged experiment package item 1. The selected design is the CPU Bernoulli estimator experiment, not online trainer integration. No GPU budget or model inference.

1. Predeclare probability-vector mixture, six regimes and their K extensions, all feasible budget cells, four allocators, four reuse modes, seeds, trial counts, metrics and intervals in experiments/cpu_gate/protocol.json. Save actual vectors before sampling.
2. Derive fixed-reuse and naive moments and local normalized-advantage risk. Validate analytic J and column norms; check integer feasibility and exhaustive small-case optimum; reproduce legacy default including floating ties.
3. Obtain fresh same-family provisional implementation review, fix blockers, then run exact sanity cases before the full paired CPU sweep.
4. Save all 16 methods, covariance/value and advantage biases, nonlinear/local errors, sign errors, paired differences and relative intervals. Aggregate heterogeneous uncertainty by vector clusters with declared stratification. Retain every negative cell.
5. Report primary and K-specific evidence, diagnostic associations with pilot fraction/dispersion/leverage and reversed-coordinate sensitivity. Judge the predeclared operational gate and preserve the distinction between Bernoulli evidence and frozen-model/training evidence.

Artifacts: experiments/cpu_gate/DERIVATIONS.md; tests/test_cpu_estimator_gate.py; refine-logs/cpu-estimator-gate/20261011_run01/. Results are written to a unique run directory and cannot overwrite prior results. Existing proposal, review and trainer stay intact.
