# CPU estimator gate

This standalone experiment uses Bernoulli reward streams and never imports the training/model stack. Dependencies are numpy, scipy and matplotlib; no GPU is used.

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m experiments.cpu_gate.run --prepare-only --output refine-logs/cpu-estimator-gate/new-run
python -m experiments.cpu_gate.run --output refine-logs/cpu-estimator-gate/new-run
python -m experiments.cpu_gate.plot refine-logs/cpu-estimator-gate/new-run
python -m experiments.cpu_gate.audit refine-logs/cpu-estimator-gate/new-run
```

protocol.json declares the benchmark and intervals. The run saves probability_vectors.json before any reward sampling. Sampling seed, trial counts and all per-method outcomes are retained in results.jsonl. A new output directory is required to preserve historical results; output is deterministic for the saved protocol and environment.

Read DERIVATIONS.md for exact covariance, nonlinear limits and allocator proofs. Read RESULTS.md, configuration_summary.csv and aggregate_risks.csv in the run directory for results. All methods use the same epsilon, prior, total budget and raw final pilot estimator; oracle coefficients are true local-risk coefficients, not a nonlinear-MSE bound. Uniform/precommitted is also the ordinary full-total-budget pooled uniform estimator because every declared budget is divisible by K.

Population comparisons average feasible cells within each vector, then equal mixture components and K; vector clusters preserve paired methods. The 26-cell configuration intervals and diagnostic bins are pointwise exploratory intervals, not simultaneous winner-selection evidence. The primary gate and fixed-vector paired intervals are distinguished in protocol.json.
