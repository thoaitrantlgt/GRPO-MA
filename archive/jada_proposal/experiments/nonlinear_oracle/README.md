# Nonlinear-oracle CPU gate

This is a small, exact K4/B16 headroom diagnostic following the failed CPU JADA gate. It enumerates all positive integer allocations and integrates every pilot/main Bernoulli count state. Reuse is budget-precommitted, m0 is 1 or 2, and no GPU/model calls are used.

```powershell
python -m unittest discover -s tests -v
python -m experiments.nonlinear_oracle.run --prepare-only --output refine-logs/nonlinear-oracle-gate/new-run
python -m experiments.nonlinear_oracle.run --output refine-logs/nonlinear-oracle-gate/new-run
python -m experiments.nonlinear_oracle.audit refine-logs/nonlinear-oracle-gate/new-run
```

Dependencies are those in experiments/cpu_gate/requirements.txt. The original CPU run's saved probability vectors and protocol supply development examples and the already declared mixture. Fresh confirmation vectors use a new committed seed. Preparation is idempotent only when protocol and vectors match exactly; existing results cannot be overwritten.

The fixed nonlinear oracle knows true p before pilots. The adaptive nonlinear oracle additionally observes current pilots. They are distinct information controls; neither is an executable pilot-only method when p is unknown. A positive result shows allocation headroom in this small scope and justifies CPU method development only. It does not reverse the previous JADA NO-GO.

See protocol.json, DERIVATIONS.md and the run's RESULTS.md. exact_risks_m0_1.npz and exact_risks_m0_2.npz expose the entire finite objective, probability weights, count candidates and chosen policies. audit.py rechecks counts using independent separator enumeration, weighted risks, exact minima, direct-enumeration samples and prior anchors. No benchmark distributions are chosen after seeing winners.
