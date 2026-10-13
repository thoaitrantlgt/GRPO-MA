# Sequential sampling: CPU feasibility

Implements the current [proposal](../../SEQUENTIAL_THOUGHT_SAMPLING_PROPOSAL.md)'s one-observation lookahead, shared numerical stability check, posterior reliability threshold 0.90 and balanced fallback. The primary terminal estimator is the same all-observation sample-std plug-in for every method.

**Current decision: stop the one-action/balanced-completion prototype before a full sweep.** Gated allocation is exactly uniform in the tested histories. This is not an impossibility result for sequential sampling.

Run checks from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The completed run is [20261011_run02](../../refine-logs/sequential-feasibility/20261011_run02/RESULTS.md). To rerun with a new output directory, preserving existing results:

```powershell
python -m experiments.sequential_sampling.run --output refine-logs/sequential-feasibility/new_run
```

The source [protocol](protocol.json) freezes the consumed-development dataset, four configurations, posterior approximation, tie handling, comparators and MC size. The saved run contains its own code/protocol snapshots. Old proposal experiments remain in `archive/jada_proposal/`.

The [mechanism diagnostic](../../refine-logs/sequential-feasibility/20261011_run02/mechanism-diagnostic/RESULTS.md) proves why one-action lookahead cannot value early information while balanced completion preserves terminal counts. Its small-case Bayes control is exploratory and uses a finite prior, rather than true probabilities. No training or GPU.

The subsequent [K4 full-planning control](../../refine-logs/information-control-k4/20261011_run01/DECISION.md) solves a permutation quotient with frozen finite-prior support and evaluates exact outcome risks. Three predeclared integration levels completed. Its 1.49% development gain and failed stress stability check do not justify a full sweep. Source: `k4_control.py`, `symmetry_control.py`; declarations: `k4_control_protocol.json` and `implement-stage/information_control_k4/ASSUMPTIONS.md`. The control entry point preserves the completed run by refusing an existing output directory.

Audit saved outputs or regenerate the figure without another policy search:

```powershell
python -m experiments.sequential_sampling.k4_audit
python -m experiments.sequential_sampling.k4_report
```
