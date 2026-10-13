# Build note: K4 control

| Rung | Feature | Acceptance command | Tier | Status |
|---|---|---|---|---|
| F0 | Real complexity audit CLI | python -m experiments.sequential_sampling.k4_control --audit-only | MUST | PASS |
| F1 | Symmetry quotient and exact finite-prior Bellman control | python -m unittest discover -s tests -v | MUST | PASS |
| F2 | Outcome graph evaluation and saved-policy reconstruction | python -m unittest discover -s tests -v | MUST | PASS |
| F3 | Frozen three-level K4 study and stability report | python -m experiments.sequential_sampling.k4_control | MUST | PASS (execution; research screen failed) |
| F4 | Same-family assumption review | fresh/reused reviewer per research-implement-feature Phase4 | SHOULD | round 2 completed; provisional |

## Run record

F0 real complexity audit PASS. RED: new quotient tests failed because symmetry_control did not exist. F1/F2:15 tests PASS, including equality with full Bellman at small budget and equality of graph risk with labelled path propagation. Three-level study started with protocol/code snapshots before solving or evaluating policies.


## Deferred

New deployable acquisition, fresh confirmation, K8/K16, gradient/real-model evaluation and GPU training.

## Blockers

No implementation blockers. Practical gain and stress integration stability did not meet the frozen research screen; no escalation.

## Completed study

All three declared levels completed: 612 evaluations, 23,987 canonical Bellman states per level, 22.51 CPU wall seconds, cumulative process peak working set 206.22 MiB. Highest-level pooled reduction 1.4866%, development interval [1.0203%,1.9956%]; maximum stress regression 0.7028%. Stress numerical change 5.5695% fails the declared 1% check. Fifteen implementation tests PASS. Saved-table direct-error reconstruction, independent posterior likelihood Bellman checks, exact budgets and code hashes PASS in SAVED_AUDIT.json. The audit's NumPy boolean JSON serialization was corrected in the separate audit writer; frozen solver/evaluator hashes remain unchanged.

Round 1 same-family review found undeclared aggregation, pooling and interval details. K15-K17 and PROTOCOL_CLARIFICATION.md now record them explicitly, with post-review provenance. Frozen metric values and policies were not changed. Round 2 found no additional material gaps or contradictions and reproduced summary statistics exactly; semantic assurance remains same-family provisional, with no correctness or method-success certification.
