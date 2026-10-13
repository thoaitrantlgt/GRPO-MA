# Build note

| Rung | Feature | Acceptance command | Tier | Status |
|---|---|---|---|---|
| F0 | CLI to labelled smoke artifact | python -m experiments.sequential_sampling.run --smoke | MUST | PASS |
| F1 | Target, posterior and uniform-completion exact risk | python -m unittest discover -s tests -v | MUST | PASS |
| F2 | Lookahead acquisition, fallback and budget invariants | python -m unittest discover -s tests -v | MUST | PASS |
| F3 | Real development evaluation and paired MC | python -m experiments.sequential_sampling.run | MUST | PASS |
| F4 | Silent-assumption sweep | fresh same-family reviewer | SHOULD | provisional |

## Run record

F0 smoke command exited 0; artifact explicitly labelled ENTRY_POINT_ONLY. RED: initial test source failed because core module did not exist. F1/F2: initial checks passed; final suite has ten tests PASS. F3: run01 completed (65.91s), then preserved with implementation limitations. Corrected run02 completed (70.05s): four configurations, 102 vectors, five methods, 2040 evaluations; gated risk identical to uniform. Ten final tests PASS. Independent risk reconstruction PASS. Exploratory mechanism/Bayes control ran separately. No live numeric stubs in results. F4: two same-family assumption reviews; round01 gaps declared/corrected, round02 no additional gaps identified, provisional assurance only. Existing uncommitted user work preserved; no commits.


## Deferred

Full 26-configuration confirmation, large-case posterior integration settings and real-model gradient evaluation require successful feasibility and a separately frozen execution protocol. No GPU.

## Blockers

No build blocker. Research feasibility failed: no gated development gain, approximate-prior numerical fallback dominates, and one-action balanced completion cannot value early acquisition. Stop current acquisition before larger sweep.
