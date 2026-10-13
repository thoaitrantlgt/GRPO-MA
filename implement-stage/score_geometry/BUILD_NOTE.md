# Score geometry build note

| Rung | Capability | Check | Status |
|---|---|---|---|
| F0 | Availability artifact from real environment | python -m experiments.score_geometry.preflight --help | CLI ready; server inputs pending |
| F1 | Trainer-aligned exact score export | gradient/parameter/mask tests | NOT IMPLEMENTED; user steered to server plan |
| F2 | Deterministic prompt selection and Gram metrics | prepare_manifest CLI / geometry tests | selection CLI ready; analyzer NOT IMPLEMENTED |
| F3 | Real target audit | run on supplied checkpoint/replay | BLOCKED: no target artifacts found |
| F4 | Assumption sweep | same-family review | deferred; implementation workflow interrupted by request for server plan |

No simulated vectors will enter a real-data result directory. F3 remains incomplete until target inputs exist.

Current deliverable is SERVER_RUN_PLAN.md with runnable preflight/prompt-freeze commands and explicit instrumentation prerequisites. No measurement success, real target geometry, independent-family review or GPU job claimed.
