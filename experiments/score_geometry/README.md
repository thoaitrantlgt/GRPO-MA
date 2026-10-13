# Target score geometry: server preparation

The current deliverable is a [server run plan](../../refine-logs/score-geometry/SERVER_RUN_PLAN.md), with [protocol](../../refine-logs/score-geometry/server_protocol.json) and [tracker](../../refine-logs/score-geometry/EXPERIMENT_TRACKER.md).

Historical raw data is checked in with ordinary Git. One artifact exceeds GitHub's blob limit and is stored as checksum-verified parts; run `python -m experiments.restore_artifacts` from repository root to reconstruct it when needed. The preparation CLIs do not require that historical raw lattice. No Git LFS installation is required.

Implemented: `preflight.py` checks supplied checkpoint/package/data paths without loading weights; `prepare_manifest.py` freezes128 prompts independent of scores/rewards and verifies image paths. Use commands in the plan from repository root.

Not implemented: target replay capture, thought-score gradient exporter and Gram analyzer. These are prerequisites for the planned four-prompt smoke and128-prompt main measurement. No measurement command is presented as runnable before those pieces exist. The local machine lacks the target checkpoint/replay/data; smaller cached models are not substituted. No training or GPU job launched.
