# Archived JADA proposal experiments

This folder groups the CPU studies from the previous JADA proposal. Experiment code, tests, frozen protocols, numerical results, figures and historical audit files were moved without changing their contents. Existing negative verdicts remain unchanged.

| Folder | Contents |
|---|---|
| [experiments](experiments/) | Seven implementation packages: estimator gate, nonlinear oracle, posterior allocator, gradient gate, surrogate diagnosis, normalization probe and posterior credit/convergence |
| [tests](tests/) | Regression tests for the archived implementations |
| [refine-logs](refine-logs/) | Original run directories, raw results, figures, plans, reviews and proposal-v0.3 snapshot |
| [idea-stage](idea-stage/) | Research and advantage-target contracts |
| [DERIVATION_PACKAGE.md](DERIVATION_PACKAGE.md) | Previous mathematical derivation package |

Start with the [consolidated evidence report](refine-logs/ESTIMATOR_RESEARCH_REPORT.md) or [previous proposal](../../JADA_GRPO_research_proposal_v0.3.md). The [current sequential-sampling proposal](../../SEQUENTIAL_THOUGHT_SAMPLING_PROPOSAL.md) remains at the repository root; its writing artifacts remain in `../../refine-logs/sequential-proposal/`.

## Running archived checks

Use this directory as the working directory so historical imports and relative protocol paths continue to resolve:

```powershell
Set-Location C:\Thoai\GRPO-MA\archive\jada_proposal
python -m unittest discover -s tests -v
```

Historical module commands such as `python -m experiments.cpu_gate.run` also use this working directory. Existing run outputs must be preserved; choose a new output directory before rerunning an experiment. This relocation does not launch experiments.

## Provenance and links

[RELOCATION_MANIFEST.json](RELOCATION_MANIFEST.json) records every original path, new path, byte count and SHA-256 before the move. [RELOCATION_AUDIT.json](RELOCATION_AUDIT.json) verifies the relocated contents against that manifest and records the test result.

Paths embedded in historical protocols, logs, audit hashes and snapshots were deliberately preserved. Historical repository-relative references to `experiments/`, `refine-logs/`, `idea-stage/` and `DERIVATION_PACKAGE.md` now resolve from this archive directory. Older absolute paths record the original execution environment and are not current launch instructions. Links in root-level proposals, notes and manifest were updated to the archive location. Stored historical proposal hashes therefore describe the original versions, rather than the root documents after their link updates.
