# Sequential-sampling CPU feasibility specification

Target: implement and exercise Sections 4–7.1 of SEQUENTIAL_THOUGHT_SAMPLING_PROPOSAL.md. Inputs: consumed 96 heterogeneous vectors and six stress vectors from the archived surrogate diagnostic; Bernoulli observations. Outputs: frozen protocol, real risk tables, runtime/acquisition diagnostics and a development-only report under refine-logs/sequential-feasibility/20261011_run02 (corrected; run01 preserved).

Success commands: `python -m unittest discover -s tests -v`; `python -m experiments.sequential_sampling.run`.

Base commit: 811bd76e7ee1b998e45199cd084f4c305d129bca. Existing uncommitted work and the archive relocation are preserved; no commits requested.

Scope: K=2/4, B=16, m0=1/2. Exact outcome-risk propagation where feasible; paired Monte Carlo as an independent check. This is feasibility/development, not the 26-configuration fresh confirmation campaign or GPU training.
