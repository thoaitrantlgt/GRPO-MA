# K4 information-control diagnostic

Target: test full finite-prior Bayes Bellman sampling at K4/B16, minimum coverage m0=1/2, after the one-action prototype failed. Inputs: unchanged 96 consumed heterogeneous vectors and six original stresses; iid Jeffreys prior, same Sobol seeds, sample-std h_tau target and full-sample plug-in. Outputs: complexity audit, frozen quadrature ladder, saved policies, exact reward risks, development intervals, numerical stability and decision report under refine-logs/information-control-k4/20261011_run01.

Success commands: `python -m experiments.sequential_sampling.k4_control --audit-only`; `python -m unittest discover -s tests -v`; `python -m experiments.sequential_sampling.k4_control`.

Base commit: 811bd76e7ee1b998e45199cd084f4c305d129bca. Preserve unrelated uncommitted work and completed trials. No commit requested.

The finite-support optimum is a diagnostic under its own prior, not a benchmark oracle or a deployable method with reliability guarantees. No fresh confirmation, GPU, prior tuning or new benchmark.
