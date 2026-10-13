# Normalization amplitude feasibility probe

This is research diagnosis code, not a production estimator or training
integration. Run `python -m experiments.normalization_probe.run` from the
repository root; prior output directories are protected from overwrite.

The sole candidate uses fixed half amplitude. Controls include the unchanged
plug-in, zero, and a true-p scalar oracle. All are evaluated against the same
normalized target using uniform samples at total8/16/32. The protocol freezes
the coefficient and rejection rule before computing the probe's outcomes.
The probability vectors are already consumed, so no new population intervals
are reported. No new benchmark, parameter sweep or winner-based rescue is used.

See [derivation package](../../DERIVATION_PACKAGE.md), the run's raw CSV,
exact bias/variance identities and plots. Tests compare direct count-state
risk, scalar quadratic risk and equality/noiseless boundary behavior.
