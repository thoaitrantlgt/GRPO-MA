# Initial implementation run: superseded

This run is retained as historical evidence. Use [run02](../20261011_run02/RESULTS.md) for the corrected feasibility result.

Run01 had identity-conditioned ties instead of the proposal's randomized tie evaluation, floating-point cancellation in the locally zero K2 Jacobian, unbalanced all-zero surrogate ties, and z=0 labels for some zero-empirical-SE Monte Carlo discrepancies. Its recorded numerical data and original code snapshot remain unchanged. The fixed-pilot comparators optimize a main-count inverse-count heuristic despite sharing an all-observation terminal estimator; this limitation is explicitly declared in the corrected implementation ledger.

These findings do not create a positive gated effect: both runs have gated risk identical to uniform. No completed result was overwritten, and no prior or posterior-reliability threshold was tuned.
