# CPU estimator implementation review

Date: 2026-10-11. Workflow: experiment-bridge. Reviewer: fresh agent /root/cpu_gate_review, gpt-6-astra, xhigh. Independence: same-family. Acceptance: provisional.

First review found two blockers before sampling:

1. K2 column-norm cancellation altered four pilot allocations at m0=4/B32 despite mathematically equal leverage. Fixed with stable q=.5 epsilon^2/(v+epsilon)^3 and direct shared variance-only allocation for K2. Added exhaustive declared pilot-state/budget regression.
2. Stratified population Student-t df incorrectly summed stratum df despite unequal contrast variances and zero-variance strata. Fixed with contrast-specific Welch-Satterthwaite df, and a degenerate interval for zero variance. Added a regression where only one of two 16-vector strata has variance, giving df=15.

Reviewer independently verified covariance, naive-reuse moments, normalization derivative, Taylor bound, integer allocator optimality, accounting, pairing and aggregation. Independent 40,000-trial check put all 16 methods within 1.55 standard errors of the exact anchor.

Second review: "Provisional clearance for the CPU run; no remaining blocking issues." Both fixes correct; six focused tests passed. Review supports CPU estimation only. Additional permanent paired-MC/exact coverage was subsequently added. No sampling distribution was selected or changed from observed winners.
