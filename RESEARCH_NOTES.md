# Research notes

## 2026-10-11: JADA-GRPO proposal review

Full report: [JADA_GRPO_REVIEW.md](JADA_GRPO_REVIEW.md). Source proposal v0.2, 2026-09-30. Reviewer `/root/jada_reviewer`, model `gpt-6-astra`, ultra; same-family, provisional. Traces: `.aris/traces/research-review/2026-10-11_run01/`.

- Fixed-weight raw pilot/main thought values are unbiased under fixed policy, independent pilot/main streams, pilot-measurable counts and at least one main sample per thought. The exact reused-value covariance is diagonal with entries `sigma_j² [lambda²/m0 + (1-lambda)² E(1/M_j)]`. A common fixed lambda preserves the allocation-dependent c/M surrogate; do not claim a fatal estimator mismatch or unavoidable adaptive covariance.
- Default K4/m0=2/main8/lambda=.5 exact Bernoulli diagnostics give only 0.73% advantage-MSE improvement on `[.95,.55,.45,.05]` and worse MSE on several ordinary regimes. Variance-only greedy counts are always uniform at this default. This does not pass the proposal's broad Gate 1; do CPU revisions before expensive training.
- Fixed nonzero lambda with fixed pilot size creates a variance floor across growing budgets. Include precommitted budget-dependent weights; do not choose them from current pilot outcomes.
- Equal per-thought outer loss weights alone do not preserve answer normalization or KL semantics. Define weighted answer moments, pilot answer-loss usage, token reduction and global group size.
- Repo K is per GPU rank; theory/allocation must use the same normalization group as training. Repo sample-std/clamp differs from proposal population-variance/sqrt-epsilon.
- GRPO-MA v4 already supplies normalization sensitivity; EPIG-Tree supplies sensitivity-weighted square-root allocation. Novelty is a specific finite-budget normalized-credit procedure and measured benefit, not these foundational derivations.
- Advantage-MSE reduction is not a guarantee of gradient-MSE or online gains. Use token-reduction-matched conditional LoRA gradient comparisons and measured generation/time costs.
- Next gates: CPU estimator sweep -> independent frozen pools -> token-matched gradient calibration -> twelve-run controlled online pilot (native uniform, two-stage uniform, variance-only, JADA x 3 seeds), only if earlier gates pass. No JADA model training was performed for this review.
