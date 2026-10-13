# Advantage target and update contract completed

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** The immediate target is the
trainer-aligned conditional thought value, transformed by an explicitly
declared sample-standard-deviation floor. The primary research loss is the
error of the thought reward-surrogate ascent gradient, with the same token
masks and token/thought reduction as the baseline.

Existing CPU gates used population variance with additive epsilon; they
remain valid for that target and cannot be renamed trainer-exact. Thought
and answer normalization are distinct, and the answer/KL branches must be
held fixed when isolating thought-credit estimation.

At ratio1, fixed-score update error is exactly E[delta^T S^T S delta].
Away from ratio1, clipping branches depend on the advantage sign, requiring
direct clipped-gradient comparison. Constant thought scaling alters its
weight relative to unchanged answer and KL gradients; it is not universally
equivalent to scaling the full optimizer learning rate.

A future estimator must beat both uniform and a development-frozen scalar
calibration control on a separately declared confirmation, with lower95
update-MSE reductions above5% and no named exact stress regression above10%.
These are proposed future criteria, not new criteria for completed gates.
Zero-target cases use absolute spurious-update energy; relative direction
metrics are undefined. Runtime information, sample cost, grouping, std-floor
and clipping behavior must be fixed before evaluation.

No concrete new normalization has been selected. Full expected-reward
gradient claims require additional treatment of baseline dependence,
length weighting and reward-score dependence. Existing NO-GO decisions
remain unchanged. No benchmark, model, trainer change, GPU or new confirmation.

[Full target contract](../../../idea-stage/docs/ADVANTAGE_TARGET_CONTRACT.md) ·
[Updated derivation](../../../DERIVATION_PACKAGE.md) ·
[Local algebra audit](AUDIT.json)

Local verification: 11 checks PASS; finite-difference residuals 3.59e-11 at ratio1 and 7.28e-12 under clipping. CPU PyTorch 2.12.1+cpu; pinned training environment not executed. Local-only review. GPU cost0.
