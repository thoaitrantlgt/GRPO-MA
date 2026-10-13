# Next CPU gate: pilot information versus allocation headroom

Status: proposed next experiment, not run. Motivated by post hoc diagnostics of consumed confirmation; no fresh evidence claimed. Original JADA NO-GO and posterior NO_CLEAR_SIGNAL remain unchanged.

Hypothesis: additional pilot observations recover material nonlinear-allocation gains when main sample capacity and reuse weight are held fixed. Alternative explanations include an objective that favors uniform pooling and limited gains from pilot-measurable decisions even under a correct population prior.

## Design to freeze before outcomes

- K=4, Bernoulli rewards, epsilon=1e-6; unchanged equally weighted six-component probability-vector distribution. Keep all six named stress regimes as descriptive cases.
- Diagnostic arm: main budget 8, m0={1,2,4}, total budgets {12,16,24}; lambda=.5 fixed before pilots. Raw estimates remain unbiased. This arm isolates pilot size while spending more total samples; it is not an equal-cost improvement claim.
- Matched total-cost arm: total budget 24, m0={1,2,4}, main budgets {20,16,8}; lambda=4*m0/24. Report this separately from the diagnostic arm. Exhaustive risks may become large; estimate reward risks by paired Monte Carlo with intervals if exact enumeration is impractical, after analytic anchors pass.
- Controls: uniform, variance-only, existing JADA and full-information nonlinear oracle. Primary posterior Jeffreys and secondary benchmark-informed mixture stay fixed; no method/prior winner selection. Oracle comparisons remain unavailable-information diagnostics.
- Development uses consumed vectors only. Fresh confirmation: seed 1618032026, 96 vectors, 16 per original mixture component, saved before fitting/evaluation. Never filter vectors or change distribution using diagnostic subgroup winners.
- Freeze quadrature size, replicate seeds, tie rule and exact/MC engine per setting after a development-only runtime feasibility check and before confirmation. Current settings must not be silently reused as an exact Bayes optimality certificate.
- Primary hypothesis contrast: Jeffreys-versus-uniform reduction at m0=4 in the fixed-main/fixed-lambda arm, with lower 95% stratified vector-cluster limit >5%. Report paired m0=4 minus m0=1 gain as the information trend. A diagnostic-arm pass alone does not establish equal-total-cost improvement.
- Equal-cost practical signal additionally requires lower 95% Jeffreys reduction limit >5% for the total24/m0=4 setting and positive point estimates across all matched-total settings. Secondary prior cannot rescue the primary. Conditional frozen-policy population intervals and prior integration variability must both be reported.
- Plot all settings against pilot fraction, dispersion and leverage. Subgroup summaries are descriptive; declare bins before confirmation, preserve all bins and methods.
- Sanity and fresh implementation review before full evaluation. CPU only, GPU cost zero. No training GO from this gate.

## Required implementation work

Generalize the exact conditional-risk engine to accept a separately precommitted main budget and lambda; retain existing outputs as regression anchors. Verify covariance and unbiasedness for both arms, enforce positive integer main allocations, lock pilot-only runtime access, and verify paired Monte Carlo against exact small cases. Profile allocation/state growth on development before choosing the evaluation engine.

The new confirmation seed is reserved by this plan and has not been generated or examined. A changed design must be documented before confirmation is accessed. This plan does not retroactively preregister the preceding exploratory diagnosis.
