# Pilot-information gate completed

**NO_CLEAR_SIGNAL**. CPU only; GPU cost 0.

| Setting | Primary reduction vs uniform | CI95 |
|---|---:|---:|
| fixed-main-m0-1 | 0.351% | [0.300%, 0.401%] |
| fixed-main-m0-2 | 0.017% | [0.005%, 0.028%] |
| fixed-main-m0-4 | -0.066% | [-0.110%, -0.022%] |
| fixed-total-m0-1 | 0.000% | [-0.000%, 0.000%] |
| fixed-total-m0-2 | -0.115% | [-0.153%, -0.076%] |
| fixed-total-m0-4 | 0.083% | [0.030%, 0.136%] |

Fixed-main holds main8 and lambda=.5 with total costs12/16/24. Fixed-total holds total24 with main20/16/8 and budget-precommitted reuse. Primary information gate requires lower95 limit>5% at fixed-main/m0=4; practical equal-cost gate adds >5% at fixed-total/m0=4 and positive points at all matched-total settings.

Jeffreys primary and benchmark-informed mixture secondary were frozen before evaluation; secondary results cannot rescue primary failure. Unchanged six-component mixture, fresh96 vectors seed1618032026 plus six stress regimes. All methods, settings and regimes retained; this confirmation is now consumed.

19 tests passed before deployment; end-to-end finite-prior sanity and saved-result audit pass. Exact reward-state evaluation covers 612 cells / 3,672 method evaluations. Prior integration is approximate and independent-scramble stability is reported separately. Confidence intervals are conditional on frozen policies.

Implementation review is local-only: the requested fresh reviewer was unavailable due to a usage limit. Do not describe this as independent review clearance. No training or LLM claim is supported.

Neither predeclared gate passes. The primary fixed-main/m0=4 outcome worsens MSE by 0.066%; matched total24/m0=4 improves it only 0.083%. The benchmark-informed secondary does not rescue the gate. Increasing pilots alone did not recover material benefit under these frozen policies. This is not a general impossibility theorem, and primary matched-cost quadrature instability remains a limitation.

[Full results](pilot-information-gate/20261011_run01/RESULTS.md) · [Figure](pilot-information-gate/20261011_run01/information_gate.png) · [Derivations](../experiments/posterior_allocator/INFORMATION_DERIVATIONS.md)
