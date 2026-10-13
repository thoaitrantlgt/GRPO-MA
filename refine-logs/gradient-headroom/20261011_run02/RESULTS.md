# Gradient-aware CPU headroom results

**Consumed-data diagnostic; synthetic scores; oracle knows true p. No confirmation or training GO.**

Verdict: **NO_PREDECLARED_GRADIENT_HEADROOM_SIGNAL**.

| Geometry | Headroom vs uniform | Advantage vs credit oracle | Development interval, advantage |
|---|---:|---:|---|
| orthogonal | 12.276% | 0.000% | [0.000%, 0.000%] |
| opposed | 15.763% | 3.975% | [2.902%, 5.103%] |
| unequal-norm | 20.451% | 9.318% | [8.591%, 10.001%] |

Pooled headroom: 16.123%, development interval [14.15253928 18.38221573]. Pooled advantage over credit oracle: 4.386%, interval [3.90515307 4.84118566].

Aligned-null has exactly zero update risk by centering; relative endpoints are NA and excluded by protocol. Orthogonal geometry has identical update and credit objectives. m0 constrains precommitted allocations and is not an adaptive pilot experiment.

Exact lattice: 455 allocations at m0=1; 165 at m0=2. All102 vectors ×24 orders ×4 geometries ×2 coverage conditions. CPU wall seconds: 17.27. Audit: PASS.

[Protocol](protocol.json) · [Raw risk lattice](risk_lattice.npz) · [Selected moments and counts](selected_results.npz) · [Metrics](metrics.csv) · [Stress](stress.csv) · [Dispersion](dispersion.csv) · [Leverage](leverage.csv) · [Audit](AUDIT.json).

[Vietnamese decision](DECISION.md) · [Figure](gradient_headroom.png) · [Independent saved-artifact and bootstrap reconstruction](SAVED_AUDIT.json).
