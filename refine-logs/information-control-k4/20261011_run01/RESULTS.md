# K4 information-control results

**Consumed-data development diagnostic. Exact finite-support optimization; continuous-prior quadrature approximate. No confirmation GO or GPU.**

Verdict: **NO_ROBUST_STABLE_DEVELOPMENT_SIGNAL**. Highest-level reduction: 1.4866%, descriptive development interval [1.0203%, 1.9956%].

| Base points / scramble | Support points | m0 | Reduction vs uniform | Development interval | Maximum original stress regression |
|---:|---:|---:|---:|---|---:|
| 16 | 768 | 1 | 1.4053% | [0.7997%, 2.0690%] | 7.6042% |
| 16 | 768 | 2 | 1.4337% | [0.8567%, 2.0552%] | 7.4673% |
| 64 | 3072 | 1 | 1.3817% | [0.7902%, 2.0398%] | 6.0055% |
| 64 | 3072 | 2 | 1.4070% | [0.8518%, 2.0006%] | 6.2723% |
| 256 | 12288 | 1 | 1.4811% | [0.9943%, 2.0107%] | 0.5618% |
| 256 | 12288 | 2 | 1.4921% | [1.0421%, 1.9836%] | 0.7028% |

## Numerical stability

Highest-versus-adjacent aggregate risk change / uniform: 0.0922%. Maximum stress risk change / uniform: 5.5695%. Prior Bayes-value change / prior uniform: 0.6213%. Maximum lower-policy regret on highest finite support / prior uniform: 0.7777%.

All four diagnostics require <=1%. These checks do not certify convergence to the continuous Jeffreys prior; no extra resolution levels were selected after winners.

| Base points | Bellman quotient states | CPU wall seconds | Process peak working set MiB | Prior uniform risk | Prior optimal risk m0=1 | Prior optimal risk m0=2 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 23987 | 3.64 | 114.58 | 0.889686 | 0.835065 | 0.836689 |
| 64 | 23987 | 5.55 | 127.86 | 0.843230 | 0.797507 | 0.799317 |
| 256 | 23987 | 11.82 | 206.22 | 0.831253 | 0.792457 | 0.794153 |

Process peak working set is cumulative within this CPU process, including scientific libraries and earlier levels. It is not an isolated policy-table size.

## Original stress cases at highest resolution

| Regime | m0 | Uniform risk | Control risk | Regression |
|---|---:|---:|---:|---:|
| symmetric-wide | 1 | 0.609175 | 0.612552 | 0.5543% |
| symmetric-mid | 1 | 0.840683 | 0.844823 | 0.4924% |
| all-equal | 1 | 2.91714 | 2.93353 | 0.5618% |
| near-equal | 1 | 4.94989 | 4.95245 | 0.0517% |
| asymmetric | 1 | 0.603491 | 0.532289 | -11.7984% |
| sparse-extreme | 1 | 0.204087 | 0.170642 | -16.3874% |
| symmetric-wide | 2 | 0.609175 | 0.613456 | 0.7028% |
| symmetric-mid | 2 | 0.840683 | 0.84556 | 0.5801% |
| all-equal | 2 | 2.91714 | 2.93353 | 0.5618% |
| near-equal | 2 | 4.94989 | 4.95252 | 0.0531% |
| asymmetric | 2 | 0.603491 | 0.532134 | -11.8241% |
| sparse-extreme | 2 | 0.204087 | 0.175059 | -14.2232% |

## Interpretation

The control optimizes the full adaptive decision tree using only counts and the frozen prior, so it avoids the earlier one-action terminal-count blindness. Its finite-prior optimality does not make it an oracle for the heterogeneous benchmark. Stress performance, integration stability and consumed-data scope must be evaluated separately from average gain.

No prior, terminal estimator, benchmark vector or threshold was selected after outcomes. The former one-action prototype verdict remains unchanged. Reliability gating and real score geometry are absent from this control; it is not a deployable replacement demonstrated to improve training.

[Summary](summary.json) · [Effects](effects.csv) · [Stress table](stress.csv) · [Full metrics](metrics.csv) · [Protocol](protocol.json) · [Checks](AUDIT.json).

[Vietnamese decision](DECISION.md) · [Stability figure](control_stability.png) · [Independent saved-policy reconstruction](SAVED_AUDIT.json) · [Metric declarations clarified after review](PROTOCOL_CLARIFICATION.md). The clarification records previously implicit aggregation and interval semantics without changing frozen policies or numerical outputs.
