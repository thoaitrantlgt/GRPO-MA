# Metric clarification after assumption review

This records the implemented meaning explicitly; frozen protocol, policies and result tables are unchanged.

The primary reduction is `1 - mean(control MSE) / mean(uniform MSE)`, with equal weights across m0=1/2 and across the six equally represented consumed components. It is not the mean of per-vector relative reductions. Stress vectors are excluded from this mean and checked separately. This metric matches the main sequential proposal but was not fully restated in the local protocol/ledger before the run.

Development95% intervals are two-sided percentile intervals from2,000 component-stratified bootstrap resamples. Each vector resample stays paired across m0 and quadrature levels; NumPy's default linear interpolation computes the2.5th and97.5th quantiles. These intervals describe variation in already-consumed vectors. Prior integration uncertainty is assessed separately and is not covered by these intervals.

The assumption ledger records these declarations with `Source: sweep`. No result was reweighted, method selected or numerical ladder extended in response to the review.

Aggregate risk stability is the absolute highest-versus-adjacent difference of the equal-m0 pooled mean, normalized by the uniform consumed mean. It describes stability of that mixture; it does not impose a separate aggregate-change criterion on each m0. The current per-m0 differences have the same sign. Stress stability and stress regression use the maximum over both m0 and all six regimes. The screening lower bound is the2.5th percentile endpoint of the two-sided95% interval.
