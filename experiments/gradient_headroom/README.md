# Gradient-aware CPU headroom diagnostic

Fixed-count true-p oracles isolate the update-risk objective before building an unknown-p acquisition. Score vectors are synthetic and frozen. The full nonlinear credit-error second moment is contracted with the score Gram matrix, including off-diagonal terms. All reward distributions are summed exactly.

[Derivation](../../refine-logs/gradient-headroom/DERIVATION_PACKAGE.md) · [Plan](../../refine-logs/gradient-headroom/EXPERIMENT_PLAN.md) · [Protocol](protocol.json) · [Completed decision](../../refine-logs/gradient-headroom/20261011_run02/DECISION.md).

Completed run: K4/B16/m0=1/2, all96 consumed heterogeneous vectors plus six original stresses, four predeclared geometries and24 coordinate orders. Oracle improvement over uniform16.12%; additional advantage over credit oracle4.39%, below the declared screen. No deployable-method, generalization or training success claimed. GPU cost0.

Run tests or reconstruct saved outputs:

```powershell
python -m unittest discover -s tests -v
python -m experiments.gradient_headroom.report
```

To reproduce in a fresh folder while preserving previous results:

```powershell
python -m experiments.gradient_headroom.run --output refine-logs/gradient-headroom/new_run
python -m experiments.gradient_headroom.report --output refine-logs/gradient-headroom/new_run
```

`m0` is a minimum count for allocation committed before rewards. The displayed coverage fraction does not test adaptive pilot learning. Run01 is preserved as incomplete after audit JSON serialization failed; run02 corrects the writer without changing the protocol or numerical objective.
