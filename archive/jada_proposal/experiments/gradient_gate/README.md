# Fixed synthetic score-Gram CPU gate

Reuses the Bernoulli count-state engine, advantage normalization and baseline allocators. No torch, model calls or GPU dependencies. Uses the existing NumPy/SciPy/Matplotlib environment.

```powershell
python -m unittest discover -s tests -v
python -m experiments.gradient_gate.run --prepare-only
python -m experiments.gradient_gate.run --sanity-only
python -m experiments.gradient_gate.run
python -m experiments.gradient_gate.audit
```

The unique run directory is refine-logs/gradient-objective-gate/20261011_run01. Preserve existing output; do not rerun over evaluated data. protocol.json declares all fixed Grams, methods, new confirmation seed, exact engine and decision before outcomes. Read DERIVATIONS.md for the fixed-score assumptions and null geometry.

This first gate measures information headroom, including a known-G pilot plug-in diagnostic. No posterior gradient allocator, pilot Gram estimator or model integration is implemented. A positive oracle result is not a deployable-method or training GO.
