# Pilot-only nonlinear posterior allocator

Standalone CPU prototype for K4/B16/m0=1,2, with frozen policy tables that accept only observed pilot counts. No model or GPU dependencies. Dependencies match experiments/cpu_gate/requirements.txt.

```powershell
python -m unittest discover -s tests -v
python -m experiments.posterior_allocator.run --prepare-only --output refine-logs/posterior-allocator-gate/new-run
python -m experiments.posterior_allocator.run --fit-only --output refine-logs/posterior-allocator-gate/new-run
python -m experiments.posterior_allocator.run --output refine-logs/posterior-allocator-gate/new-run
python -m experiments.posterior_allocator.audit refine-logs/posterior-allocator-gate/new-run
```

Policy use after offline construction:

```python
import json
import numpy as np
from experiments.posterior_allocator.policy import PilotPolicy

policy = PilotPolicy.from_dict(json.load(open('policy_posterior-Jeffreys_m0_2.json')))
counts = policy.allocate(np.array([0, 1, 2, 1]), np.random.default_rng(123))
```

Generate exactly these counts of independent fresh main answers, then combine raw pilot/main means with precommitted lambda=4*m0/16. Do not update counts from main outcomes. The prototype assumes Bernoulli reward statistics and population-variance advantage normalization; it is not integrated with the trainer.

protocol.json commits two priors, integration seeds, sample counts, decision and a new confirmation distribution. Jeffreys is primary; the known synthetic mixture prior is explicitly benchmark-informed sensitivity. Continuous posterior integration is approximate; finite reward-state evaluation is exact. DERIVATIONS.md explains this distinction, exchangeability and unbiased reuse. Results and every negative method are retained; no posterior-method winner is selected after inspecting confirmation.

## Pilot-information gate

The follow-up scripts `profile_information.py`, `information_sanity.py`, `information.py` and `information_report.py` implement the predeclared fixed-main/fixed-lambda versus fixed-total-cost CPU design. See INFORMATION_DERIVATIONS.md and refine-logs/PILOT_INFORMATION_GATE_PLAN.md. Outputs are in refine-logs/pilot-information-gate/20261011_run01. Preserve completed outputs; the runner refuses to overwrite prepared or evaluated runs.

The gate profiles all six settings without accessing fresh confirmation, commits integration counts and exact reward-state engines, saves fresh vectors before fitting, constructs both priors, freezes all policies, evaluates, audits and reports. Conditional normalized-credit risk is exact over finite reward states; continuous posterior integration remains approximate. Independent-scramble disagreement is reported. No GPU dependency or model call is introduced.

Each new policy JSON includes `reuse_lambda` outside the count-policy payload. Combine raw pilot and main means with that precommitted value. The fixed-main arm uses lambda=.5 at main8 and m0=1,2,4; the total24 arm uses lambda=m0/6. `PilotPolicy.allocate` returns positive main counts summing to the saved main budget and accepts no true probabilities or main outcomes.
