"""Development-only runtime profile, never inspecting fresh confirmation."""
import time
from pathlib import Path
import numpy as np
from experiments.nonlinear_oracle.oracle import conditional_risk_grid
from experiments.cpu_gate.run import dump

SETTINGS = [{'id': f'fixed-main-m0-{m}', 'arm': 'fixed-main', 'total': 8+4*m, 'm0': m, 'lambda': .5} for m in [1,2,4]] + [
    {'id': f'fixed-total-m0-{m}', 'arm': 'fixed-total', 'total': 24, 'm0': m, 'lambda': m/6} for m in [1,2,4]]


def main():
    points = np.random.default_rng(123456).beta(.5,.5,(32,4))
    records = []
    for s in SETTINGS:
        start = time.perf_counter()
        g = conditional_risk_grid(points, s['total'], s['m0'], s['lambda'])
        records.append({**s, 'profile_points': len(points), 'seconds': time.perf_counter()-start,
                        'allocations': len(g['allocations']), 'pilot_histories': len(g['pilots'])})
        print(records[-1], flush=True)
    perpoint = sum(r['seconds'] for r in records)/len(points)
    n = 256
    while n > 32 and perpoint*4*(n+6*(n//4)) > 900:
        n //= 2
    config = {'seed': 5772152026, 'replicates': 4, 'Jeffreys_points_per_replicate': n,
              'mixture_points_per_component_per_replicate': n//4}
    out = Path('refine-logs/pilot-information-gate/20261011_run01')
    out.mkdir(parents=True, exist_ok=True)
    dump(out/'FEASIBILITY.json', {'settings': records, 'offline_integration': config,
         'rule': 'halve power-of-two counts from Jeffreys256/mixture64 until estimated fit <=900s, floor32/8; runtime only',
         'engine': 'exact reward-state risk for every setting', 'confirmation_accessed': False,
         'estimated_fit_seconds': perpoint*4*(n+6*(n//4))})
    print(config, flush=True)


if __name__ == '__main__':
    main()
