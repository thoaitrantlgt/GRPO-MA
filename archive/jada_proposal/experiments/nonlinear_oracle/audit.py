"""Independent checks of saved exhaustive objective tables and selected policies."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from experiments.cpu_gate import estimator as e
from experiments.cpu_gate.run import dump
from .oracle import METHODS


def verify_table(table, p, total, m0):
    counts = table['allocations']
    b = total-4*m0
    # Independent stars-and-bars separator construction.
    expected = set()
    for cuts in itertools.combinations(range(1, b), 3):
        expected.add(tuple(np.diff((0,)+cuts+(b,))))
    if set(map(tuple, counts)) != expected or len(counts) != math.comb(b-1, 3):
        raise ValueError('Incomplete or duplicate allocation space')
    if not np.array_equal(table['p'], p):
        raise ValueError('Saved probability vectors differ from declaration')
    pilots = table['pilots']
    if set(map(tuple, pilots)) != set(itertools.product(range(m0+1), repeat=4)):
        raise ValueError('Incomplete pilot state space')
    weights = table['pilot_probabilities']
    risks = table['conditional_risks']
    if risks.shape != (len(p), len(pilots), len(counts)):
        raise ValueError('Wrong risk table dimensions')
    if not np.all(np.isfinite(risks)) or np.any(risks < 0) or np.any(risks > 4+1e-12):
        raise ValueError('Invalid conditional risks')
    if not np.allclose(weights.sum(1), 1, atol=1e-13, rtol=0):
        raise ValueError('Pilot PMF is not normalized')
    weighted = (weights[:, :, None]*risks).sum(1)
    if not np.allclose(weighted, table['unconditional_risks'], atol=1e-12, rtol=0):
        raise ValueError('Incorrect unconditional risks')
    values = {}
    for method in METHODS:
        choices = table[f'selected_{method}']
        if choices.shape != risks.shape[:2] or np.any(choices < 0) or np.any(choices >= len(counts)):
            raise ValueError('Invalid policy indices')
        conditional = np.take_along_axis(risks, choices[:, :, None], axis=2)[:, :, 0]
        values[method] = (weights*conditional).sum(1)
        if method == 'adaptive-nonlinear-oracle' and not np.allclose(conditional, risks.min(2), atol=1e-12, rtol=0):
            raise ValueError('Selected adaptive oracle is not the global conditional optimum')
        if method == 'fixed-nonlinear-oracle':
            if not np.all(choices == choices[:, :1]) or not np.allclose(values[method], weighted.min(1), atol=1e-12, rtol=0):
                raise ValueError('Selected fixed oracle is not the global unconditional optimum')
    if any(np.any(values['adaptive-nonlinear-oracle'] > x+1e-12) for x in values.values()):
        raise ValueError('Adaptive dominance failure')
    max_direct_error = 0.
    for v in sorted(set([0, min(5, len(p)-1), max(len(p)-96, 0), len(p)-1])):
        for pi in [0, len(pilots)//2, len(pilots)-1]:
            for ci in [0, len(counts)//2, len(counts)-1]:
                m = counts[ci]
                main, prob = e.binomial_states(m, p[v])
                value = e.values(pilots[pi], main, m, m0, total, 'budget-precommitted')
                direct = prob@np.mean((e.advantage(value)-e.advantage(p[v]))**2, axis=1)
                max_direct_error = max(max_direct_error, abs(direct-risks[v, pi, ci]))
    if max_direct_error > 1e-12:
        raise ValueError('Vectorized objective disagrees with direct enumeration')
    return values, float(max_direct_error)


def audit(out):
    records = json.loads((out/'probability_vectors.json').read_text(encoding='utf-8'))
    p = np.array([v['p'] for v in records])
    rows = [json.loads(line) for line in (out/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(v['id'], m0) for v in records for m0 in [1, 2]}
    if {(r['id'], r['m0']) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError('Output coverage mismatch')
    lookup = {(r['id'], r['m0']): r for r in rows}
    max_error = max_direct = 0.
    for m0 in [1, 2]:
        with np.load(out/f'exact_risks_m0_{m0}.npz') as table:
            values, direct_error = verify_table(table, p, 16, m0)
        max_direct = max(max_direct, direct_error)
        for i, v in enumerate(records):
            row = lookup[(v['id'], m0)]
            if row['p'] != v['p'] or len(row['methods']) != len(METHODS):
                raise ValueError('Saved metrics/probability mismatch')
            for method in METHODS:
                error = abs(values[method][i]-row['methods'][f'{method}/budget-precommitted']['mse'])
                max_error = max(max_error, error)
    if max_error > 1e-12:
        raise ValueError('Reported risks differ from saved objective tables')
    confirmation = [v for v in records if v['kind'] == 'confirmation']
    if len(confirmation) != 96 or any(sum(v['component'] == str(c) for v in confirmation) != 16 for c in range(6)):
        raise ValueError('Confirmation distribution was filtered or reweighted')
    result = {'status': 'PASS', 'vectors': len(records), 'exact_cells': len(rows), 'method_evaluations': len(rows)*len(METHODS),
              'global_minima_and_exhaustive_coverage': 'PASS', 'max_saved_metric_residual': max_error,
              'max_direct_enumeration_residual': max_direct, 'GPU_cost': 0}
    dump(out/'AUDIT.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
