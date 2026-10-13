"""Audit frozen posterior tables and exact evaluation without refitting."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from experiments.cpu_gate.run import dump
from experiments.nonlinear_oracle.audit import verify_table
from .fit import PRIORS, prior_points
from .policy import PilotPolicy, make_policy, evaluate


def audit(out):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    records = json.loads((out/'probability_vectors.json').read_text(encoding='utf-8'))
    p = np.array([r['p'] for r in records])
    rows = [json.loads(line) for line in (out/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup = {(r['id'], r['m0']): r for r in rows}
    expected = {(v['id'], m0) for v in records for m0 in [1, 2]}
    if set(lookup) != expected or len(lookup) != len(rows):
        raise ValueError('Coverage or duplication failure')
    policies = {}
    freeze = json.loads((out/'POLICY_FREEZE.json').read_text(encoding='utf-8'))
    if freeze['evaluation_started'] is not False or freeze['methods'] != list(PRIORS):
        raise ValueError('Missing or inconsistent pre-evaluation policy freeze')
    max_policy_residual = max_direct = max_metric = max_bias = 0.
    for prior in PRIORS:
        with np.load(out/f'prior_points_{prior}.npz') as archive:
            for rep in range(protocol['offline_integration']['replicates']):
                points, labels = prior_points(prior, rep, protocol['offline_integration'])
                if not np.array_equal(points, archive['points'][rep]) or not np.array_equal(labels, archive['components'][rep]):
                    raise ValueError('Prior integration samples differ from committed rule')
        policies[prior] = {}
        for m0 in [1, 2]:
            data = json.loads((out/f'policy_{prior}_m0_{m0}.json').read_text(encoding='utf-8'))
            policy = PilotPolicy.from_dict(data)
            if policy.prior != prior or policy.total != 16 or policy.m0 != m0:
                raise ValueError('Policy settings differ from declared scope')
            if data['construction']['fit_completed_utc'] > freeze['frozen_utc']:
                raise ValueError('Policy was constructed after its freeze')
            with np.load(out/f'integration_{prior}_m0_{m0}.npz') as integral:
                reconstructed = make_policy(integral['pilots'], integral['allocations'], integral['numerators'].mean(0),
                                             integral['masses'].mean(0), 16, m0, prior)
            if not np.array_equal(reconstructed.probabilities, policy.probabilities):
                raise ValueError('Policy was not the committed posterior minimum')
            max_policy_residual = max(max_policy_residual, float(np.max(np.abs(reconstructed.posterior_risk-policy.posterior_risk))))
            hmap = {tuple(h): i for i, h in enumerate(policy.pilots)}
            cmap = {tuple(m): i for i, m in enumerate(policy.allocations)}
            for perm in itertools.permutations(range(4)):
                hi = [hmap[tuple(h[list(perm)])] for h in policy.pilots]
                ci = [cmap[tuple(m[list(perm)])] for m in policy.allocations]
                if not np.allclose(policy.probabilities, policy.probabilities[np.ix_(hi, ci)], atol=1e-14, rtol=0):
                    raise ValueError('Permutation equivariance failure')
            policies[prior][m0] = policy
    for m0 in [1, 2]:
        with np.load(out/f'exact_evaluation_m0_{m0}.npz') as archive:
            grid = dict(archive)
        risks, direct = verify_table(grid, p, 16, m0)
        max_direct = max(max_direct, direct)
        for prior in PRIORS:
            risks[prior] = evaluate(grid, policies[prior][m0])
            if np.any(risks[prior] < risks['adaptive-nonlinear-oracle']-1e-12):
                raise ValueError('Pilot-only method beat the full-information optimum')
        max_bias = max(max_bias, float(np.max(np.abs((4*m0/16)*(grid['pilot_probabilities']@(grid['pilots']/m0)-p)))))
        for i, vector in enumerate(records):
            row = lookup[(vector['id'], m0)]
            if row['p'] != vector['p'] or len(row['methods']) != 8:
                raise ValueError('Wrong evaluation vector or missing method')
            for name, risk in risks.items():
                max_metric = max(max_metric, abs(risk[i]-row['methods'][f'{name}/budget-precommitted']['mse']))
    if max_policy_residual > 1e-12 or max_metric > 1e-12 or max_bias > 1e-12:
        raise ValueError('Saved quantities are inconsistent')
    confirmation = [r for r in records if r['kind'] == 'confirmation']
    if len(confirmation) != 96 or any(sum(v['component'] == str(c) for v in confirmation) != 16 for c in range(6)):
        raise ValueError('Fresh confirmation was filtered')
    result = {'status': 'PASS', 'vectors': len(records), 'exact_cells': len(rows), 'method_evaluations': len(rows)*8,
              'policy_prior_rule_and_freeze': 'PASS', 'posterior_minimum_and_permutation_equivariance': 'PASS',
              'max_policy_risk_reconstruction_residual': max_policy_residual,
              'max_saved_metric_residual': max_metric, 'max_direct_enumeration_residual': max_direct,
              'max_unbiased_raw_value_bias': max_bias, 'GPU_cost': 0}
    dump(out/'AUDIT.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
