"""Check saved coverage/moments and tabulate all configurations; no resampling."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from . import estimator as e
from .run import dump, population_pair, save_csv


def audit(directory):
    rows = [json.loads(line) for line in (directory/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    vectors = json.loads((directory/'probability_vectors.json').read_text(encoding='utf-8'))
    expected = {(v['K'], v['id'], b, m0) for v in vectors for k, b, m0 in e.feasible_cells() if k == v['K']}
    actual = {(r['K'], r['id'], r['B_total'], r['m0']) for r in rows}
    assert actual == expected and len(actual) == len(rows)
    methods = {f'{a}/{reuse}' for a in e.ALLOCATIONS for reuse in e.REUSE}
    assert all(set(row['methods']) == methods and row['B_main'] >= row['K'] for row in rows)
    assert all(np.all(np.isfinite([m['mse'], m['value_mse'], m['linear_mse']]))
               for row in rows for m in row['methods'].values())
    exact_rows = [r for r in rows if r['engine'] == 'exact']
    max_bias = max_cov = max_linear = 0.
    for row in exact_rows:
        for name, metric in row['methods'].items():
            if name.split('/')[1] == 'naive':
                continue
            max_bias = max(max_bias, float(np.max(np.abs(metric['value_bias']))))
            max_cov = max(max_cov, float(np.max(np.abs(np.array(metric['value_covariance'])-
                                                          np.diag(metric['covariance_formula'])))))
            max_linear = max(max_linear, abs(metric['linear_mse']-metric['surrogate_mse']))
    assert max_bias < 1e-12 and max_cov < 1e-12
    for row in rows:
        if row['K'] == 2:
            for reuse in e.REUSE:
                assert row['methods'][f'JADA/{reuse}']['mse'] == row['methods'][f'variance-only/{reuse}']['mse']
    hetero = [r for r in rows if r['kind'] == 'heterogeneous']
    configuration_rows = []
    counts = {'positive': 0, 'negative': 0, 'inconclusive': 0, 'K2-equality': 0}
    for k, b, m0 in e.feasible_cells():
        selected = [r for r in hetero if (r['K'], r['B_total'], r['m0']) == (k, b, m0)]
        pair = population_pair(selected, 'variance-only', 'budget-precommitted')
        label = 'K2-equality' if k == 2 else 'positive' if pair['relative_ci95'][0] > 0 \
            else 'negative' if pair['relative_ci95'][1] < 0 else 'inconclusive'
        counts[label] += 1
        configuration_rows.append({'K': k, 'B_total': b, 'm0': m0, 'pilot_fraction': k*m0/b,
                                   'variance_mse': pair['baseline_mse'], 'JADA_mse': pair['JADA_mse'],
                                   'relative_reduction': pair['relative_reduction'],
                                   'ci_low': pair['relative_ci95'][0], 'ci_high': pair['relative_ci95'][1],
                                   'pointwise_status': label})
    save_csv(directory/'configuration_summary.csv', configuration_rows)
    # All allocator/reuse absolute risks retain exactly the primary weighting.
    risks = []
    for method in sorted(methods):
        cluster = {}
        for r in hetero:
            cluster.setdefault((r['K'], r['id']), []).append([r['methods'][method]['mse'],
                r['methods'][method]['linear_mse'], r['methods'][method]['value_mse']])
        vals = np.array([np.mean(x, axis=0) for x in cluster.values()])
        mean = vals.mean(0)
        risks.append({'method': method, 'nonlinear_mse': mean[0], 'linear_mse': mean[1], 'value_mse': mean[2]})
    save_csv(directory/'aggregate_risks.csv', risks)
    jacobian_error = 0.
    for v in vectors:
        p = np.array(v['p'])
        h = 1e-7
        fd = np.column_stack([(e.advantage(p+h*d)-e.advantage(p-h*d))/(2*h) for d in np.eye(len(p))])
        jacobian_error = max(jacobian_error, float(np.max(np.abs(e.jacobian(p)-fd))))
    sensitivity = json.loads((directory/'tie_sensitivity.json').read_text(encoding='utf-8'))
    sensitivity_rows = []
    for s in sensitivity:
        anchor = next(r for r in rows if r['kind'] == 'regime' and r['id'] == s['id'] and
                      (r['K'], r['B_total'], r['m0']) == (4, s['B_total'], 2))
        key = 'JADA-vs-variance-only/budget-precommitted'
        a, b = anchor['paired'][key], s['paired'][key]
        se = np.hypot(a['relative_se'], b['relative_se'])
        sensitivity_rows.append({'regime': s['id'], 'B_total': s['B_total'],
                                 'original_reduction': a['relative_reduction'],
                                 'reversed_reduction': b['relative_reduction'],
                                 'reversed_minus_original': b['relative_reduction']-a['relative_reduction'],
                                 'difference_ci_low': b['relative_reduction']-a['relative_reduction']-1.96*se,
                                 'difference_ci_high': b['relative_reduction']-a['relative_reduction']+1.96*se})
    save_csv(directory/'tie_sensitivity_summary.csv', sensitivity_rows)
    default_equal = next(r for r in rows if r['id'] == 'all-equal' and (r['K'], r['B_total'], r['m0']) == (4, 16, 2))
    anchor = default_equal['methods']['JADA/fixed-0.5']
    verification = {'coverage': 'PASS', 'cells': len(rows), 'method_evaluations': len(rows)*16,
                    'exact_cells': len(exact_rows), 'paired_MC_cells': len(rows)-len(exact_rows),
                    'paired_MC_trials': sum(r['trials'] for r in rows), 'max_exact_unbiased_value_bias': max_bias,
                    'max_exact_covariance_residual': max_cov, 'max_exact_linear_surrogate_residual': max_linear,
                    'max_analytic_jacobian_FD_error_over_408_vectors': jacobian_error,
                    'K2_allocation_boundary': 'PASS', 'pointwise_configuration_status': counts,
                    'all_equal_default_nonlinear_MSE': anchor['mse'],
                    'all_equal_default_local_MSE': anchor['linear_mse'],
                    'GPU_hours': 0, 'GPU_cost': 0,
                    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
                    'review_independence': 'same-family', 'acceptance_status': 'provisional'}
    dump(directory/'verification.json', verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
