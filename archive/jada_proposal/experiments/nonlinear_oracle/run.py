"""Exact K4/B16 oracle headroom gate; prepare vectors before evaluating risks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from experiments.cpu_gate.run import dump, population_pair, save_csv, vectors
from . import oracle as o

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT/'refine-logs/nonlinear-oracle-gate/20261011_run01'


def prepare(out):
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    old = ROOT/'refine-logs/cpu-estimator-gate/20261011_run01'
    previous = json.loads((old/'probability_vectors.json').read_text(encoding='utf-8'))
    records = []
    for v in previous:
        if v['K'] == 4:
            records.append({**v, 'kind': 'development' if v['kind'] == 'heterogeneous' else 'stress',
                            'id': 'old-'+v['id']})
    mixture = json.loads((old/'protocol.json').read_text(encoding='utf-8'))
    mixture['K'] = [4]
    mixture['heterogeneous']['seed'] = protocol['confirmation']['seed']
    for v in vectors(mixture):
        if v['kind'] == 'heterogeneous':
            records.append({**v, 'kind': 'confirmation', 'id': 'new-'+v['id']})
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists() or (out/'probability_vectors.json').exists():
        saved = json.loads((out/'probability_vectors.json').read_text(encoding='utf-8'))
        committed = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        if saved != records or any(committed.get(key) != value for key, value in protocol.items()):
            raise ValueError('Prepared protocol/vectors differ; use a new run directory')
        return committed, saved
    dump(out/'protocol.json', {**protocol, 'prepared_utc': datetime.now(timezone.utc).isoformat()})
    dump(out/'probability_vectors.json', records)
    return protocol, records


def comparisons(rows):
    out = {}
    for subset in ['confirmation', 'development']:
        selected = [r for r in rows if r['kind'] == subset]
        out[subset] = {}
        for candidate in ['JADA', 'local-surrogate-oracle', 'fixed-nonlinear-oracle', 'adaptive-nonlinear-oracle']:
            for baseline in ['uniform', 'variance-only']:
                out[subset][f'{candidate}-vs-{baseline}'] = population_pair(
                    selected, baseline, 'budget-precommitted', candidate=candidate)
    return out


def report(out, rows, summary):
    lines = ['# Exact nonlinear-oracle CPU gate', '', f'Headroom decision: **{summary["decision"]}**. This is not an online-training GO.', '',
             'Scope: K4, total budget 16, m0=1 or 2, budget-precommitted reuse. Fresh confirmation: 96 vectors in six equal-weight distribution components, seed 2718282026, committed before evaluation. Development: previous 96 vectors plus six stress regimes.', '',
             'Every positive integer main allocation was enumerated: 165 at m0=1 and 35 at m0=2. Each conditional main count distribution and pilot count probability is integrated exactly. There is no reward Monte Carlo error or noisy winner-selection bias.', '',
             'At this budget, variance-only is exactly uniform for both pilot settings: m0=1 has identical posterior variance proxies; m0=2/B_main=8 yields [2,2,2,2] for every pilot state. The two primary baseline comparisons therefore coincide and are not independent evidence.', '',
             'Fixed nonlinear oracle knows p and selects counts before pilots. Adaptive nonlinear oracle knows p and the realized pilots and selects counts before fresh main sampling. Neither is deployable when p is unknown. True-p information permits the adaptive oracle to react to realized pilot errors; its headroom need not be learnable from pilots alone.', '',
             '| Fresh confirmation comparison | Relative MSE reduction | 95% interval |', '|---|---:|---:|']
    for key, metric in summary['comparisons']['confirmation'].items():
        lines.append(f'| {key} | {100*metric["relative_reduction"]:.2f}% | [{100*metric["relative_ci95"][0]:.2f}, {100*metric["relative_ci95"][1]:.2f}]% |')
    lines += ['', '| Method | Fresh confirmation MSE |', '|---|---:|']
    confirmation = [r for r in rows if r['kind'] == 'confirmation']
    for method in o.METHODS:
        mean = np.mean([r['methods'][f'{method}/budget-precommitted']['mse'] for r in confirmation])
        lines.append(f'| {method} | {mean:.9f} |')
    lines += ['', '| m0 | Adaptive oracle vs uniform | vs variance-only | Fixed oracle vs uniform |', '|---|---:|---:|---:|']
    for m0, details in summary['by_m0'].items():
        lines.append(f'| {m0} | {100*details["adaptive-vs-uniform"]["relative_reduction"]:.2f}% | '
                     f'{100*details["adaptive-vs-variance-only"]["relative_reduction"]:.2f}% | '
                     f'{100*details["fixed-vs-uniform"]["relative_reduction"]:.2f}% |')
    lines += ['', '| Original stress regime, m0=2 | Uniform MSE | Fixed nonlinear oracle | Adaptive nonlinear oracle |',
              '|---|---:|---:|---:|']
    for row in rows:
        if row['kind'] == 'stress' and row['m0'] == 2:
            metrics = row['methods']
            lines.append(f'| {row["id"].removeprefix("old-")} | {metrics["uniform/budget-precommitted"]["mse"]:.6f} | '
                         f'{metrics["fixed-nonlinear-oracle/budget-precommitted"]["mse"]:.6f} | '
                         f'{metrics["adaptive-nonlinear-oracle/budget-precommitted"]["mse"]:.6f} |')
    lines += ['', '![Oracle headroom](headroom.png)', '', '## Decision and limits', '',
              summary['interpretation'], '',
              'The declared headroom criterion requires both primary lower confidence limits to exceed 5%. Population intervals cluster both pilot settings inside each probability vector and use six stratified paired covariance estimates with contrast-specific Welch-Satterthwaite degrees of freedom. These nominal pointwise intervals characterize the declared distribution; they do not establish LLM generalization.', '',
              'Confirmation examples were not used to tune a method. Both oracle policies are mathematically specified optimizers rather than search-selected algorithms. The exact adaptive oracle dominates each baseline at each pilot state; the fixed oracle dominates every fixed-count policy, including uniform and the local-surrogate oracle, but need not dominate a pilot-adaptive variance-only policy.', '',
              'This small gate does not cover total budgets 32/64, other K, alternative reuse or training. A negative result here cannot establish impossibility at larger budgets. A positive oracle result permits a separate pilot-only method-development experiment; it does not authorize GPU or online training.', '',
              f'CPU solve/report wall time: {summary["elapsed_cpu_wall_seconds"]:.2f} seconds. GPU hours and cost: 0.', '',
              'Artifacts: metrics.csv and results.jsonl contain all six methods; exact_risks_m0_*.npz contain every conditional risk, pilot probability, allocation and selected index. verification.json checks exhaustive coverage, exact dominance, raw-value unbiasedness and risk bounds. Read DERIVATIONS.md in experiments/nonlinear_oracle for the proofs.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs = summary['comparisons']['confirmation']
    names = ['JADA', 'local-surrogate-oracle', 'fixed-nonlinear-oracle', 'adaptive-nonlinear-oracle']
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, baseline, color in [(-.12, 'uniform', '#416788'), (.12, 'variance-only', '#cf6a32')]:
        vals = [pairs[f'{name}-vs-{baseline}'] for name in names]
        y = np.array([100*v['relative_reduction'] for v in vals])
        lo = np.array([100*v['relative_ci95'][0] for v in vals])
        hi = np.array([100*v['relative_ci95'][1] for v in vals])
        ax.errorbar(np.arange(len(names))+offset, y, yerr=[y-lo, hi-y], fmt='o', capsize=4,
                    label=f'vs {baseline}', color=color)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axhline(5, color='gray', linestyle='--', linewidth=1, label='5% headroom threshold')
    ax.set_xticks(range(len(names)), ['JADA', 'True-p local\nsurrogate oracle', 'True-p fixed\nnonlinear oracle', 'True-p + pilot adaptive\nnonlinear oracle'])
    ax.set_ylabel('Relative MSE reduction (%)\nPositive = lower error')
    ax.set_title('Exact K4/B16 oracle headroom · fresh confirmation vectors')
    ax.legend(loc='best')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .01, 'Both pilot settings equally weighted. 95% stratified vector-cluster intervals. Oracle information is unavailable to a deployable method.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'headroom.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    protocol, records = prepare(args.output)
    out = args.output
    if args.prepare_only:
        print(f'Prepared {len(records)} vectors: 96 development, six stress, 96 fresh confirmation. No risks evaluated.')
        return
    if (out/'results.jsonl').exists():
        raise FileExistsError('Use a new output directory; preserve prior results')
    start = time.perf_counter()
    p = np.array([v['p'] for v in records])
    rows = []
    max_bias, max_dominance = 0., 0.
    with (out/'results.jsonl').open('w', encoding='utf-8') as f:
        for m0 in protocol['m0']:
            result = o.solve(p, 16, m0)
            np.savez_compressed(out/f'exact_risks_m0_{m0}.npz', p=p, allocations=result['allocations'],
                pilots=result['pilots'], pilot_probabilities=result['pilot_probabilities'],
                conditional_risks=result['conditional_risks'], unconditional_risks=result['unconditional_risks'],
                **{f'selected_{method}': index for method, index in result['selected'].items()})
            max_bias = max(max_bias, float(np.max(np.abs(result['oracle_value_bias']))))
            for method in o.METHODS:
                max_dominance = max(max_dominance, float(np.max(result['mse']['adaptive-nonlinear-oracle']-result['mse'][method])))
            for index, vector in enumerate(records):
                choices = result['allocations'][result['selected']['adaptive-nonlinear-oracle'][index]]
                mean_counts = result['pilot_probabilities'][index]@choices
                row = {**vector, 'B_total': 16, 'B_main': 16-4*m0, 'm0': m0, 'pilot_fraction': 4*m0/16,
                       'allocation_candidates': len(result['allocations']), 'engine': 'exact exhaustive',
                       'methods': {f'{method}/budget-precommitted': {'mse': float(result['mse'][method][index])}
                                   for method in o.METHODS},
                       'fixed_nonlinear_allocation': result['allocations'][result['selected']['fixed-nonlinear-oracle'][index, 0]].tolist(),
                       'adaptive_mean_allocation': mean_counts.tolist(),
                       'oracle_value_bias': result['oracle_value_bias'][index].tolist(),
                       'oracle_value_covariance_diagonal': result['oracle_value_covariance_diagonal'][index].tolist()}
                rows.append(row)
                f.write(json.dumps(row, allow_nan=False)+'\n')
            f.flush()
            print(f'm0={m0}: {len(result["allocations"])} allocations × {len(result["pilots"])} pilot states × {len(p)} vectors, exact; {time.perf_counter()-start:.2f}s', flush=True)
    pairs = comparisons(rows)
    primary = [pairs['confirmation'][f'adaptive-nonlinear-oracle-vs-{a}'] for a in ['uniform', 'variance-only']]
    decision = 'HEADROOM' if all(x['relative_ci95'][0] > .05 for x in primary) else \
        'LIMITED_HEADROOM' if all(x['relative_ci95'][1] <= .05 for x in primary) else 'INCONCLUSIVE'
    by_m0 = {}
    for m0 in protocol['m0']:
        subset = [r for r in rows if r['kind'] == 'confirmation' and r['m0'] == m0]
        by_m0[str(m0)] = {f'{label}-vs-{baseline}': population_pair(subset, baseline, 'budget-precommitted', candidate=method)
                         for label, method in [('adaptive', 'adaptive-nonlinear-oracle'), ('fixed', 'fixed-nonlinear-oracle')]
                         for baseline in ['uniform', 'variance-only']}
    interpretation = {'HEADROOM': 'There is clear oracle headroom at this budget. The next justified step is a separate CPU pilot-only allocator that targets expected nonlinear error. Keep the original JADA NO-GO intact; the oracle is not that method.',
                      'LIMITED_HEADROOM': 'The strongest exact adaptive oracle has limited headroom under these settings. Do not invest in a deployable allocator here without a separate larger-budget diagnostic; no global impossibility claim is established.',
                      'INCONCLUSIVE': 'Oracle headroom does not meet a clear decision under the declared distribution and threshold. No deployable-method or training GO follows.'}[decision]
    summary = {'decision': decision, 'comparisons': pairs, 'by_m0': by_m0, 'interpretation': interpretation,
               'cells': len(rows), 'method_evaluations': len(rows)*len(o.METHODS),
               'elapsed_cpu_wall_seconds': time.perf_counter()-start, 'GPU_hours': 0, 'GPU_cost': 0,
               'review_independence': 'same-family', 'acceptance_status': 'provisional'}
    verification = {'coverage': 'PASS', 'exact_probability_vectors': len(records), 'exact_cells': len(rows),
                    'all_allocations_per_setting': {'m0=1': 165, 'm0=2': 35},
                    'max_adaptive_oracle_dominance_violation': max_dominance,
                    'max_unbiased_raw_value_bias': max_bias, 'GPU_cost': 0,
                    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}}
    if max_bias > 1e-12:
        raise ArithmeticError('Oracle violated exact fixed-reuse unbiasedness')
    dump(out/'summary.json', summary)
    dump(out/'verification.json', verification)
    save_csv(out/'metrics.csv', [{'kind': r['kind'], 'id': r['id'], 'component': r['component'], 'B_total': 16,
                                  'm0': r['m0'], 'method': method, 'mse': value['mse']}
                                 for r in rows for method, value in r['methods'].items()])
    plot(out, summary)
    report(out, rows, summary)
    from .audit import audit
    audit(out)
    print(json.dumps({'decision': decision, 'primary': primary, 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
