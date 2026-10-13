"""Run the preregistered CPU gate: python -m experiments.cpu_gate.run."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
from scipy.stats import t
from . import estimator as e

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def vectors(protocol):
    rng = np.random.default_rng(protocol['heterogeneous']['seed'])
    out = []
    for k in protocol['K']:
        for name, base in protocol['regimes_K4'].items():
            p = np.array(base)[[0, 3]] if k == 2 else np.repeat(base, k//4)
            out.append({'K': k, 'id': name, 'kind': 'regime', 'component': name, 'p': p.tolist()})
        for component, ab in enumerate([(.5, .5), (2, 2), (8, 8), (.5, 3), (3, .5), None]):
            for i in range(protocol['heterogeneous']['vectors_per_K']//6):
                if ab is None:
                    location = rng.normal()
                    scale = np.exp(rng.uniform(np.log(.15), np.log(3)))
                    p = expit(location+scale*rng.normal(size=k))
                else:
                    p = rng.beta(*ab, size=k)
                out.append({'K': k, 'id': f'hetero-{component}-{i:02d}', 'kind': 'heterogeneous',
                            'component': str(component), 'p': p.tolist()})
    return out


def exact_comparisons(result):
    out = {}
    for reuse in e.REUSE:
        for baseline in ['uniform', 'variance-only', 'oracle']:
            x, y = result[f'{baseline}/{reuse}']['mse'], result[f'JADA/{reuse}']['mse']
            d, r = x-y, 1-y/x if x > 0 else 0.
            out[f'JADA-vs-{baseline}/{reuse}'] = {'difference': float(d), 'difference_se': 0.,
                'difference_ci95': [float(d), float(d)], 'relative_reduction': float(r),
                'relative_se': 0., 'relative_ci95': [float(r), float(r)]}
    return out


def population_pair(rows, baseline, reuse, candidate='JADA'):
    """Stratified vector clusters; all budget cells stay inside each cluster."""
    clusters = {}
    for row in rows:
        key = (row['K'], row['component'], row['id'])
        clusters.setdefault(key, []).append([row['methods'][f'{baseline}/{reuse}']['mse'],
                                             row['methods'][f'{candidate}/{reuse}']['mse']])
    strata = {}
    for (k, component, _), vals in clusters.items():
        strata.setdefault((k, component), []).append(np.mean(vals, axis=0))
    means = np.array([np.mean(vals, axis=0) for vals in strata.values()])
    mean = means.mean(0)
    contributions = [np.cov(vals, rowvar=False, ddof=1)/len(vals)/len(strata)**2
                     for vals in strata.values()]
    x, y = mean
    rel = 1-y/x
    g = np.array([y/x**2, -1/x])
    d = x-y
    def interval(effect, gradient):
        variances = np.array([max(float(gradient@c@gradient), 0.) for c in contributions])
        variance = variances.sum()
        if variance == 0:
            return [float(effect), float(effect)], None
        denom = sum(v*v/(len(vals)-1) for v, vals in zip(variances, strata.values()))
        df = float(variance**2/denom)
        half = t.ppf(.975, df)*np.sqrt(variance)
        return [float(effect-half), float(effect+half)], df
    relative_ci, rdf = interval(rel, g)
    difference_ci, ddf = interval(d, np.array([1., -1.]))
    return {'baseline_mse': float(x), 'candidate_mse': float(y),
            **({'JADA_mse': float(y)} if candidate == 'JADA' else {}), 'relative_reduction': float(rel),
            'relative_ci95': relative_ci, 'difference': float(d), 'difference_ci95': difference_ci,
            'vector_clusters': len(clusters), 'strata': len(strata), 'df_relative': rdf, 'df_difference': ddf}


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bin_summary(rows, field, edges):
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        subset = [r for r in rows if low <= r[field] < high or (r[field] == high == edges[-1])]
        if not subset:
            continue
        clusters = {}
        for row in subset:
            clusters.setdefault((row['K'], row['id']), []).append([
                row['methods']['variance-only/budget-precommitted']['mse'],
                row['methods']['JADA/budget-precommitted']['mse']])
        vals = np.array([np.mean(x, axis=0) for x in clusters.values()])
        x, y = vals.mean(0)
        r = 1-y/x
        influence = y/x**2*(vals[:, 0]-x)-(vals[:, 1]-y)/x
        se = influence.std(ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0.
        critical = t.ppf(.975, max(len(vals)-1, 1))
        out.append({'variable': field, 'low': low, 'high': high, 'cells': len(subset),
                    'vectors': len(vals), 'relative_reduction': float(r),
                    'ci_low': float(r-critical*se), 'ci_high': float(r+critical*se)})
    return out


def chart(path, bins):
    """Standalone SVG of descriptive fixed-bin curves with vector-cluster CIs."""
    fields = ['pilot_fraction', 'dispersion_v', 'max_leverage']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="420" viewBox="0 0 1080 420">',
             '<rect width="1080" height="420" fill="white"/>',
             '<style>text{font:13px sans-serif;fill:#172436}.axis{stroke:#748094}.ci{stroke:#1466a0;stroke-width:2}</style>',
             '<text x="24" y="25">JADA vs variance-only: budget-precommitted reuse, relative MSE reduction (%)</text>']
    for panel, field in enumerate(fields):
        group = [r for r in bins if r['variable'] == field]
        x0, y0, w, h = 55+panel*355, 60, 280, 270
        ymin = min(-5., min(r['ci_low']*100 for r in group)-2)
        ymax = max(5., max(r['ci_high']*100 for r in group)+2)
        yy = lambda val: y0+h-(val-ymin)/(ymax-ymin)*h
        parts.append(f'<text x="{x0}" y="{y0-12}">{field}</text>')
        parts.append(f'<path class="axis" fill="none" d="M{x0},{y0} V{y0+h} H{x0+w}"/>')
        parts.append(f'<path stroke="#aaaaaa" stroke-dasharray="4 4" d="M{x0},{yy(0)} H{x0+w}"/>')
        for val in np.linspace(ymin, ymax, 5):
            parts.append(f'<text x="{x0-38}" y="{yy(val)+4}">{val:.1f}</text>')
        points = []
        for i, row in enumerate(group):
            x = x0+w*(i+.5)/len(group)
            y = yy(row['relative_reduction']*100)
            points.append(f'{x},{y}')
            parts.append(f'<path class="ci" d="M{x},{yy(row["ci_low"]*100)} V{yy(row["ci_high"]*100)}"/>')
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#1466a0"/>')
            parts.append(f'<text x="{x-28}" y="{y0+h+22}">{row["low"]:g}-{row["high"]:g}</text>')
        parts.append(f'<polyline points="{" ".join(points)}" stroke="#1466a0" fill="none"/>')
    parts.append('<text x="24" y="395">Positive = lower JADA error. Descriptive bins; different K and budgets are mixed. 95% pointwise intervals.</text></svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def summarize(out, rows, protocol, elapsed):
    hetero = [r for r in rows if r['kind'] == 'heterogeneous']
    primary = population_pair(hetero, 'variance-only', 'budget-precommitted')
    by_k = {str(k): population_pair([r for r in hetero if r['K'] == k], 'variance-only', 'budget-precommitted')
            for k in protocol['K']}
    comparisons = {f'JADA-vs-{a}/{reuse}': population_pair(hetero, a, reuse)
                   for a in ['uniform', 'variance-only', 'oracle'] for reuse in e.REUSE}
    passed = primary['relative_ci95'][0] > .05 and all(v['relative_reduction'] > 0 for k, v in by_k.items() if int(k) >= 4) \
        and comparisons['JADA-vs-uniform/budget-precommitted']['relative_reduction'] > 0
    bins = []
    for field, edges in protocol['diagnostic_bins'].items():
        bins += bin_summary(hetero, field, edges)
    save_csv(out/'diagnostic_bins.csv', bins)
    chart(out/'improvement.svg', bins)
    summary = {'gate': 'GO' if passed else 'NO-GO', 'primary': primary, 'by_K': by_k,
               'comparisons': comparisons, 'cells': len(rows), 'method_evaluations': len(rows)*16,
               'feasible_configurations': len(e.feasible_cells()), 'elapsed_cpu_wall_seconds': elapsed,
               'GPU_hours': 0, 'GPU_cost': 0, 'review_status': 'same-family provisional'}
    dump(out/'summary.json', summary)
    table = []
    for row in rows:
        for method, metric in row['methods'].items():
            table.append({**{key: row[key] for key in ['K', 'B_total', 'm0', 'id', 'kind', 'component',
                            'engine', 'pilot_fraction', 'dispersion_v', 'max_leverage']},
                          'method': method, **{key: metric[key] for key in ['mse', 'value_mse', 'linear_mse',
                            'surrogate_mse', 'sign_error']}, 'mse_se': metric.get('mse_se', 0.)})
    save_csv(out/'metrics.csv', table)
    paired = []
    for row in rows:
        for key, metric in row['paired'].items():
            paired.append({'K': row['K'], 'B_total': row['B_total'], 'm0': row['m0'], 'id': row['id'],
                           'kind': row['kind'], 'comparison': key, 'relative_reduction': metric['relative_reduction'],
                           'ci_low': metric['relative_ci95'][0], 'ci_high': metric['relative_ci95'][1],
                           'difference': metric['difference'], 'difference_se': metric['difference_se']})
    save_csv(out/'paired_comparisons.csv', paired)
    lines = ['# CPU estimator gate results', '', f'Gate: **{summary["gate"]}**. Bernoulli estimator evidence; zero GPU cost.', '',
             'The primary comparison uses the predeclared equal-weight heterogeneous mixture, equally weighted K and feasible cells within K. Each probability vector is an uncertainty cluster. All 16 methods and negative results are retained.', '',
             f'{len(e.feasible_cells())} feasible settings; {len(rows)} probability-vector/settings cells; {len(rows)*16} method evaluations. CPU wall time: {elapsed:.1f} seconds.', '',
             '| Comparison | Relative MSE reduction | 95% interval |', '|---|---:|---:|']
    for key, v in comparisons.items():
        lines.append(f'| {key} | {100*v["relative_reduction"]:.2f}% | [{100*v["relative_ci95"][0]:.2f}, {100*v["relative_ci95"][1]:.2f}]% |')
    lines += ['', '| K | JADA vs variance-only, precommitted | 95% interval |', '|---|---:|---:|']
    for k, v in by_k.items():
        lines.append(f'| {k} | {100*v["relative_reduction"]:.2f}% | [{100*v["relative_ci95"][0]:.2f}, {100*v["relative_ci95"][1]:.2f}]% |')
    lines += ['', '## Original six regimes at K4/B16/m0=2', '',
              '| Regime | Uniform fixed .5 MSE | JADA fixed .5 MSE | JADA reduction |', '|---|---:|---:|---:|']
    for r in rows:
        if r['kind'] == 'regime' and (r['K'], r['B_total'], r['m0']) == (4, 16, 2):
            x = r['methods']['uniform/fixed-0.5']['mse']
            y = r['methods']['JADA/fixed-0.5']['mse']
            lines.append(f'| {r["id"]} | {x:.9f} | {y:.9f} | {100*(1-y/x):.2f}% |')
    lines += ['', '## Diagnostic curves', '', '![Improvement curves](improvement_curves.png)', '',
              'The curves use predeclared bins, with paired methods and vector-cluster pointwise intervals. These are descriptive associations, not isolated causal effects: K, budget, distribution component and pilot fraction can co-vary. Empty bins are omitted. Details are in diagnostic_bins.csv; the complete crossed table is metrics.csv.', '',
              '## Interpretation and limits', '',
              'The declared gate requires a lower primary confidence limit above 5%, positive point estimates at each K>=4, and positive matched-reuse uniform comparison. This is a new operational threshold; the proposal only says to require consistent clear benefit. K2 has equal analytic leverage coefficients and supplies an equality boundary control, clarified before any new results.', '',
              'Exact calculations cover every K2 cell and the six K4/B16/m0=2 anchors. Larger cells use 20,000 paired trials per named regime and 2,000 per heterogeneous vector. Fixed-vector intervals reflect Monte Carlo uncertainty; the primary interval reflects variation over the stratified vector distribution. It does not quantify LLM/task generalization. Sampling streams are independent with replacement and the policy is fixed.', '',
              'The oracle optimizes the local surrogate and is not a nonlinear-MSE lower bound. linear_mse and surrogate_mse expose approximation failure near equal groups. Nonzero fixed .5 pilot weights retain a variance floor at large budgets. Biased naive pooling may beat an unbiased estimator in MSE.', '',
              'At the symmetric-mid default, analytic norms differ from historical materialized-J arithmetic by 2.92e-5 MSE because three floating-point ties allocate to different coordinates. Other regimes have different shifts (up to about 3.96e-4 among the six anchors). Tests reproduce both arithmetic paths. Reverse-coordinate results are stored as a separate sensitivity, not used to choose the benchmark or the winner.', '',
              'Reproduce: `python -m unittest discover -s tests -v`, then `python -m experiments.cpu_gate.run --output <new-directory>`. Distribution and vectors are saved before sampling. Read DERIVATIONS.md for the exact covariance and surrogate proofs. No frozen-model or online run is authorized by these results alone.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'refine-logs'/'cpu-estimator-gate'/'20261011_run01')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    if (out/'results.jsonl').exists():
        raise FileExistsError('Use a new output directory; historical results are never overwritten')
    all_vectors = vectors(protocol)
    dump(out/'protocol.json', {**protocol, 'prepared_utc': datetime.now(timezone.utc).isoformat()})
    dump(out/'probability_vectors.json', all_vectors)
    dump(out/'feasibility.json', {'feasible': e.feasible_cells(), 'excluded': [
        (k, b, m0) for k in protocol['K'] for b in protocol['total_budgets'] for m0 in protocol['m0']
        if b < k*(m0+1)]})
    if args.prepare_only:
        print(f'Prepared {len(all_vectors)} vectors, {len(e.feasible_cells())} feasible settings before sampling.', flush=True)
        return
    start = time.perf_counter()
    rows = []
    with (out/'results.jsonl').open('w', encoding='utf-8') as f:
        for index, vector in enumerate(all_vectors):
            k = vector['K']
            p = np.array(vector['p'])
            z = p-p.mean()
            v = np.mean(z*z)
            for kk, b, m0 in e.feasible_cells():
                if kk != k:
                    continue
                seed = [protocol['sampling_seed'], k, index, b, m0]
                is_exact = k == 2 or (vector['kind'] == 'regime' and (k, b, m0) == (4, 16, 2))
                if is_exact:
                    methods = e.exact(p, b, m0)
                    paired = exact_comparisons(methods)
                    n = 0
                else:
                    n = protocol['regime_trials'] if vector['kind'] == 'regime' else protocol['heterogeneous_trials_per_vector']
                    methods, paired = e.monte_carlo(p, b, m0, n, seed)
                row = {**vector, 'B_total': b, 'B_main': b-k*m0, 'm0': m0,
                       'engine': 'exact' if is_exact else 'paired-MC', 'trials': n, 'seed': seed,
                       'pilot_fraction': k*m0/b, 'dispersion_v': float(v),
                       'max_leverage': float(np.max(z*z)/(k*v)) if v > 0 else 0.,
                       'methods': methods, 'paired': paired}
                rows.append(row)
                f.write(json.dumps(row, allow_nan=False)+'\n')
                f.flush()
            if (index+1) % 8 == 0:
                print(f'{index+1}/{len(all_vectors)} vectors; {len(rows)} cells; {time.perf_counter()-start:.1f}s', flush=True)
    summary = summarize(out, rows, protocol, time.perf_counter()-start)
    # Sensitivity has identical marginal sampling but different tie ownership.
    sensitivity = []
    for index, vector in enumerate(all_vectors):
        if vector['kind'] != 'regime' or vector['K'] != 4:
            continue
        for b in [16, 32, 64]:
            p = np.array(vector['p'])[::-1]
            methods, paired = e.monte_carlo(p, b, 2, protocol['regime_trials'], [protocol['sampling_seed'], 999, index, b])
            sensitivity.append({'id': vector['id'], 'B_total': b, 'm0': 2, 'coordinate_order': 'reversed',
                                'methods': methods, 'paired': paired})
    dump(out/'tie_sensitivity.json', sensitivity)
    dump(out/'environment.json', {'numpy': np.__version__, 'python_scope': 'CPU only',
                                  'review_status': 'cleared; same-family provisional'})
    from .plot import plot
    from .audit import audit
    plot(out)
    audit(out)
    print(json.dumps({'gate': summary['gate'], 'primary': summary['primary'], 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
