"""Fit pilot-only posterior allocations and evaluate on a fresh CPU holdout."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from experiments.cpu_gate.run import dump, population_pair, save_csv, vectors
from experiments.nonlinear_oracle import oracle as o
from .fit import PRIORS, fit_prior
from .policy import PilotPolicy, evaluate

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT/'refine-logs/posterior-allocator-gate/20261011_run01'


def prepare(out):
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    old = ROOT/'refine-logs/nonlinear-oracle-gate/20261011_run01/probability_vectors.json'
    previous = json.loads(old.read_text(encoding='utf-8'))
    records = [{**v, 'kind': 'stress' if v['kind'] == 'stress' else 'development'} for v in previous]
    mixture = json.loads((ROOT/'experiments/cpu_gate/protocol.json').read_text(encoding='utf-8'))
    mixture['K'] = [4]
    mixture['heterogeneous']['seed'] = protocol['confirmation']['seed']
    for v in vectors(mixture):
        if v['kind'] == 'heterogeneous':
            records.append({**v, 'kind': 'confirmation', 'id': 'confirm3-'+v['id']})
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        committed = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        if any(committed.get(k) != v for k, v in protocol.items()) or \
                json.loads((out/'probability_vectors.json').read_text(encoding='utf-8')) != records:
            raise ValueError('Prepared inputs changed; use a new output directory')
        return committed, records
    dump(out/'protocol.json', {**protocol, 'prepared_utc': datetime.now(timezone.utc).isoformat()})
    dump(out/'probability_vectors.json', records)
    return protocol, records


def fit_all(protocol, out):
    policies = {}
    for prior in PRIORS:
        paths = [out/f'policy_{prior}_m0_{m0}.json' for m0 in protocol['m0']]
        if any(p.exists() for p in paths) and not all(p.exists() for p in paths):
            raise ValueError('Incomplete prior fit; preserve run and use a new output directory')
        if all(p.exists() for p in paths):
            policies[prior] = {m0: PilotPolicy.from_dict(json.loads(path.read_text(encoding='utf-8')))
                               for m0, path in zip(protocol['m0'], paths)}
        else:
            policies[prior] = fit_prior(prior, protocol, out)
    return policies


def pair_tables(rows):
    result = {}
    for kind in ['confirmation', 'development']:
        subset = [r for r in rows if r['kind'] == kind]
        result[kind] = {f'{candidate}-vs-{baseline}': population_pair(subset, baseline, 'budget-precommitted', candidate=candidate)
                        for candidate in ['JADA', *PRIORS, 'adaptive-nonlinear-oracle']
                        for baseline in ['uniform', 'variance-only']}
    return result


def report(out, rows, summary):
    lines = ['# Pilot-only posterior nonlinear allocator', '', f'Primary decision: **{summary["decision"]}**. CPU evidence only.', '',
             'Scope: K4/B16/m0=1,2, raw budget-precommitted reuse. Offline policy construction receives only the declared prior and numeric integration seeds. Runtime allocation receives only pilot counts and independent tie randomization; no true p or main outcomes.', '',
             'Primary: independent Jeffreys Beta(.5,.5) prior. Secondary: the known six-component synthetic benchmark population prior. The secondary cannot replace a failed primary and is not distribution-free evidence.', '',
             'Fresh confirmation: 96 vectors from the unchanged mixture, seed 3141592026 saved before evaluation. Previously inspected examples are development/stress only and were not used as integration points or to tune priors.', '',
             '| Confirmation comparison | Relative MSE reduction | 95% interval |', '|---|---:|---:|']
    for name, metric in summary['comparisons']['confirmation'].items():
        lines.append(f'| {name} | {100*metric["relative_reduction"]:.2f}% | [{100*metric["relative_ci95"][0]:.2f}, {100*metric["relative_ci95"][1]:.2f}]% |')
    lines += ['', 'Uniform and variance-only coincide under both declared settings; these paired comparisons are the same control.', '',
              '| m0 | Jeffreys reduction vs uniform | Mixture reduction vs uniform |', '|---|---:|---:|']
    for m0, metrics in summary['by_m0'].items():
        lines.append(f'| {m0} | {100*metrics[PRIORS[0]]["relative_reduction"]:.2f}% | {100*metrics[PRIORS[1]]["relative_reduction"]:.2f}% |')
    lines += ['', '![Posterior allocator results](posterior_results.png)', '', '## Numeric integration and exact evaluation', '',
              'Each posterior policy uses eight independent scrambled Sobol replicates: 8,192 prior points for Jeffreys and 12,288 stratified prior points for the mixture. Counts and reward states are enumerated exactly at each integration point. Joint pilot-likelihood-weighted risk numerators and masses are averaged before posterior division, then symmetrized over all 24 thought permutations. Ties within 1e-12 are randomized uniformly, preserving exchangeability.', '',
              'Policy construction approximates the continuous posterior integral; it does not certify exact Bayes optimality. Construction diagnostics record scramble variability and first-half/last-half policy disagreement. Fresh-vector MSE integrates all pilot/main outcomes and tie probabilities exactly. Population confidence intervals are conditional on these frozen policies; prior integration variability is reported separately.', '',
              '| Prior | m0 | Half-scramble weighted policy TV | Max selected-risk SE |', '|---|---:|---:|---:|']
    for d in summary['integration_diagnostics']:
        lines.append(f'| {d["prior"]} | {d["m0"]} | {d["pilot_mass_weighted_half_policy_TV"]:.6f} | '
                     f'{d["max_selected_posterior_risk_SE_across_scrambles"]:.6f} |')
    lines += ['', '## Interpretation', '', summary['interpretation'], '',
              'The predeclared primary criterion is a lower 95% reduction limit above 5% with positive point estimates at both m0 settings. The true-p adaptive oracle remains an information upper comparator, not a deployed policy. No result here establishes gradient, training or LLM improvement, or benefit at larger K/budgets.', '',
              'Raw value estimates remain unbiased: counts are positive and committed before independent main sampling, with lambda fixed before pilots. A policy may use auxiliary tie randomness independent of main rewards without changing that proof. Normalized advantages remain nonlinear and may be biased.', '',
              'No prior was switched or sample count increased after confirmation outcomes. This confirmation set is now consumed. See exact_evaluation_m0_*.npz for all risks and selected oracle policies, policy_*.json for executable pilot-only tables, integration_*.npz for prior integrals, metrics.csv/results.jsonl for complete results, and AUDIT.json for verification.', '',
              f'CPU fit/evaluation wall time: {summary["elapsed_cpu_wall_seconds"]:.2f}s. GPU cost: 0. Review status: same-family provisional.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['JADA', *PRIORS, 'adaptive-nonlinear-oracle']
    vals = [summary['comparisons']['confirmation'][f'{name}-vs-uniform'] for name in names]
    y = np.array([100*x['relative_reduction'] for x in vals])
    low = np.array([100*x['relative_ci95'][0] for x in vals])
    high = np.array([100*x['relative_ci95'][1] for x in vals])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.errorbar(np.arange(4), y, yerr=[y-low, high-y], fmt='o', capsize=5, color='#356788')
    ax.axhline(0, color='gray', linewidth=1)
    ax.axhline(5, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(range(4), ['JADA', 'Pilot-only posterior\nJeffreys (primary)', 'Pilot-only posterior\nknown mixture (secondary)', 'True-p adaptive\noracle (diagnostic)'])
    ax.set_ylabel('Relative MSE reduction vs uniform (%)')
    ax.set_title('Fresh confirmation · K4/B16 · budget-precommitted reuse')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .01, '95% stratified vector-cluster intervals, conditional on frozen policies. Variance-only equals uniform in this scope.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(out/f'posterior_results.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    protocol, records = prepare(args.output)
    out = args.output
    if args.prepare_only:
        print(f'Prepared {len(records)} vectors: 192 development, six stress, 96 fresh confirmation; no fitting/evaluation.', flush=True)
        return
    if (out/'results.jsonl').exists():
        raise FileExistsError('Use a new run directory; do not overwrite results')
    start = time.perf_counter()
    policies = fit_all(protocol, out)
    dump(out/'POLICY_FREEZE.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
         'methods': list(PRIORS), 'information': 'no evaluation p accessed during fitting', 'evaluation_started': False})
    if args.fit_only:
        print('Policies frozen. Confirmation has not been evaluated.', flush=True)
        return
    p = np.array([v['p'] for v in records])
    rows = []
    with (out/'results.jsonl').open('w', encoding='utf-8') as f:
        for m0 in protocol['m0']:
            grid = o.solve(p, 16, m0)
            np.savez_compressed(out/f'exact_evaluation_m0_{m0}.npz', p=p, allocations=grid['allocations'], pilots=grid['pilots'],
                pilot_probabilities=grid['pilot_probabilities'], conditional_risks=grid['conditional_risks'],
                unconditional_risks=grid['unconditional_risks'], **{f'selected_{method}': indices for method, indices in grid['selected'].items()})
            methods = dict(grid['mse'])
            for prior in PRIORS:
                methods[prior] = evaluate(grid, policies[prior][m0])
                if np.any(methods[prior]+1e-12 < methods['adaptive-nonlinear-oracle']):
                    raise ArithmeticError('Pilot-only policy illegally beat global true-p oracle')
            for i, vector in enumerate(records):
                row = {**vector, 'B_total': 16, 'B_main': 16-4*m0, 'm0': m0,
                       'methods': {f'{name}/budget-precommitted': {'mse': float(risks[i])} for name, risks in methods.items()}}
                rows.append(row)
                f.write(json.dumps(row, allow_nan=False)+'\n')
            f.flush()
    pairs = pair_tables(rows)
    by_m0 = {str(m0): {prior: population_pair([r for r in rows if r['kind'] == 'confirmation' and r['m0'] == m0],
                                               'uniform', 'budget-precommitted', candidate=prior)
                          for prior in PRIORS} for m0 in protocol['m0']}
    primary = pairs['confirmation']['posterior-Jeffreys-vs-uniform']
    passed = primary['relative_ci95'][0] > .05 and all(x['posterior-Jeffreys']['relative_reduction'] > 0 for x in by_m0.values())
    decision = 'PASS_CPU_SIGNAL' if passed else 'NO_CLEAR_SIGNAL'
    interpretation = 'The primary pilot-only method passes the scoped CPU criterion. Broader CPU robustness is justified before any frozen-model study; this is not a training GO.' if passed else \
        'The primary pilot-only method does not pass the declared CPU criterion. Keep the negative outcome; the secondary benchmark-informed prior cannot rescue this primary gate. Oracle headroom does not guarantee recoverable pilot-only gains.'
    diagnostics = [json.loads((out/f'integration_diagnostic_{prior}_m0_{m0}.json').read_text(encoding='utf-8'))
                   for prior in PRIORS for m0 in protocol['m0']]
    summary = {'decision': decision, 'comparisons': pairs, 'by_m0': by_m0, 'integration_diagnostics': diagnostics,
               'interpretation': interpretation, 'cells': len(rows), 'method_evaluations': len(rows)*8,
               'elapsed_cpu_wall_seconds': time.perf_counter()-start, 'GPU_cost': 0,
               'review_independence': 'same-family', 'acceptance_status': 'provisional'}
    dump(out/'summary.json', summary)
    save_csv(out/'metrics.csv', [{'kind': r['kind'], 'id': r['id'], 'component': r['component'], 'm0': r['m0'],
                                'method': name, 'mse': value['mse']} for r in rows for name, value in r['methods'].items()])
    plot(out, summary)
    report(out, rows, summary)
    from .audit import audit
    audit(out)
    print(json.dumps({'decision': decision, 'primary': primary, 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
