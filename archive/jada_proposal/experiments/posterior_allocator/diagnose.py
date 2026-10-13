"""Post hoc diagnostics of consumed confirmation, without refitting policies."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.cpu_gate.run import dump
from .policy import PilotPolicy

SOURCE = Path('refine-logs/posterior-allocator-gate/20261011_run01')
OUTPUT = Path('refine-logs/posterior-information-diagnostic/20261011_run01')


def main():
    if OUTPUT.exists():
        raise FileExistsError('Keep previous diagnostics; use a new output directory')
    OUTPUT.mkdir(parents=True)
    records = json.loads((SOURCE/'probability_vectors.json').read_text(encoding='utf-8'))
    indices = np.array([i for i, r in enumerate(records) if r['kind'] == 'confirmation'])
    source_files = list(SOURCE.glob('*.npz'))+list(SOURCE.glob('policy_*.json'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    dump(OUTPUT/'PROVENANCE.json', {'source': str(SOURCE), 'source_sha256': hashes,
         'status': 'post hoc descriptive analysis; confirmation consumed',
         'new_sampling': False, 'refitting': False, 'GPU_cost': 0})
    rows, histories, overall = [], [], []
    for m0 in [1, 2]:
        with np.load(SOURCE/f'exact_evaluation_m0_{m0}.npz') as a:
            risk = a['conditional_risks'][indices]
            weights = a['pilot_probabilities'][indices]
            p = a['p'][indices]
        policy = PilotPolicy.from_dict(json.loads((SOURCE/f'policy_posterior-Jeffreys_m0_{m0}.json').read_text(encoding='utf-8')))
        uniform = np.flatnonzero(np.all(policy.allocations == (16-4*m0)//4, axis=1))[0]
        base = risk[:, :, uniform]
        chosen = np.sum(risk*policy.probabilities[None], axis=2)
        oracle = risk.min(2)
        baseline = np.sum(weights*base, axis=1)
        policy_mse = np.sum(weights*chosen, axis=1)
        oracle_mse = np.sum(weights*oracle, axis=1)
        oracle_gain = weights*(base-oracle)
        policy_gain = weights*(base-chosen)
        residue = weights*(chosen-oracle)
        if not np.allclose(oracle_gain, policy_gain+residue, atol=1e-13):
            raise ArithmeticError('Headroom decomposition failed')
        if np.min(residue) < -1e-12:
            raise ArithmeticError('Frozen policy beats full-information oracle')
        z = p-p.mean(1, keepdims=True)
        dispersion = np.mean(z*z, axis=1)
        leverage = np.max(z*z, axis=1)/(4*dispersion)
        aggregate = {'m0': m0, 'pilot_fraction': m0/4, 'uniform_mse': float(baseline.mean()),
                     'policy_mse': float(policy_mse.mean()), 'oracle_mse': float(oracle_mse.mean()),
                     'policy_reduction_pct': float(100*(baseline-policy_mse).mean()/baseline.mean()),
                     'oracle_reduction_pct': float(100*(baseline-oracle_mse).mean()/baseline.mean()),
                     'fraction_oracle_gain_recovered': float(policy_gain.sum()/oracle_gain.sum()),
                     'probability_nonuniform_allocation': float(weights.mean(0)@(1-policy.probabilities[:, uniform]))}
        overall.append(aggregate)
        with np.load(SOURCE/f'integration_posterior-Jeffreys_m0_{m0}.npz') as a:
            nums, masses = a['numerators'], a['masses']
        prior_risk = nums.mean(0)/masses.mean(0)[:, None]
        for h, pilot in enumerate(policy.pilots):
            history = {'m0': m0, 'pilot': pilot.tolist(), 'probability': float(weights[:, h].mean()),
                       'policy_nonuniform_probability': float(1-policy.probabilities[h, uniform]),
                       'oracle_gain_contribution': float(oracle_gain[:, h].mean()),
                       'policy_gain_contribution': float(policy_gain[:, h].mean()),
                       'remaining_information_gap': float(residue[:, h].mean()),
                       'prior_expected_improvement': float(prior_risk[h, uniform]-policy.posterior_risk[h].min()),
                       'selected_allocations': policy.allocations[policy.probabilities[h] > 0].tolist()}
            histories.append(history)
        for local, index in enumerate(indices):
            rows.append({'id': records[index]['id'], 'component': records[index]['component'], 'm0': m0,
                         'pilot_fraction': m0/4, 'dispersion': float(dispersion[local]),
                         'max_leverage': float(leverage[local]), 'uniform_mse': float(baseline[local]),
                         'policy_mse': float(policy_mse[local]), 'oracle_mse': float(oracle_mse[local])})
    dump(OUTPUT/'summary.json', overall)
    dump(OUTPUT/'pilot_histories.json', histories)
    with (OUTPUT/'vectors.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups = []
    for m0 in [1, 2]:
        selected = [r for r in rows if r['m0'] == m0]
        for feature in ['component', 'dispersion', 'max_leverage']:
            values = np.array([r[feature] for r in selected])
            cuts = None if feature == 'component' else np.quantile(values.astype(float), [1/3, 2/3])
            labels = values if cuts is None else np.searchsorted(cuts, values.astype(float), side='right')
            for label in np.unique(labels):
                subset = [r for r, lab in zip(selected, labels) if lab == label]
                base = np.mean([r['uniform_mse'] for r in subset])
                groups.append({'m0': m0, 'feature': feature, 'group': str(label), 'n': len(subset),
                               'cutpoints': None if cuts is None else cuts.tolist(),
                               'policy_reduction_pct': 100*(base-np.mean([r['policy_mse'] for r in subset]))/base,
                               'oracle_reduction_pct': 100*(base-np.mean([r['oracle_mse'] for r in subset]))/base})
    dump(OUTPUT/'groups.json', groups)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, feature in zip(axes, ['pilot_fraction', 'dispersion', 'max_leverage']):
        if feature == 'pilot_fraction':
            ax.plot([x['pilot_fraction'] for x in overall], [x['policy_reduction_pct'] for x in overall], 'o-', label='Pilot-only')
            ax.plot([x['pilot_fraction'] for x in overall], [x['oracle_reduction_pct'] for x in overall], 'o--', label='True-p oracle')
        else:
            for m0 in [1, 2]:
                data = [x for x in groups if x['feature'] == feature and x['m0'] == m0]
                ax.plot(range(3), [x['policy_reduction_pct'] for x in data], 'o-', label=f'Pilot-only m0={m0}')
                ax.plot(range(3), [x['oracle_reduction_pct'] for x in data], 'o--', label=f'Oracle m0={m0}')
            ax.set_xticks(range(3), ['Low', 'Middle', 'High'])
        ax.set_xlabel(feature.replace('_', ' ')); ax.axhline(0, color='gray', lw=.7)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('MSE reduction vs uniform (%)')
    fig.suptitle('Consumed confirmation: descriptive information-gap diagnostics')
    fig.tight_layout()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT/f'information_gap.{extension}', dpi=180)
    plt.close(fig)
    lines = ['# Posterior allocator information-gap diagnosis', '',
             'Post hoc descriptive analysis of consumed confirmation. No policy refit, new outcomes or new independent evidence. K4/B16, budget-precommitted reuse; equal 16-vector strata. Subgroup cutpoints are empirical tertiles chosen without comparing winners; no subgroup significance claims.', '',
             'For each pilot h, uniform-to-oracle gain = uniform-to-frozen-policy gain + frozen-policy-to-oracle regret. This identity is verified against all saved count risks. The remaining regret is an information comparator gap, not a proven lower bound for all pilot-only methods.', '',
             '| m0 | Pilot fraction | Pilot-only reduction | Oracle reduction | Oracle gain recovered | Nonuniform allocation probability |',
             '|---|---:|---:|---:|---:|---:|']
    for x in overall:
        lines.append(f'| {x["m0"]} | {x["pilot_fraction"]:.2f} | {x["policy_reduction_pct"]:.4f}% | {x["oracle_reduction_pct"]:.2f}% | {100*x["fraction_oracle_gain_recovered"]:.3f}% | {100*x["probability_nonuniform_allocation"]:.2f}% |')
    lines += ['', 'The frozen policies use uniform counts at all 16 m0=1 histories and deviate in only eight of 81 m0=2 histories. Increasing pilot fraction here also decreases the main budget and changes reuse weight; two points do not isolate a causal effect of pilot information.', '',
              '![Diagnostic curves](information_gap.png)', '',
              '| m0 | Feature | Group | Vectors | Pilot-only reduction | Oracle reduction |',
              '|---|---|---|---:|---:|---:|']
    for x in groups:
        lines.append(f'| {x["m0"]} | {x["feature"]} | {x["group"]} | {x["n"]} | {x["policy_reduction_pct"]:.4f}% | {x["oracle_reduction_pct"]:.2f}% |')
    lines += ['', 'Interpretation: stable prior-integrated policies prefer near-uniform allocation, while full-information decisions depend on p beyond the pilot history. This supports testing an information-limited explanation; it does not establish an impossibility theorem. Reuse/normalization also remain candidate causes.', '',
              'Next hypothesis to preregister: increasing pilot information while holding main budget and reuse weight fixed changes the posterior-optimal gain. This requires a separate diagnostic design with total cost explicitly accounted for, alongside matched total-budget controls. Use the unchanged six-component distribution and new confirmation; do not use tertile winners to select the benchmark. No training GO, no GPU cost.', '']
    (OUTPUT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ArithmeticError('Source modified during diagnostic')
    dump(OUTPUT/'AUDIT.json', {'status': 'PASS', 'decomposition': 'PASS', 'oracle_dominance': 'PASS',
         'source_unchanged': True, 'vectors': 96, 'settings': 2, 'GPU_cost': 0})
    print(json.dumps(overall, indent=2))


if __name__ == '__main__':
    main()
