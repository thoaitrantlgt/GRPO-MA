"""Render frozen K4 diagnostic outputs without solving or selecting policies."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path('refine-logs/information-control-k4/20261011_run01')

def main():
    rows = list(csv.DictReader((BASE/'effects.csv').open()))
    summary = json.loads((BASE/'summary.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for m0 in [1, 2]:
        selected = [r for r in rows if int(r['m0']) == m0]
        x = np.array([int(r['base_points_per_scramble']) for r in selected])
        y = np.array([float(r['reduction'])*100 for r in selected])
        low = np.array([float(r['development_ci_low'])*100 for r in selected])
        high = np.array([float(r['development_ci_high'])*100 for r in selected])
        axes[0].errorbar(x, y, yerr=[y-low, high-y], marker='o', capsize=3, label=f'm0={m0}')
    axes[0].axhline(5, color='gray', linestyle='--', label='Practical threshold')
    axes[0].set(xscale='log', xticks=[16,64,256], xticklabels=['16','64','256'], xlabel='Base points per scramble', ylabel='Risk reduction (%)', title='Consumed-data development intervals')
    axes[0].legend(fontsize=8)
    stress = list(csv.DictReader((BASE/'stress.csv').open()))
    names = [r['id'] for r in stress if int(r['m0']) == 1]
    for m0 in [1,2]:
        y = [float(r['regression'])*100 for r in stress if int(r['m0']) == m0]
        axes[1].barh(np.arange(6)+(m0-1.5)*.32, y, height=.3, label=f'm0={m0}')
    axes[1].axvline(0,color='gray',linewidth=.6)
    axes[1].axvline(10,color='gray',linestyle='--')
    axes[1].set(yticks=range(6),yticklabels=names,xlabel='Risk regression (%)',title='Original stresses; highest resolution')
    axes[1].legend(fontsize=8)
    keys = ['highest_vs_adjacent_aggregate_risk_change_over_uniform','maximum_stress_risk_change_over_uniform','prior_value_change_over_uniform','max_lower_policy_regret_over_uniform']
    values = [100*summary[k] for k in keys]
    axes[2].barh(range(4), values, color=['#b44545' if v>1 else '#447b9c' for v in values])
    axes[2].axvline(1,color='gray',linestyle='--',label='Stability threshold')
    axes[2].set(yticks=range(4),yticklabels=['Aggregate risk','Stress risk','Prior value','Policy regret'],xlabel='Normalized difference (%)',title='Numerical stability diagnostics')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(BASE/'control_stability.png',dpi=200)
    fig.savefig(BASE/'control_stability.pdf')
    plt.close(fig)

if __name__ == '__main__':
    main()
