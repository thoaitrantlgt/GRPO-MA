"""Render the predeclared diagnostic bins with Matplotlib; no extra sampling."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directory):
    with (directory/'diagnostic_bins.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, field, title in zip(axes, ['pilot_fraction', 'dispersion_v', 'max_leverage'],
                               ['Pilot fraction', 'Group dispersion (population variance)', 'Maximum leverage']):
        selected = [r for r in rows if r['variable'] == field]
        y = np.array([100*float(r['relative_reduction']) for r in selected])
        lo = np.array([100*float(r['ci_low']) for r in selected])
        hi = np.array([100*float(r['ci_high']) for r in selected])
        x = np.arange(len(selected))
        ax.errorbar(x, y, yerr=np.array([y-lo, hi-y]), fmt='o-', capsize=4, color='#17638a')
        ax.axhline(0, color='#888888', linestyle='--', linewidth=1)
        ax.set_xticks(x, [f'{float(r["low"]):g}–{float(r["high"]):g}\n(n={r["vectors"]})' for r in selected], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Relative MSE reduction (%)\nPositive favors JADA')
    fig.suptitle('JADA vs variance-only · budget-precommitted reuse', fontsize=14)
    fig.text(.5, .015, 'Predeclared bins; pointwise 95% vector-cluster intervals. Descriptive associations; K and budgets are mixed.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(directory/f'improvement_curves.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    plot(parser.parse_args().directory)
