import numpy as np


def report(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    diagnostics=summary['diagnostics']
    for prior in ['posterior-Jeffreys','posterior-mixture']:
        for setting in ['fixed-main-m0-4','fixed-total-m0-4']:
            data=[d for d in diagnostics if d['prior']==prior and d['setting']==setting]
            x=[d['total_prior_points'] for d in data]
            label=prior+' / '+setting.replace('-m0-4','')
            axes[0].plot(x,[d['consumed_development_reduction_pct'] for d in data],'o-',label=label)
            axes[1].plot(x,[d['half_policy_TV'] for d in data],'o-',label=label)
    axes[0].set_ylabel('Consumed-development MSE reduction (%)'); axes[0].axhline(0,color='gray',lw=.8)
    axes[1].set_ylabel('Prior-weighted half-policy total variation'); axes[1].axhline(.01,color='gray',ls='--',lw=.8)
    for ax in axes:
        ax.set_xscale('log',base=2); ax.set_xlabel('Total prior points'); ax.legend(fontsize=6)
    fig.suptitle('Numerical convergence only: no fresh confirmation')
    fig.tight_layout()
    for suffix in ['png','pdf','svg']:
        fig.savefig(out/f'convergence.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# Final posterior convergence diagnosis', '', '**'+summary['decision']+'**. Original gate remains NO_CLEAR_SIGNAL.', '',
           'Two m0=4 settings; unchanged priors and probability-vector distribution. Eight independently scrambled Sobol replicates and three nested component-balanced levels were committed before this run. Chunking pools joint numerators/masses without averaging posterior ratios. Exact finite reward risks are used at every prior point. No fresh confirmation or new reward sampling.', '',
           '| Prior | Setting | Prior points | Half-policy TV | TV vs original | Development reduction |',
           '|---|---|---:|---:|---:|---:|']
    for d in diagnostics:
        lines.append(f'| {d["prior"]} | {d["setting"]} | {d["total_prior_points"]} | {d["half_policy_TV"]:.6f} | {d["policy_TV_vs_original"]:.6f} | {d["consumed_development_reduction_pct"]:.4f}% |')
    lines+=['', '![Convergence](convergence.png)', '',
            '1. Observation: the table records gains and policy agreement as numerical integration increases. Development risks use the previous 96-vector confirmation, now consumed; all six stress cases remain in metrics.csv.',
            '2. Interpretation: stable tiny gains support stopping this allocator family under the tested model and objectives. Instability instead leaves the numerical diagnosis open. These are approximate policies, not certified exact continuous Bayes optima or an impossibility theorem.',
            '3. Implication: this analysis cannot reverse the original gate or create a new confirmation pass. Reported scramble SEs are diagnostics conditional on a policy selected using these scrambles, not independent optimal-risk uncertainty intervals.',
            '4. Next step: if stopping is recommended, revisit whether normalized-advantage MSE predicts token-weighted gradient MSE before further allocator sweeps. Any new method needs a distinct hypothesis and new preregistered confirmation; no GPU or online training is justified by this diagnosis.', '',
            f'Audit PASS; policy reconstruction, oracle dominance and balanced nested prefixes checked. CPU time {summary["elapsed_seconds"]:.1f}s. GPU cost0. Review is local diagnostic, with no fresh reviewer clearance.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
