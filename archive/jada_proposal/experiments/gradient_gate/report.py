"""Tables and standalone figures for the predeclared synthetic geometry gate."""
from experiments.cpu_gate.run import save_csv
from .core import NAMES


def report(out,protocol,rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    pairs=summary['comparisons']
    groups=[]
    for feature,edges in protocol['diagnostic_bins'].items():
        for low,high in zip(edges[:-1],edges[1:]):
            subset=[r for r in rows if r['kind']=='confirmation' and r['gram'] in protocol['primary_grams'] and
                    (low<=r[feature]<high or r[feature]==high==edges[-1])]
            if not subset:
                groups.append({'feature':feature,'low':low,'high':high,'vectors':0,
                               'oracle_vs_uniform_pct':None,'oracle_vs_scalar_pct':None,'gradient_JADA_vs_uniform_pct':None}); continue
            def mean(method):
                return np.mean([r['methods'][method+'/budget-precommitted']['mse'] for r in subset])
            oracle=mean('adaptive-gradient-oracle'); uniform=mean('uniform'); scalar=mean('adaptive-scalar-oracle')
            groups.append({'feature':feature,'low':low,'high':high,'vectors':len({r['id'] for r in subset}),
                           'oracle_vs_uniform_pct':100*(1-oracle/uniform),'oracle_vs_scalar_pct':100*(1-oracle/scalar),
                           'gradient_JADA_vs_uniform_pct':100*(1-mean('gradient-JADA')/uniform)})
    save_csv(out/'diagnostic_groups.csv',groups)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    active=[name for name in NAMES if name!='aligned-null']
    keys=['adaptive-gradient-oracle-vs-uniform','adaptive-gradient-oracle-vs-adaptive-scalar-oracle','gradient-JADA-vs-uniform']
    labels=['Oracle vs uniform','Oracle vs scalar oracle','Pilot gradient-JADA vs uniform']
    for offset,key,label in zip([-.15,0,.15],keys,labels):
        metrics=[pairs['by_gram'][name][key] for name in active]
        y=np.array([100*m['relative_reduction'] for m in metrics])
        lo=np.array([100*m['relative_ci95'][0] for m in metrics]); hi=np.array([100*m['relative_ci95'][1] for m in metrics])
        axes[0].errorbar(np.arange(len(active))+offset,y,yerr=[y-lo,hi-y],fmt='o',capsize=2,label=label)
    axes[0].set_xticks(range(len(active)),active,rotation=45,ha='right'); axes[0].set_ylabel('Gradient-MSE reduction (%)'); axes[0].legend(fontsize=6,loc='center right')
    for method in ['adaptive-gradient-oracle','gradient-JADA','gradient-surrogate-oracle','JADA']:
        metrics=[pairs['by_setting'][s['id']][method+'-vs-uniform'] for s in protocol['settings']]
        axes[1].plot([4*s['m0']/s['total'] for s in protocol['settings']],[100*m['relative_reduction'] for m in metrics],'o-',label=method)
    axes[1].set_xlabel('Pilot fraction (different budgets)'); axes[1].legend(fontsize=6)
    for feature,marker in [('dispersion','o'),('max_leverage','s')]:
        data=[r for r in groups if r['feature']==feature and r['vectors']]
        axes[2].plot(range(len(data)),[r['oracle_vs_scalar_pct'] for r in data],marker+'-',label=feature)
    axes[2].set_xticks(range(3),['Low fixed bin','Middle','High']); axes[2].set_ylabel('Oracle improvement over scalar oracle (%)'); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.axhline(0,color='gray',lw=.8)
    fig.suptitle('Synthetic fixed-Gram CPU headroom; subgroup curves descriptive')
    fig.tight_layout()
    for extension in ['png','pdf','svg']:
        fig.savefig(out/f'gradient_headroom.{extension}',dpi=180)
    plt.close(fig)
    primary=pairs['primary']
    lines=['# Synthetic fixed-Gram gradient-MSE CPU gate', '', '**'+summary['decision']+'**. Information headroom only; GPU cost0.', '',
           'Scope K4, B16/m0=1,2 and B24/m0=4, budget-precommitted reuse. Fixed trace-one PSD Grams supplied by the synthetic model independently of p and outcomes. This is not measured LLM score geometry or reward-score dependence.', '',
           'Fresh confirmation retains all96 vectors (seed1732052026),16 per unchanged probability-mixture component. Prior confirmation is development; six named stress regimes retained. Primary equally weights four predeclared nontrivial Grams and three settings within each vector; interval sample count remains96.', '',
           '| Primary comparison | Relative gradient-MSE reduction | CI95 |', '|---|---:|---|']
    for key in ['adaptive-gradient-oracle-vs-uniform','adaptive-gradient-oracle-vs-adaptive-scalar-oracle',
                'fixed-gradient-oracle-vs-uniform','gradient-surrogate-oracle-vs-uniform','gradient-JADA-vs-uniform','JADA-vs-uniform']:
        m=primary[key]; lines.append(f'| {key} | {100*m["relative_reduction"]:.3f}% | [{100*m["relative_ci95"][0]:.3f}%, {100*m["relative_ci95"][1]:.3f}%] |')
    lines+=['', 'Both oracle lower95 limits must exceed5% for HEADROOM_FOR_GRADIENT_CPU. This decision does not declare gradient-JADA successful. True-p oracles remain unavailable-information controls; known fixed G is supplied by the toy design rather than learned from pilots.', '',
            'The objective-specific point effect is positive and its interval excludes zero; NO_OBJECTIVE_SPECIFIC_SIGNAL means failure of the predeclared material-benefit criterion, not proof that Gram weighting has no effect. The extra oracle gain is3.091% [2.392%,3.789%], below5%. The known-G pilot gradient-JADA increases weighted error1.623%, and even the true-p local surrogate increases it5.102%. Changing the local coefficient alone has not repaired the surrogate mismatch under this benchmark.', '',
            '| Gram | Weighted oracle vs uniform | Weighted vs scalar adaptive oracle | Pilot gradient-JADA vs uniform |',
            '|---|---:|---:|---:|']
    for name in active:
        pp=pairs['by_gram'][name]
        vals=[100*pp[key]['relative_reduction'] for key in keys]
        lines.append(f'| {name} | {vals[0]:.3f}% | {vals[1]:.3f}% | {vals[2]:.3f}% |')
    lines+=['| aligned-null | Zero absolute loss | Relative ratios undefined | Zero absolute loss |', '',
            'Isotropic equals the scalar loss exactly. Equicorrelated-control is exactly half the scalar loss on centered credits. Aligned-null annihilates all credit errors. These controls are reported but excluded from the objective-specific primary average; no Gram was selected after observing winners.', '',
            '![Gradient headroom](gradient_headroom.png)', '',
            'Confidence intervals are stratified paired vector-cluster Welch-Satterthwaite t delta intervals across the unchanged six probability strata. Reward count states, allocation candidates and risks are exact; repeated Grams/settings are not independent observations. Fixed-bin subgroup curves are descriptive. Pilot fraction curves change budgets and do not isolate a causal pilot effect.', '',
            f'Completed {summary["cells"]} cells and {summary["method_evaluations"]} method evaluations. CPU wall time {summary["elapsed_seconds"]:.1f}s. Review: {summary["review_status"]}. See AUDIT.json for independent saved-table minima, scalar/null boundaries and direct weighted-risk checks.', '',
            'The gate fails its predeclared objective-specific threshold. Preserve the negative outcome rather than choosing a winning Gram subset or changing the probability distribution. A further method requires a distinct hypothesis and a new protocol/confirmation; this result does not justify launching a weighted posterior policy automatically. No claim about all possible score geometries follows from these fixed matrices. No training, real LLM-gradient or GPU benefit has been established. The confirmation set is consumed.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
