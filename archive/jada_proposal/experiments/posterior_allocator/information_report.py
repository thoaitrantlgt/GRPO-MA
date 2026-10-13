"""Independent saved-artifact checks and figures for the information gate."""
import itertools
import json
import math
import numpy as np
from experiments.cpu_gate import estimator as e
from experiments.cpu_gate.run import dump, save_csv, vectors
from .policy import PilotPolicy, make_policy, evaluate
from .fit import PRIORS, prior_points


def report_and_audit(out,protocol,records,rows,summary):
    summary['review_status']='local-only; delegated reviewer unavailable (usage limit)'
    dump(out/'summary.json',summary)
    from .information import comparator_risks, comparisons
    freeze=json.loads((out/'POLICY_FREEZE.json').read_text(encoding='utf-8'))
    if freeze['evaluation_started'] is not False:
        raise ValueError('Missing pre-evaluation freeze')
    lookup={(r['id'],r['setting']):r for r in rows}
    if len(lookup)!=len(records)*len(protocol['settings']) or len(rows)!=len(lookup):
        raise ValueError('Incomplete or duplicate coverage')
    confirmation=[r for r in records if r['kind']=='confirmation']
    if len(confirmation)!=96 or any(sum(r['component']==str(c) for r in confirmation)!=16 for c in range(6)):
        raise ValueError('Confirmation filtered')
    from pathlib import Path
    population=json.loads(Path('experiments/cpu_gate/protocol.json').read_text(encoding='utf-8'))
    population['K']=[4]; population['heterogeneous']['seed']=protocol['confirmation_seed']
    expected=[{**r,'kind':'confirmation' if r['kind']=='heterogeneous' else 'stress',
               'id':'confirm4-'+r['id'] if r['kind']=='heterogeneous' else r['id']} for r in vectors(population)]
    if records!=expected:
        raise ValueError('Evaluation vectors differ from committed seed/distribution')
    maxdirect=maxmetric=maxbias=0.
    equal_cost_uniform=[]
    for prior in PRIORS:
        with np.load(out/f'prior_points_{prior}.npz') as a:
            for rep in range(protocol['offline_integration']['replicates']):
                p,c=prior_points(prior,rep,protocol['offline_integration'])
                if not np.array_equal(p,a['points'][rep]) or not np.array_equal(c,a['components'][rep]):
                    raise ValueError('Prior rule changed')
    for s in protocol['settings']:
        with np.load(out/f'exact_{s["id"]}.npz') as a:
            g=dict(a)
        p=g['p']; m=g['allocations']; pilots=g['pilots']; weights=g['pilot_probabilities']; risk=g['conditional_risks']
        if not np.array_equal(p,np.array([r['p'] for r in records])):
            raise ValueError('Wrong evaluation p')
        if len(m)!=math.comb(s['total']-4*s['m0']-1,3) or len(set(map(tuple,m)))!=len(m) or np.any(m<1) or np.any(m.sum(1)!=s['total']-4*s['m0']):
            raise ValueError('Allocation space incomplete')
        if set(map(tuple,pilots))!=set(itertools.product(range(s['m0']+1),repeat=4)):
            raise ValueError('Pilot coverage incomplete')
        np.testing.assert_allclose(weights.sum(1),1,atol=1e-13)
        maxbias=max(maxbias,float(np.max(np.abs(s['lambda']*(weights@(pilots/s['m0'])-p)))))
        methods=comparator_risks(g,s)
        if s['arm']=='fixed-total':
            equal_cost_uniform.append(methods['uniform'])
        for prior in PRIORS:
            data=json.loads((out/f'policy_{prior}_{s["id"]}.json').read_text(encoding='utf-8'))
            policy=PilotPolicy.from_dict(data)
            if data['reuse_lambda']!=s['lambda'] or data['construction']['fit_completed_utc']>freeze['frozen_utc']:
                raise ValueError('Reuse or freeze mismatch')
            with np.load(out/f'integration_{prior}_{s["id"]}.npz') as a:
                rebuilt=make_policy(a['pilots'],a['allocations'],a['numerators'].mean(0),a['masses'].mean(0),s['total'],s['m0'],prior)
            np.testing.assert_array_equal(rebuilt.probabilities,policy.probabilities)
            np.testing.assert_allclose(rebuilt.posterior_risk,policy.posterior_risk,atol=1e-13)
            hmap={tuple(h):i for i,h in enumerate(pilots)}; mmap={tuple(x):i for i,x in enumerate(m)}
            for perm in itertools.permutations(range(4)):
                hi=[hmap[tuple(h[list(perm)])] for h in pilots]
                mi=[mmap[tuple(x[list(perm)])] for x in m]
                np.testing.assert_allclose(policy.probabilities,policy.probabilities[np.ix_(hi,mi)],atol=1e-13)
            methods[prior]=evaluate(g,policy)
        for name,values in methods.items():
            if np.any(values<methods['adaptive-nonlinear-oracle']-1e-12):
                raise ValueError('Full-information optimum violated')
            for i,r in enumerate(records):
                maxmetric=max(maxmetric,abs(values[i]-lookup[(r['id'],s['id'])]['methods'][name+'/precommitted']['mse']))
        for v,h,j in [(0,0,0),(6,len(pilots)//2,len(m)//2),(len(p)-1,len(pilots)-1,len(m)-1)]:
            t,prob=e.binomial_states(m[j],p[v])
            val=s['lambda']*pilots[h]/s['m0']+(1-s['lambda'])*t/m[j]
            direct=prob@np.mean((e.advantage(val)-e.advantage(p[v]))**2,axis=1)
            maxdirect=max(maxdirect,abs(direct-risk[v,h,j]))
    if maxmetric>1e-12 or maxbias>1e-12 or maxdirect>1e-12:
        raise ValueError('Saved risk audit failed')
    for values in equal_cost_uniform[1:]:
        np.testing.assert_allclose(values,equal_cost_uniform[0],atol=1e-13,rtol=0)
    if comparisons(rows)!=summary['comparisons']:
        raise ValueError('Reported intervals differ from saved risks')
    dump(out/'AUDIT.json',{'status':'PASS','exact_cells':len(rows),'methods_per_cell':6,
         'max_direct_residual':maxdirect,'max_saved_metric_residual':maxmetric,'max_raw_value_bias':maxbias,
         'posterior_minimizers_and_permutation_equivariance':'PASS','prior_rule_and_freeze':'PASS',
         'matched_total_uniform_pooling_invariant':'PASS','GPU_cost':0})
    groups=[]
    for s in protocol['settings']:
        fresh=[r for r in rows if r['kind']=='confirmation' and r['setting']==s['id']]
        for feature,edges in [('dispersion',protocol['dispersion_edges']),('max_leverage',protocol['leverage_edges'])]:
            for low,high in zip(edges[:-1],edges[1:]):
                subset=[r for r in fresh if low<=r[feature]<high or r[feature]==high==edges[-1]]
                if not subset:
                    groups.append({'setting':s['id'],'feature':feature,'low':low,'high':high,'n':0,
                                   'posterior_reduction_pct':None,'oracle_reduction_pct':None}); continue
                baseline=np.mean([r['methods']['uniform/precommitted']['mse'] for r in subset])
                groups.append({'setting':s['id'],'feature':feature,'low':low,'high':high,'n':len(subset),
                     'posterior_reduction_pct':100*(1-np.mean([r['methods']['posterior-Jeffreys/precommitted']['mse'] for r in subset])/baseline),
                     'oracle_reduction_pct':100*(1-np.mean([r['methods']['adaptive-nonlinear-oracle/precommitted']['mse'] for r in subset])/baseline)})
    save_csv(out/'diagnostic_groups.csv',groups)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i,arm in enumerate(['fixed-main','fixed-total']):
        settings=[s for s in protocol['settings'] if s['arm']==arm]
        ax=axes[i,0]
        for method in ['posterior-Jeffreys','posterior-mixture','adaptive-nonlinear-oracle','JADA']:
            metrics=[summary['comparisons'][s['id']][method+'-vs-uniform'] for s in settings]
            y=np.array([100*x['relative_reduction'] for x in metrics])
            lower=np.array([100*x['relative_ci95'][0] for x in metrics]); upper=np.array([100*x['relative_ci95'][1] for x in metrics])
            ax.errorbar([4*s['m0']/s['total'] for s in settings],y,yerr=[y-lower,upper-y],marker='o',capsize=3,label=method)
        ax.set_xlabel('Pilot fraction'); ax.set_ylabel(arm+'\nMSE reduction (%)'); ax.legend(fontsize=6)
        for j,feature in enumerate(['dispersion','max_leverage'],1):
            ax=axes[i,j]
            for s in settings:
                data=[g for g in groups if g['setting']==s['id'] and g['feature']==feature and g['n']>0]
                ax.plot([(g['low']+g['high'])/2 for g in data],[g['posterior_reduction_pct'] for g in data],marker='o',label=f'm0={s["m0"]}')
            ax.set_xlabel(feature); ax.legend(fontsize=7)
        for ax in axes[i]:
            ax.axhline(0,color='gray',lw=.7)
    fig.suptitle('Pilot-information CPU gate: conditional policy intervals; descriptive fixed-bin groups')
    fig.tight_layout()
    for suffix in ['png','svg','pdf']:
        fig.savefig(out/f'information_gate.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# Pilot-information CPU gate', '', '**'+summary['decision']+'**. GPU cost 0. Implementation review: '+summary['review_status']+'.', '',
           'Unchanged six-component mixture; 96 fresh vectors (seed1618032026) plus six stress regimes saved before fitting. Jeffreys primary; known-mixture secondary cannot rescue a failed primary. All positive integer allocations and pilot/main count states are enumerated exactly. Continuous prior integration is approximate; intervals are stratified paired vector-cluster t intervals conditional on frozen policies.', '',
           '| Arm | m0 | Total/main | Reuse lambda | Method vs uniform | MSE reduction | CI95 |',
           '|---|---:|---|---:|---|---:|---|']
    for s in protocol['settings']:
        for method in ['posterior-Jeffreys','posterior-mixture','JADA','adaptive-nonlinear-oracle']:
            x=summary['comparisons'][s['id']][method+'-vs-uniform']
            lines.append(f'| {s["arm"]} | {s["m0"]} | {s["total"]}/{s["total"]-4*s["m0"]} | {s["lambda"]:.3f} | {method} | {100*x["relative_reduction"]:.3f}% | [{100*x["relative_ci95"][0]:.3f}%, {100*x["relative_ci95"][1]:.3f}%] |')
    trend=summary['paired_gain_trend_m0_4_minus_1']
    lines+=['',f'Paired fixed-main absolute MSE gain trend (m0=4 minus m0=1): {trend["difference"]:.6f}, CI95 {trend["difference_ci95"]}. This tests gains relative to each matched uniform control; it is not total-cost improvement.', '',
            'The fixed-main arm holds main8 and lambda=.5 but spends total12/16/24. The fixed-total arm spends total24 and uses main20/16/8 with budget-precommitted reuse. Fixed-main benefit alone cannot establish equal-cost value.', '',
            'Primary information gate: Jeffreys lower CI95 >5% at fixed-main/m0=4. Practical gate additionally requires >5% at fixed-total/m0=4 and positive points in every fixed-total setting. Neither is a training GO.', '',
            '![Gate curves](information_gate.png)', '',
            'All methods versus variance-only are also retained in summary.json; variance-only may differ from uniform at m0=4. Metrics.csv/results.jsonl preserve stress and confirmation results. Fixed bins were committed before confirmation; subgroup lines are descriptive, without significance claims.', '',
            '## Integration stability', '', '| Prior | Setting | Weighted half-policy TV | Changed histories | Max paired scramble SE |', '|---|---|---:|---:|---:|']
    for d in summary['integration_diagnostics']:
        lines.append(f'| {d["prior"]} | {d["setting"]} | {d["pilot_mass_weighted_half_policy_TV"]:.5f} | {d["half_policy_changed_histories"]} | {d["max_selected_vs_uniform_paired_scramble_SE"]:.5f} |')
    lines+=['',f'Integration uses {protocol["offline_integration"]}; selected by development-only runtime profiling. Nonzero policy disagreement limits claims about continuous Bayes optimality. No integration size/prior changes followed confirmation outcomes.', '',
            '## Interpretation', '',
            'Neither predeclared gate passes. Increasing pilot size to four does not recover material allocation benefit for these frozen posterior policies. At fixed main8/lambda=.5, primary Jeffreys worsens MSE by 0.066%; the correctly benchmark-informed secondary has only 0.002% gain with an interval spanning zero. At fixed total24/m0=4, primary gains 0.083% and secondary gains 0.145%, far below the 5% criterion. True-p oracle headroom persists; this is not proof that all pilot-only methods are impossible.', '',
            'The negative fixed-main gain trend and modest matched-cost improvements do not support moving this allocator family to model training. Preserve the distribution and negative results. A further experiment requires a distinct hypothesis, rather than a winner-selected subset or more confirmation tuning. Primary fixed-total/m0=4 also has quadrature-policy disagreement, so its interval must remain conditional on this frozen approximation.', '',
            f'Audit PASS: {len(rows)} exact cells, {6*len(rows)} method evaluations; direct risk residual {maxdirect:.2e}, raw value bias {maxbias:.2e}. Counts remain positive and fixed before independent main sampling. CPU fit/evaluation time {summary["elapsed_seconds"]:.1f}s. The confirmation set is now consumed.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
