"""Fit before evaluation access, then report exact consumed-data risks."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cpu_gate.run import dump,save_csv
from experiments.gradient_gate.core import grams,NAMES
from .core import fit,exact_metrics

OUT=Path('refine-logs/posterior-credit/20261011_run01')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve runs; choose a new directory')
    OUT.mkdir(parents=True)
    protocol=json.loads(Path('experiments/posterior_credit/protocol.json').read_text())
    protocol['frozen_utc']=datetime.now(timezone.utc).isoformat(); dump(OUT/'protocol.json',protocol)
    start=time.perf_counter(); table=fit(protocol['prior_fit']); fit_seconds=time.perf_counter()-start
    np.savez_compressed(OUT/'frozen_table.npz',**table)
    dump(OUT/'FIT_FREEZE.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'development_outcomes_read':False,
         'scalar_control':table['scalar'],'half_support_max_credit_difference':table['half_difference'],
         'base_points':protocol['prior_fit']['scrambles']*protocol['prior_fit']['points_per_scramble'],
         'symmetrized_support_entries':len(table['support']),'fit_seconds':fit_seconds,'GPU_cost':0})
    print(f'Prior table and scalar frozen: c={table["scalar"]:.6f}, fit{fit_seconds:.2f}s',flush=True)
    records=json.loads(Path(protocol['source']).read_text()); p=np.array([r['p'] for r in records])
    matrices=grams(); project=np.eye(4)-np.ones((4,4))/4
    matrices=np.einsum('ij,gjk,kl->gil',project,matrices,project); matrices[1]=0
    begin=time.perf_counter(); metrics=exact_metrics(p,table,matrices); rows=[]
    for i,meta in enumerate(records):
        for g,name in enumerate(NAMES):
            for method,values in metrics.items():
                row={**meta,'method':method,'gram':name,**{key:float(value[i,g]) for key,value in values.items()}}
                row['dispersion']=float(np.var(p[i])); mass=row['direction_defined_mass']
                row['conditional_direction_cosine']=row['direction_cosine_numerator']/mass if mass>1e-12 else None
                rows.append(row)
    dump(OUT/'moments.json',rows); save_csv(OUT/'metrics.csv',[{k:v for k,v in r.items() if k!='p'} for r in rows])
    aggregates={}
    for method in metrics:
        group=[r for r in rows if r['kind']=='consumed' and r['gram'] in protocol['primary_grams'] and r['method']==method]
        aggregates[method]={key:float(np.mean([r[key] for r in group])) for key in ['mse','bias_squared','variance']}
    comparisons={base:(aggregates[base]['mse']-aggregates['posterior-credit']['mse'])/aggregates[base]['mse'] for base in ['plugin','prior-scale']}
    violations=[]; stress_rows=[]
    for r in rows:
        if r['kind']=='stress' and r['gram'] in protocol['primary_grams'] and r['method']=='posterior-credit':
            base=next(x['mse'] for x in rows if x['id']==r['id'] and x['gram']==r['gram'] and x['method']=='plugin')
            reduced=(base-r['mse'])/base; stress_rows.append({**r,'reduction':reduced})
            if reduced<-.1:
                violations.append({'id':r['id'],'gram':r['gram'],'relative_reduction':reduced})
    mean_pass=all(x>.05 for x in comparisons.values()); stable=table['half_difference']<=.02
    decision='DEVELOPMENT_CRITERIA_MET_NOT_CONFIRMED' if mean_pass and stable and not violations else 'STOP_POSTERIOR_CREDIT_PROTOTYPE'
    # Finite-support Bayes audit, independent of the development population.
    prior_risks={}
    for method,credit in [('plugin',table['plugin']),('prior-scale',table['scalar']*table['plugin']),('posterior-credit',table['credit'])]:
        prior_risks[method]=np.einsum('gij,ji->g',matrices,table['second'])+np.sum(table['mass'][:,None]*np.einsum('hi,gij,hj->hg',credit,matrices,credit),axis=0)-2*np.sum(np.einsum('hi,gij,hj->hg',credit,matrices,table['numerator']),axis=0)
    bayes_excess=max(float(np.max(prior_risks['posterior-credit']-r)) for r in prior_risks.values())
    residual=max(abs(r['mse']-r['bias_squared']-r['variance']) for r in rows)
    center=float(np.max(np.abs(table['credit'].sum(1))))
    lookup={tuple(s):i for i,s in enumerate(table['states'])}; perm=[2,0,3,1]
    symmetry=max(float(np.max(np.abs(table['credit'][lookup[tuple(s[perm])]]-table['credit'][i,perm]))) for i,s in enumerate(table['states']))
    null_max=max(r['mse'] for r in rows if r['gram']=='aligned-null')
    audit={'status':'PASS','bias_variance_residual':residual,'centering_residual':center,'exchangeability_residual':symmetry,
           'finite_prior_bayes_max_excess':bayes_excess,'predictive_mass_sum':float(table['mass'].sum()),'null_update_max':null_max,
           'scope':'local audit; exact finite-support Bayes control, approximate continuous prior; synthetic fixed scores only'}
    if max(residual,center,symmetry,bayes_excess,null_max,abs(table['mass'].sum()-1))>1e-10:
        raise ArithmeticError(audit)
    dump(OUT/'AUDIT.json',audit)
    summary={'decision':decision,'aggregates':aggregates,'candidate_reduction_vs':comparisons,'mean_criteria_pass':mean_pass,
             'quadrature_stability_pass':stable,'half_support_max_credit_difference':table['half_difference'],
             'stress_regressions_over10pct':violations,'scalar_control':table['scalar'],'vectors':len(records),
             'method_evaluations':len(rows),'fit_seconds':fit_seconds,'evaluation_seconds':time.perf_counter()-begin,
             'GPU_cost':0,'fresh_confirmation':False}
    dump(OUT/'summary.json',summary)
    lines=['# Posterior-credit prototype: consumed-data CPU diagnosis','', '**'+decision+'**. No fresh confirmation or training claim. GPU cost0.', '',
           'K4/total16, uniform four reward samples per thought. The target is sample-std normalized true Bernoulli values with floor1e-4. Prior Jeffreys iid Beta(.5,.5); candidate is the posterior expectation of that target, not normalization of the posterior mean. All96 consumed mixture vectors, six stress regimes,27 fixed dispersion probes and seven frozen synthetic Grams retained.', '',
           f'Prior-scale control coefficient{table["scalar"]:.6f} was fitted from the prior alone and frozen before development risk access. Continuous integration uses four512-point Sobol supports, each augmented with all24 permutations ({len(table["support"])} support entries,2048 base points). Half-support maximum table difference{table["half_difference"]:.6f}; tolerance0.02.', '',
           '| Four-Gram / consumed96 | Exact update MSE | Squared bias | Variance | Posterior reduction against row |','|---|---:|---:|---:|---:|']
    for method,val in aggregates.items():
        relative=comparisons.get(method,0)
        lines.append(f'| {method} | {val["mse"]:.6f} | {val["bias_squared"]:.6f} | {val["variance"]:.6f} | {100*relative:.2f}% |')
    lines+=['','## All stress results, four primary fixed Grams','', '| Regime | Gram | Posterior reduction vs plug-in |','|---|---|---:|']
    for r in stress_rows:
        lines.append(f'| {r["id"]} | {r["gram"]} | {100*r["reduction"]:.2f}% |')
    lines+=['','## Findings','',
        f'1. Observation: mean criteria {"pass" if mean_pass else "fail"}; {len(violations)}/24 primary stress cells regress more than10%; quadrature criterion {"passes" if stable else "fails"}. Interpretation: posterior integrated optimality does not imply pointwise robustness under a different population. Implication: the complete frozen rule yields **{decision}**. Next step: do not rescue a failed prototype with a different prior, budget or favorable geometry.', '',
        '2. Observation: the frozen finite-prior posterior mean dominates both controls in integrated fixed-Gram risk. Interpretation: this verifies the conditional-expectation identity under its finite model; it does not certify the continuous prior or benchmark superiority. Implication: separate numerical validity from empirical gate success. Next step: keep this distinction in future research claims.', '',
        '3. Observation: credits use count histories only; uniform sampling and target remain fixed. Interpretation: this is a credit-estimation test, distinct from the prior allocator tests. Implication: no inference about adaptive allocation, measured score geometry, clipping or online learning follows. Next step: revisit the information model only if an independently justified design is proposed; no automatic follow-up run.', '',
        'Primary metrics average all four predeclared nontrivial Grams within each vector and equal-weight all six mixture components. No fresh intervals on consumed data. Direction/alignment/energy metrics, including direction-defined probability mass, are exported; null update energy is retained as an absolute metric. A cosine conditional on nonzero norms is not a replacement for total risk.', '',
        f'Local audit PASS: bias/variance residual{residual:.3g}, centering{center:.3g}, symmetry{symmetry:.3g}; finite-prior Bayes maximum excess{bayes_excess:.3g}. Runtime table625 rows; exact outcome-state enumeration. Continuous prior approximation remains separate from exact evaluation of the frozen table.', '',
        '[Design and derivation](../../../experiments/posterior_credit/DESIGN.md) · [Frozen protocol](protocol.json) · [Raw metrics](metrics.csv) · [Frozen estimator](frozen_table.npz)', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(3); names=list(aggregates)
    ax.bar(x,[aggregates[n]['mse'] for n in names]); ax.set_xticks(x,names); ax.set_ylabel('Mean exact fixed-Gram update MSE')
    ax.set_title('Consumed96 / K4 total16 / four predeclared synthetic Grams'); fig.tight_layout()
    fig.savefig(OUT/'posterior_credit.png',dpi=180); fig.savefig(OUT/'posterior_credit.pdf'); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
