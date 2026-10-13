"""Predeclare a finite quadrature ladder and diagnose every retained outcome."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cpu_gate.run import dump,save_csv
from experiments.gradient_gate.core import grams,NAMES
from .core import exact_metrics
from .convergence import integrate

OUT=Path('refine-logs/posterior-credit-convergence/20261011_run01')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume-frozen',action='store_true'); args=parser.parse_args()
    if OUT.exists() and not args.resume_frozen:
        raise FileExistsError('Preserve existing run; choose a new directory')
    if args.resume_frozen:
        if (OUT/'summary.json').exists():
            raise FileExistsError('Completed results may not be overwritten')
        protocol=json.loads((OUT/'protocol.json').read_text())
    else:
        OUT.mkdir(parents=True)
        protocol=json.loads(Path('experiments/posterior_credit/convergence_protocol.json').read_text())
        protocol['frozen_utc']=datetime.now(timezone.utc).isoformat()
    original=Path(protocol['original_run'])
    original_protocol=json.loads((original/'protocol.json').read_text())
    original_summary=json.loads((original/'summary.json').read_text())
    protocol['fixed_scalar_control']=original_summary['scalar_control']
    protocol['original_protocol']=original_protocol
    if args.resume_frozen:
        tables={}
        for level in protocol['nested_points_per_scramble']:
            with np.load(OUT/f'table_n{level}.npz') as saved:
                tables[level]={key:saved[key] for key in saved.files}
            for key in ['scalar','half_difference']:
                tables[level][key]=float(tables[level][key])
        fit_seconds=json.loads((OUT/'TABLES_FREEZE.json').read_text())['fit_seconds']
    else:
        dump(OUT/'protocol.json',protocol)
        start=time.perf_counter()
        tables,points=integrate(protocol,lambda rep:print(f'Nested quadrature: scramble{rep}/{protocol["scrambles"]}, elapsed{time.perf_counter()-start:.1f}s',flush=True))
        fit_seconds=time.perf_counter()-start
        np.savez_compressed(OUT/'prior_base_points.npz',points=points)
        for level,table in tables.items():
            table['scalar']=protocol['fixed_scalar_control']
            np.savez_compressed(OUT/f'table_n{level}.npz',**table)
        dump(OUT/'TABLES_FREEZE.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'new_tables_frozen_before_risk_evaluation':True,
             'levels':list(tables),'scalar_unchanged':protocol['fixed_scalar_control'],'fit_seconds':fit_seconds,'GPU_cost':0})
    records=json.loads(Path(original_protocol['source']).read_text())
    original_rows=json.loads((original/'moments.json').read_text())
    original_vectors={r['id']:r['p'] for r in original_rows}
    if {r['id']:r['p'] for r in records}!=original_vectors:
        raise ArithmeticError('Source vectors differ from the consumed original population')
    dump(OUT/'probability_vectors.json',records)
    p=np.array([r['p'] for r in records]); matrices=grams(); project=np.eye(4)-np.ones((4,4))/4
    matrices=np.einsum('ij,gjk,kl->gil',project,matrices,project); matrices[1]=0
    begin=time.perf_counter(); diagnostics=[]; rows=[]; stress=[]; previous=None; risk_previous=None
    levels=protocol['nested_points_per_scramble']; residual=0; bayes_excess=0; center=0
    for level in levels:
        table=tables[level]; metrics=exact_metrics(p,table,matrices)
        for i,meta in enumerate(records):
            for g,name in enumerate(NAMES):
                for method,values in metrics.items():
                    row={**meta,'level':level,'method':method,'gram':name,
                         **{key:float(value[i,g]) for key,value in values.items()}}
                    rows.append(row); residual=max(residual,abs(row['mse']-row['bias_squared']-row['variance']))
        selected=[r for r in rows if r['level']==level and r['kind']=='consumed' and r['gram'] in original_protocol['primary_grams']]
        means={method:float(np.mean([r['mse'] for r in selected if r['method']==method])) for method in metrics}
        comparisons={base:(means[base]-means['posterior-credit'])/means[base] for base in ['plugin','prior-scale']}
        current_stress=[]; violations=[]
        for i,meta in enumerate(records):
            if meta['kind']!='stress':
                continue
            for g,name in enumerate(NAMES):
                if name not in original_protocol['primary_grams']:
                    continue
                baseline=metrics['plugin']['mse'][i,g]; candidate=metrics['posterior-credit']['mse'][i,g]
                reduction=(baseline-candidate)/baseline
                entry={'level':level,'id':meta['id'],'gram':name,'plugin_mse':float(baseline),
                       'posterior_mse':float(candidate),'relative_reduction':float(reduction)}
                stress.append(entry); current_stress.append(entry)
                if reduction<-.1:
                    violations.append(entry)
        stress_change=None if risk_previous is None else max(abs(a['posterior_mse']-b['posterior_mse'])/a['plugin_mse'] for a,b in zip(current_stress,risk_previous))
        credit_change=None if previous is None else float(np.max(np.abs(table['credit']-previous['credit'])))
        diagnostic={'points_per_scramble':level,'base_points':level*protocol['scrambles'],
                    'implicit_symmetrized_entries':level*protocol['scrambles']*24,
                    'half_difference':table['half_difference'],'max_credit_change_vs_previous':credit_change,
                    'max_stress_risk_change_vs_previous_normalized':stress_change,
                    'mean_risks':means,'relative_reductions':comparisons,'stress_regressions':violations,
                    'mean_criteria_pass':all(x>.05 for x in comparisons.values())}
        diagnostics.append(diagnostic)
        previous=table; risk_previous=current_stress
        center=max(center,float(np.max(np.abs(table['credit'].sum(1)))))
        integrated={}
        for method,credit in [('plugin',table['plugin']),('prior-scale',table['scalar']*table['plugin']),('posterior-credit',table['credit'])]:
            integrated[method]=np.einsum('gij,ji->g',matrices,table['second'])+np.sum(table['mass'][:,None]*np.einsum('hi,gij,hj->hg',credit,matrices,credit),axis=0)-2*np.sum(np.einsum('hi,gij,hj->hg',credit,matrices,table['numerator']),axis=0)
        bayes_excess=max(bayes_excess,max(float(np.max(integrated['posterior-credit']-value)) for value in integrated.values()))
        print(f'Level{level}: half-credit difference{table["half_difference"]:.6f}; stress failures{len(violations)}/24',flush=True)
    # First four original seed/prefix streams reproduce the frozen prototype.
    first=tables[levels[0]]
    reconstructed=first['replicate_numerator'][:4].mean(0)/first['replicate_mass'][:4].mean(0)[:,None]
    with np.load(original/'frozen_table.npz') as old:
        reproduction=float(np.max(np.abs(reconstructed-old['credit'])))
    audit={'status':'PASS','bias_variance_residual':residual,'centering_residual':center,
           'finite_prior_bayes_max_excess':bayes_excess,'original_first4_prefix_reproduction_residual':reproduction,
           'predictive_mass_residual':max(abs(t['mass'].sum()-1) for t in tables.values()),
           'constant_scalar_preserved':all(t['scalar']==protocol['fixed_scalar_control'] for t in tables.values()),
           'null_update_max':max(r['mse'] for r in rows if r['gram']=='aligned-null'),
           'all_declared_levels_evaluated':list(tables)==levels}
    if max(residual,center,bayes_excess,reproduction,audit['predictive_mass_residual'],audit['null_update_max'])>1e-10 or not audit['constant_scalar_preserved']:
        raise ArithmeticError(audit)
    dump(OUT/'AUDIT.json',audit)
    high=diagnostics[-1]
    stable=high['half_difference']<=.02 and high['max_credit_change_vs_previous']<=.01 and high['max_stress_risk_change_vs_previous_normalized']<=.01
    decision=('NUMERICALLY_INCONCLUSIVE' if not stable else
              'STOP_POSTERIOR_CREDIT_AFTER_CONVERGENCE' if not high['mean_criteria_pass'] or high['stress_regressions'] else
              'DEVELOPMENT_CRITERIA_MET_NOT_CONFIRMED')
    summary={'decision':decision,'highest_level_stable':stable,'diagnostics':diagnostics,'fit_seconds':fit_seconds,
             'evaluation_seconds':time.perf_counter()-begin,'method_evaluations':len(rows),
             'fresh_confirmation':False,'GPU_cost':0,'limits':'numerical stability diagnostic only, no rigorous continuous-prior error certificate or fresh inference'}
    dump(OUT/'summary.json',summary); dump(OUT/'moments.json',rows)
    save_csv(OUT/'metrics.csv',[{k:v for k,v in r.items() if k!='p'} for r in rows]); save_csv(OUT/'stress.csv',stress)
    lines=['# Posterior-credit numerical convergence','', '**'+decision+'**. All original model choices retained; no fresh confirmation, GPU cost0.', '',
           'Eight independent Sobol scrambles, same original seed+replicate streams, nested512/2048/8192/32768 base points each. Symmetrizing joint integrals is algebraically equivalent to24-permutation support augmentation; tested against direct augmentation. Joint numerators/masses are pooled before division. Original prior-scale coefficient remains fixed.', '',
           '| Points/scramble | Half max credit difference | Adjacent max credit change | Max stress risk change / plug-in | Reduction vs plug-in | Reduction vs scale control | Stress failures |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for d in diagnostics:
        change='—' if d['max_credit_change_vs_previous'] is None else f'{d["max_credit_change_vs_previous"]:.6f}'
        riskchange='—' if d['max_stress_risk_change_vs_previous_normalized'] is None else f'{100*d["max_stress_risk_change_vs_previous_normalized"]:.4f}%'
        lines.append(f'| {d["points_per_scramble"]} | {d["half_difference"]:.6f} | {change} | {riskchange} | {100*d["relative_reductions"]["plugin"]:.3f}% | {100*d["relative_reductions"]["prior-scale"]:.3f}% | {len(d["stress_regressions"])}/24 |')
    lines+=['','## Highest-level stress results','', '| Regime | Gram | Posterior reduction vs plug-in |','|---|---|---:|']
    for r in stress:
        if r['level']==levels[-1]:
            lines.append(f'| {r["id"]} | {r["gram"]} | {100*r["relative_reduction"]:.3f}% |')
    lines+=['','## Findings','',
        f'1. Observation: largest-level numerical stability {"passes" if stable else "fails"} all three frozen tolerances. Interpretation: this is finite numerical evidence of reproducibility, not an error bound for the continuous integral. Implication: {decision}. Next step: retain the ceiling and decision; no adaptive quadrature extension or favorable-level selection.', '',
        f'2. Observation: highest-level mean criteria {"pass" if high["mean_criteria_pass"] else "fail"}, with {len(high["stress_regressions"])}/24 stress failures. Interpretation: an integrated Bayes advantage need not hold pointwise across probability regimes. Implication: means cannot rescue a stress failure. Next step: apply the unchanged development contract before considering any fresh confirmation; none is started here.', '',
        '3. Observation: first four lowest-level streams reproduce the original prototype; exact finite-prior Bayes and moment identities pass. Interpretation: efficient integral symmetrization preserves the mathematical estimator. Implication: differences across levels are quadrature changes, not a prior/target/budget/geometry or scalar-control change. Next step: keep raw tables and all levels for reproducibility.', '',
        f'Local audit PASS; original table reproduction residual{reproduction:.3g}. Prior fit{fit_seconds:.2f}s CPU; evaluation{summary["evaluation_seconds"]:.2f}s. {len(rows)} exact method/vector/Gram evaluations. All129 consumed/stress/path vectors and seven geometries retained. No population intervals on consumed data; local verification only.', '',
        'The earlier prototype STOP remains in its original report. This diagnosis provides a separate numerical recommendation and is not a revised confirmation gate. Synthetic fixed score geometry only; no clipping/full-update, measured score, training, or novelty claim.', '',
        '[Frozen convergence protocol](protocol.json) · [All metrics](metrics.csv) · [All stress/level outcomes](stress.csv) · [Numerical audit](AUDIT.json)', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8)); x=[d['base_points'] for d in diagnostics]
    axes[0].loglog(x,[d['half_difference'] for d in diagnostics],marker='o',label='half-scramble max credit difference')
    axes[0].axhline(.02,ls='--',color='gray',label='fixed half tolerance'); axes[0].set_ylabel('Credit difference'); axes[0].legend(fontsize=7)
    target_stress=[r for r in stress if r['id']=='sparse-extreme' and r['gram']=='paired-correlated']
    axes[1].semilogx(x,[-100*r['relative_reduction'] for r in target_stress],marker='o',label='Original failed stress cell')
    axes[1].axhline(10,ls='--',color='gray',label='fixed regression threshold'); axes[1].set_ylabel('Risk worsening vs plug-in (%)'); axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel('Total base prior points, eight scrambles'); ax.grid(alpha=.2)
    fig.suptitle('Posterior-credit convergence: fixed prior, target, budget and scale control')
    fig.tight_layout(); fig.savefig(OUT/'convergence.png',dpi=180); fig.savefig(OUT/'convergence.pdf'); plt.close(fig)
    print(json.dumps({'decision':decision,'highest':high,'fit_seconds':fit_seconds},indent=2),flush=True)


if __name__=='__main__':
    main()
