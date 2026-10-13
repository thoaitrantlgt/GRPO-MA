"""Freeze and evaluate a fixed amplitude-normalization feasibility probe."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cpu_gate.run import dump, save_csv
from .core import moments

OUT=Path('refine-logs/normalization-probe/20261011_run01')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve runs; choose a new directory')
    OUT.mkdir(parents=True)
    protocol=json.loads(Path('experiments/normalization_probe/protocol.json').read_text())
    protocol['frozen_utc']=datetime.now(timezone.utc).isoformat()
    records=json.loads(Path(protocol['source']).read_text())
    dump(OUT/'protocol.json',protocol); dump(OUT/'probability_vectors.json',records)
    rows=[]; start=time.perf_counter()
    p=np.array([r['p'] for r in records]); residual=0
    for total in protocol['total_budgets']:
        result=moments(p,total)
        for name,metric in result['methods'].items():
            for i,meta in enumerate(records):
                r={**meta,'total':total,'method':name,
                   **{key:float(value[i]) for key,value in metric.items()},
                   'dispersion':float(np.var(p[i])), 'advantage_energy':float(result['energy'][i]),
                   'target_alignment':float(result['alignment'][i])}
                r['relative_reduction_vs_plugin']=float((result['methods']['plugin']['mse'][i]-r['mse'])/result['methods']['plugin']['mse'][i])
                residual=max(residual,abs(r['mse']-r['bias_squared']-r['variance']))
                rows.append(r)
        print(f'Total{total}: {len(records)} vectors, {len(result["states"])} exact states each',flush=True)
    dump(OUT/'moments.json',rows)
    save_csv(OUT/'metrics.csv',[{k:v for k,v in r.items() if k!='p'} for r in rows])
    aggregates={}
    for total in protocol['total_budgets']:
        group=[r for r in rows if r['kind']=='consumed' and r['total']==total]
        aggregates[str(total)]={name:{key:float(np.mean([r[key] for r in group if r['method']==name]))
                                        for key in ['mse','bias_squared','variance','coefficient']}
                                for name in result['methods']}
        base=aggregates[str(total)]['plugin']['mse']
        for val in aggregates[str(total)].values():
            val['relative_reduction']=(base-val['mse'])/base
    stress=[r for r in rows if r['kind']=='stress' and r['method']=='fixed-half']
    violations=[{'id':r['id'],'total':r['total'],'relative_reduction':r['relative_reduction_vs_plugin']}
                for r in stress if r['relative_reduction_vs_plugin']<-.1]
    mean_pass=aggregates['16']['fixed-half']['relative_reduction']>.05 and all(aggregates[str(b)]['fixed-half']['relative_reduction']>0 for b in [8,32])
    decision='DEVELOPMENT_CRITERIA_MET_NOT_CONFIRMED' if mean_pass and not violations else 'REJECT_FIXED_HALF_AS_GENERAL_FOLLOWUP'
    summary={'decision':decision,'aggregates':aggregates,'stress_regressions_over10pct':violations,
             'mean_criteria_pass':mean_pass,'vectors':len(records),'method_evaluations':len(rows),
             'elapsed_seconds':time.perf_counter()-start,'GPU_cost':0,'fresh_confirmation':False}
    dump(OUT/'summary.json',summary)
    formula_residual=max(abs(r['mse']-(r['coefficient']**2*r['advantage_energy']-2*r['coefficient']*r['target_alignment']+float(np.mean(moments([r['p']],8)['target_energy'])))) for r in rows)
    if residual>1e-12 or formula_residual>1e-12:
        raise ArithmeticError('Risk identity failure')
    dump(OUT/'AUDIT.json',{'status':'PASS','bias_variance_residual':residual,'quadratic_risk_residual':formula_residual,
                         'reward_states':'complete finite enumeration','scope':'local numerical audit; no independent reviewer'})
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for direction in ['balanced','one-outlier','paired']:
        for method,ls in [('plugin','-'),('fixed-half','--')]:
            path=[r for r in rows if r['kind']=='path' and r['component']==direction and r['total']==16 and r['method']==method]
            axs[0].semilogx([r['dispersion']+1e-6 for r in path],[r['mse'] for r in path],ls=ls,label=direction+' '+method)
        path=[r for r in rows if r['kind']=='path' and r['component']==direction and r['total']==16 and r['method']=='fixed-half']
        axs[1].semilogx([r['dispersion']+1e-6 for r in path],[100*r['relative_reduction_vs_plugin'] for r in path],label=direction)
    axs[0].set_ylabel('Exact MSE against unchanged target'); axs[1].set_ylabel('Half-amplitude reduction (%)'); axs[1].axhline(0,color='black',lw=.7)
    for ax in axs:
        ax.set_xlabel('Dispersion + epsilon'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Fixed normalization-amplitude probe, K4 / total16, consumed-data diagnosis')
    fig.tight_layout(); fig.savefig(OUT/'normalization_tradeoff.png',dpi=180); fig.savefig(OUT/'normalization_tradeoff.pdf'); plt.close(fig)
    lines=['# Fixed normalization-amplitude CPU probe','', '**'+decision+'**. Consumed-data feasibility diagnosis only; no new confirmation interval. GPU cost0.', '',
        'The unchanged target is f_epsilon(p). Uniform pooled sampling spends all total8/16/32 samples. The only candidate is0.5*f_epsilon(V), fixed before these outcomes. Plug-in, zero, and true-p scalar oracle are controls. All96 consumed mixture vectors, six stress regimes and27 fixed dispersion paths are retained.', '',
        '## Exact aggregate risks: consumed mixture vectors','',
        '| Total | Estimator | MSE | Squared bias | Variance | Reduction vs plug-in |','|---|---|---:|---:|---:|---:|']
    for total, methods in aggregates.items():
        for name,r in methods.items():
            lines.append(f'| {total} | {name} | {r["mse"]:.6f} | {r["bias_squared"]:.6f} | {r["variance"]:.6f} | {100*r["relative_reduction"]:.2f}% |')
    lines+=['','## Every stress result for the fixed-half candidate','',
        '| Regime | Total | Plug-in MSE | Half MSE | Relative reduction |','|---|---:|---:|---:|---:|']
    for r in stress:
        base=next(x['mse'] for x in rows if x['id']==r['id'] and x['total']==r['total'] and x['method']=='plugin')
        lines.append(f'| {r["id"]} | {r["total"]} | {base:.6f} | {r["mse"]:.6f} | {100*r["relative_reduction_vs_plugin"]:.2f}% |')
    lines+=['','## Findings','',
        f'1. Observation: the frozen mean criteria {"pass" if mean_pass else "fail"}, but {len(violations)}/18 stress/budget cells regress by more than10%. Interpretation: finite-budget average benefit does not establish robust normalization. Implication: the complete frozen rule gives **{decision}**. Next step: do not deploy or confirm fixed-half based on a favorable mixture mean.', '',
        '2. Observation: half amplitude quarters variance at every fixed probability vector; bias changes as ||0.5*E[A]-target||²/K. Interpretation: this is an amplitude/bias tradeoff, not a correction of noisy advantage direction. Implication: improved scalar MSE alone can come from attenuating credits. Next step: a future design must specify whether the scientific objective is estimating the current normalized target or changing the target itself.', '',
        '3. Observation: at equality, half amplitude cuts risk75%, but the zero control has zero risk; at nonconstant deterministic rewards, plug-in is exact while half amplitude has positive error. Interpretation: a universal fixed-shrinkage benefit is false. Implication: no training gain or distinct algorithm follows; multiplying a fixed-score update by a constant overlaps learning-rate scaling. Next step: document the required credit/update objective before choosing any new normalization mechanism.', '',
        '## Frozen rejection rule','',protocol['rejection_rule']+'. No winner-based rescue, coefficient sweep or replacement candidate. The primary mean equally weights the unchanged six mixture components (16 vectors each); paths/stress cases are outside this mean. Exact reward enumeration removes outcome Monte Carlo error, not uncertainty over unseen probability vectors.', '',
        'Exact identity: R(c)=a*c²-2*b*c+d; R(1)-R(0.5)=0.75*a-b. Oracle scalar clip(b/a,0,1) requires true-p moments and is unavailable to a runtime estimator. See [derivation package](../../../DERIVATION_PACKAGE.md).', '',
        '[Raw metrics](metrics.csv) · [Protocol frozen before outcomes](protocol.json) · [Tradeoff plot](normalization_tradeoff.png) · [Numerical audit](AUDIT.json)', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
