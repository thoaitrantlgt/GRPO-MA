"""Freeze descriptive paths, run exact CPU diagnosis, and export evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cpu_gate.run import dump, save_csv
from .core import diagnose

OUT = Path('refine-logs/surrogate-diagnostic/20261011_run02')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous run; choose a new output directory')
    OUT.mkdir(parents=True)
    protocol = json.loads(Path('experiments/surrogate_diagnostic/protocol.json').read_text())
    protocol['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    records = json.loads(Path(protocol['source']).read_text())
    for r in records:
        r['kind'] = 'consumed' if r['kind']=='confirmation' else 'stress'
        r['amplitude'] = None
    for name, direction in protocol['directions'].items():
        for amp in protocol['amplitudes']:
            records.append({'id':f'path-{name}-{amp}','kind':'path','component':name,'amplitude':amp,
                            'p':(.5+amp*np.array(direction)).tolist()})
    dump(OUT/'protocol.json',protocol); dump(OUT/'probability_vectors.json',records)
    start = time.perf_counter(); rows=[]; rank_rows=[]; residuals=[]
    for setting in protocol['settings']:
        for begin in range(0,len(records),16):
            batch=records[begin:begin+16]; p=np.array([r['p'] for r in batch])
            metrics, result = diagnose(p,**setting)
            for r in metrics:
                meta=batch[r.pop('index')]; z=np.array(meta['p'])-np.mean(meta['p'])
                r.update(meta); r.update(setting)
                r['pilot_fraction']=4*setting['m0']/setting['total']
                r['max_leverage']=float(np.max(z*z)/(4*r['dispersion'])) if r['dispersion'] else 0
                rows.append(r)
                residuals.append(abs(r['mse']-r['linear']-r['cross']-r['remainder']))
            from experiments.cpu_gate.estimator import column_norms, EPS
            for i, meta in enumerate(batch):
                fixed = result['unconditional_risks'][i]; candidates=result['allocations']
                lam=4*setting['m0']/setting['total']; q=p[i]
                sur=np.sum(column_norms(q)*q*(1-q)*(lam**2/setting['m0']+(1-lam)**2/candidates),axis=1)/4
                local=result['selected']['local-surrogate-oracle'][i,0]; opt=int(fixed.argmin())
                # Accept any mathematical surrogate minimizer, including deterministic ties.
                if sur[local]>sur.min()+1e-9:
                    raise ArithmeticError('Separable allocation failed exhaustive surrogate optimum')
                regret=float((fixed[local]-fixed[opt])/fixed[opt])
                calibration=float(abs(sur[local]-fixed[local])/fixed[local])
                rank_rows.append({**meta,**setting,'dispersion':float(np.var(q)),
                    'local_allocation':candidates[local].tolist(),'exact_allocation':candidates[opt].tolist(),
                    'exact_local_mse':float(fixed[local]),'exact_optimal_mse':float(fixed[opt]),
                    'fixed_regret':regret,'local_calibration_error':calibration,
                    'passes_descriptive_tolerances':bool(regret<=protocol['descriptive_fixed_allocation_regret_tolerance'] and calibration<=protocol['descriptive_calibration_tolerance'])})
            print(f'Exact diagnosis {begin+1}-{min(begin+16,len(records))}/{len(records)}',flush=True)
    dump(OUT/'moments.json',rows); dump(OUT/'allocation_diagnosis.json',rank_rows)
    save_csv(OUT/'moments.csv',[{k:v for k,v in r.items() if k!='p'} for r in rows])
    save_csv(OUT/'allocation_diagnosis.csv',[{k:v for k,v in r.items() if k!='p'} for r in rank_rows])
    audit={'status':'PASS','max_taylor_identity_residual':max(residuals),
           'max_linear_formula_residual':max(abs(r['linear']-r['surrogate']) for r in rows),
           'max_raw_bias':max(r['raw_bias_max'] for r in rows),
           'max_linear_mean':max(r['linear_mean_max'] for r in rows),
           'max_oracle_excess':max(r['exact_optimal_mse']-r['exact_local_mse'] for r in rank_rows)}
    if audit['max_taylor_identity_residual']>1e-8 or audit['max_linear_formula_residual']>1e-8 or audit['max_raw_bias']>1e-12:
        raise ArithmeticError(audit)
    dump(OUT/'AUDIT.json',audit)
    aggregates={}
    for method in protocol['methods']:
        group=[r for r in rows if r['kind']=='consumed' and r['method']==method]
        aggregates[method]={key:float(np.mean([r[key] for r in group])) for key in ['mse','bias_squared','variance','linear','cross','remainder']}
        aggregates[method]['bias_share']=aggregates[method]['bias_squared']/aggregates[method]['mse']
    consumed=[r for r in rank_rows if r['kind']=='consumed']
    summary={'scope':protocol['scope'],'consumed_vectors':len(consumed),'stress_vectors':6,'path_vectors':27,
             'method_evaluations':len(rows),'elapsed_seconds':time.perf_counter()-start,
             'aggregates':aggregates,'median_fixed_regret':float(np.median([r['fixed_regret'] for r in consumed])),
             'calibration_and_regret_pass_count':sum(r['passes_descriptive_tolerances'] for r in consumed),
             'GPU_cost':0,'fresh_confirmation':False}
    dump(OUT/'summary.json',summary)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for name in protocol['directions']:
        path=[r for r in rows if r['kind']=='path' and r['component']==name and r['method']=='uniform']
        x=[r['dispersion']+1e-6 for r in path]
        axs[0].loglog(x,[r['mse'] for r in path],label=name+' exact')
        axs[0].loglog(x,[r['surrogate'] for r in path],ls='--',label=name+' local')
        axs[1].semilogx(x,[r['bias_squared']/r['mse'] for r in path],label=name)
        allocation=[r for r in rank_rows if r['kind']=='path' and r['component']==name]
        axs[2].semilogx([r['dispersion']+1e-6 for r in allocation],[100*r['fixed_regret'] for r in allocation],label=name)
    axs[0].set_ylabel('Normalized-advantage MSE'); axs[1].set_ylabel('Squared bias / MSE'); axs[2].set_ylabel('Local allocation regret (%)')
    for ax in axs:
        ax.set_xlabel('Group dispersion + epsilon'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Exact descriptive diagnosis: K4, total16, m0=2, fixed reuse0.5')
    fig.tight_layout(); fig.savefig(OUT/'surrogate_diagnosis.png',dpi=180); fig.savefig(OUT/'surrogate_diagnosis.pdf'); plt.close(fig)
    lines=['# Exact surrogate mechanism diagnosis','',protocol['scope']+'. No fresh intervals or training claim; CPU only.', '',
           'K4 / total16 / m0=2 / budget-precommitted reuse0.5. Six stress regimes, 96 consumed vectors from the unchanged predeclared mixture, and all 27 deterministic dispersion-path points. Paths are probes, not a selected benchmark distribution.', '',
           '| Method | Exact MSE | Squared bias | Variance | Linear surrogate | Bias share |', '|---|---:|---:|---:|---:|---:|']
    for name,a in aggregates.items():
        lines.append(f'| {name} | {a["mse"]:.6f} | {a["bias_squared"]:.6f} | {a["variance"]:.6f} | {a["linear"]:.6f} | {100*a["bias_share"]:.2f}% |')
    lines+=['','## Decomposition and interpretation','',
        'Let delta=f(V)-f(p), L=J(p)(V-p), R=delta-L. Exact MSE = squared bias + variance = E||L||²/K + 2E[LᵀR]/K + E||R||²/K. The raw estimate and L are unbiased under fixed reuse even with pilot-dependent allocation. Normalized advantage can be biased.', '',
        '| Method | Linear | Cross term | Remainder energy | Exact |','|---|---:|---:|---:|---:|']
    for name,a in aggregates.items():
        lines.append(f'| {name} | {a["linear"]:.6f} | {a["cross"]:.6f} | {a["remainder"]:.6f} | {a["mse"]:.6f} |')
    lines+=['',f'Both diagnostic tolerances (risk calibration within10%, fixed allocation regret at most1%) hold for {summary["calibration_and_regret_pass_count"]}/{len(consumed)} consumed vectors. Median fixed local-allocation regret is {100*summary["median_fixed_regret"]:.3f}%. These are descriptive tolerances, not sufficient mathematical conditions or a population guarantee.', '',
        'At equal probabilities, the target advantage and symmetric uniform-estimator bias are zero. Exact risk stays below1, whereas J scales as epsilon^(-1/2) and its surrogate can be enormous. Thus failed linearization can occur with zero bias. For other vectors, bias and curvature both contribute; attributing the full discrepancy to bias alone is incorrect.', '',
        'The local expansion requires errors small relative to sqrt(dispersion+epsilon), with control of tails. The exported centered-noise/dispersion ratio measures typical scale, but does not certify tails or establish a universal threshold. Leverage and all deterministic paths are retained without selecting favorable directions.', '',
        '## Dispersion bins: consumed vectors only','',
        '| Dispersion interval | Vectors | Mean local fixed regret | Joint tolerance passes |','|---|---:|---:|---:|']
    edges=protocol['dispersion_bins']
    for low,high in zip(edges[:-1],edges[1:]):
        group=[r for r in consumed if low<=r['dispersion']<high]
        if group:
            lines.append(f'| [{low}, {high}) | {len(group)} | {100*np.mean([r["fixed_regret"] for r in group]):.3f}% | {sum(r["passes_descriptive_tolerances"] for r in group)}/{len(group)} |')
    lines+=['','## Stress regimes','', '| Regime | Uniform exact | Uniform surrogate | Uniform bias share | Local fixed regret |','|---|---:|---:|---:|---:|']
    for meta in records[:6]:
        r=next(r for r in rows if r['id']==meta['id'] and r['method']=='uniform'); a=next(r for r in rank_rows if r['id']==meta['id'])
        lines.append(f'| {r["id"]} | {r["mse"]:.6f} | {r["surrogate"]:.6f} | {100*r["bias_squared"]/r["mse"]:.2f}% | {100*a["fixed_regret"]:.3f}% |')
    lines+=['',f'Audit PASS: Taylor residual {audit["max_taylor_identity_residual"]:.3g}; linear/covariance formula residual {audit["max_linear_formula_residual"]:.3g}; raw bias {audit["max_raw_bias"]:.3g}. All35 positive main allocations enumerated; allocation optimum checked against complete candidate list.', '',
            '[Figure](surrogate_diagnosis.png) · [Raw moments](moments.csv) · [Allocation diagnostics](allocation_diagnosis.csv) · [Frozen protocol](protocol.json)', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
