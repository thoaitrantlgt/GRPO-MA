"""CPU feasibility entry point; smoke files never represent results."""
import argparse
import json
from pathlib import Path
import csv
import gc
import hashlib
import time
import itertools
import shutil
from datetime import datetime, timezone
import numpy as np
from .core import RiskModel, support
from .evaluation import exact_sequential, exact_fixed, paired_mc, average_permutations


OUT=Path('refine-logs/sequential-feasibility/20261011_run02')


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows,records,protocol):
    methods=protocol['methods'];configs=[(k,m) for k in protocol['K'] for m in protocol['m0']]
    consumed=[i for i,x in enumerate(records) if x['kind']=='consumed']
    arr=np.empty((len(configs),len(records),len(methods)))
    for r in rows:arr[configs.index((r['K'],r['m0'])),r['vector_index'],methods.index(r['method'])]=r['risk']
    rng=np.random.default_rng(protocol['bootstrap']['seed'])
    strata=[[i for i in consumed if records[i]['component']==str(c)] for c in range(6)]
    draws=np.array([np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata])
                    for _ in range(protocol['bootstrap']['replicates'])])
    effects=[]
    for ci,(k,m0) in enumerate(configs):
        base=arr[ci,consumed,0].mean()
        for mi,name in enumerate(methods):
            effect=1-arr[ci,consumed,mi].mean()/base
            boot=1-arr[ci,draws,mi].mean(axis=1)/arr[ci,draws,0].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            stress=[i for i,x in enumerate(records) if x['kind']=='stress']
            regression=np.max(arr[ci,stress,mi]/arr[ci,stress,0]-1)
            effects.append({'K':k,'B':16,'m0':m0,'pilot_fraction':k*m0/16,'method':name,
                            'reduction':float(effect),'development_ci_low':float(lo),'development_ci_high':float(hi),
                            'max_stress_regression':float(regression)})
    mi=methods.index('gated');base=arr[:,consumed,0].mean();gain=1-arr[:,consumed,mi].mean()/base
    boot=1-arr[:,draws,mi].mean(axis=(0,2))/arr[:,draws,0].mean(axis=(0,2))
    lo,hi=np.quantile(boot,[.025,.975])
    k4stress=max(x['max_stress_regression'] for x in effects if x['K']==4 and x['method']=='gated')
    verdict='DEVELOPMENT_SIGNAL_ONLY' if gain>.05 and k4stress<=.10 else 'NO_DEVELOPMENT_SIGNAL'
    return effects,{'verdict':verdict,'primary_reduction':float(gain),'development_ci':[float(lo),float(hi)],
                   'maximum_original_K4_stress_regression':float(k4stress),'confirmation_GO':False,'GPU_cost':0}


def write_report(summary,effects,diagnostics,rows):
    lines=['# Sequential sampling: CPU feasibility results','',
           '**Status: consumed-data development only. No fresh confirmation or GPU.**','',
           f"Verdict: **{summary['verdict']}**. Aggregate gated reduction: {100*summary['primary_reduction']:.4f}%; descriptive development 95% interval [{100*summary['development_ci'][0]:.4f}%, {100*summary['development_ci'][1]:.4f}%].",'',
           'The intervals resample already-consumed vectors; they cannot establish a fresh generalization claim. Reward-state risks are exact for each frozen implemented policy; posterior integration remains approximate.','',
           '| K | B | m0 | Pilot fraction | Method | Reduction vs uniform | Development interval | Maximum stress regression |',
           '|---:|---:|---:|---:|---|---:|---|---:|']
    for r in effects:
        lines.append(f"| {r['K']} | 16 | {r['m0']} | {r['pilot_fraction']:.3f} | {r['method']} | {100*r['reduction']:.4f}% | [{100*r['development_ci_low']:.4f}%, {100*r['development_ci_high']:.4f}%] | {100*r['max_stress_regression']:.4f}% |")
    lines+=['','K2 stress vectors are coordinate projections. The original six-regime stress gate refers to K4.','',
            '## Acquisition and runtime','',
            '| K | m0 | Method | Preferred observations / remaining budget | Numerical fallback | Reliability fallback | Exact-evaluation seconds |',
            '|---:|---:|---|---:|---:|---:|---:|']
    for d in diagnostics:
        lines.append(f"| {d['K']} | {d['m0']} | {d['method']} | {100*d.get('preferred_fraction',0):.4f}% | {100*d.get('numerical_fallback_fraction',0):.4f}% | {100*d.get('reliability_fallback_fraction',0):.4f}% | {d['seconds']:.3f} |")
    lines+=['','See [metrics.csv](metrics.csv), [effects.csv](effects.csv), [acquisition.json](acquisition.json), [paired_mc.csv](paired_mc.csv), [protocol.json](protocol.json), and [summary.json](summary.json).','',
            '## Decision limits','',
            'A fallback-dominated policy can reproduce uniform without demonstrating that sequential acquisition has no potential. The two small nested quadrature levels are a feasibility test; disagreement must be reported, not hidden by fallback. No prior, reliability threshold, benchmark distribution or terminal estimator was tuned on outcomes.','',
            'The trial excludes the full 26-configuration sweep, true-p nonlinear oracle, historical reuse factorial, synthetic gradient validation and training. Those stages are not counted as completed.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def evaluate():
    if OUT.exists():raise FileExistsError('Preserve completed/partial runs; choose a new output path')
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True)
    protocol['frozen_utc']=datetime.now(timezone.utc).isoformat()
    protocol['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    dump(OUT/'protocol.json',protocol)
    (OUT/'code').mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copy2(source,OUT/'code'/source.name)
    shutil.copy2(Path(__file__).with_name('protocol.json'),OUT/'code/protocol.json')
    records=json.loads(Path(protocol['source']).read_text(encoding='utf-8'))
    records=[r for r in records if r['kind'] in protocol['include_kinds']]
    dump(OUT/'probability_vectors.json',records)
    rows=[];diagnostics=[];mcrows=[];configurations=[];started=time.perf_counter()
    try:
        for k in protocol['K']:
            p=np.array([r['p'] for r in records])
            if k==2:p=p[:,protocol['K2_projection']]
            permutations=np.array(list(itertools.permutations(range(k))))
            expanded=p[:,permutations].reshape(-1,k)
            points,subsets=support(k,protocol['quadrature']['seed'])
            np.savez_compressed(OUT/f'prior_support_K{k}.npz',points=points,low=subsets[1],half0=subsets[2],half1=subsets[3])
            for m0 in protocol['m0']:
                print(f'Configuration K={k}, B=16, m0={m0}; {len(points)} finite-prior points',flush=True)
                start=time.perf_counter();model=RiskModel(points,16,subsets,protocol['resource_ceiling']['seconds_per_configuration']);exact={}
                for method in protocol['methods']:
                    begin=time.perf_counter()
                    result=exact_fixed(expanded,m0,16,method) if method in ['variance-only','JADA'] else exact_sequential(model,expanded,m0,method)
                    result=average_permutations(result,permutations)
                    exact[method]=result
                    if result['mass_error']>1e-10:raise ArithmeticError('Reward-path probabilities do not sum to one')
                    diagnostic={'K':k,'m0':m0,'method':method,'seconds':time.perf_counter()-begin,'mass_error':result['mass_error']}
                    mask=np.array([r['kind']=='consumed' for r in records])
                    for reason,key in [('preferred','preferred_fraction'),('numerical-fallback','numerical_fallback_fraction'),('reliability-fallback','reliability_fallback_fraction'),('no-positive-gain','no_positive_gain_fraction')]:
                        diagnostic[key]=float(result.get('reasons',{}).get(reason,np.zeros(len(p)))[mask].mean()/(16-k*m0))
                    diagnostic['history_counts']=result.get('history_counts',{});diagnostics.append(diagnostic)
                    for ix,record in enumerate(records):
                        row={'K':k,'B':16,'m0':m0,'pilot_fraction':k*m0/16,'vector_index':ix,'id':record['id'],'kind':record['kind'],'component':record['component'],'method':method,
                             'dispersion':float(np.std(p[ix],ddof=1)),'risk':float(result['risk'][ix]),'bias_squared':float(result['bias_squared'][ix]),'variance':float(result['variance'][ix])}
                        rows.append(row)
                    np.savez_compressed(OUT/f'exact_K{k}_m{m0}_{method}.npz',**{key:value for key,value in result.items() if isinstance(value,np.ndarray)})
                check,mcseconds=paired_mc(model,p,m0,exact,protocol['paired_mc']['episodes_per_vector'],protocol['paired_mc']['seed']+100*k+m0)
                mcrows.extend([{'K':k,'m0':m0,**r} for r in check])
                configurations.append({'K':k,'m0':m0,'seconds':time.perf_counter()-start,'mc_seconds':mcseconds,'risk_cache':model.uniform_risk.cache_info()._asdict(),'score_cache':model.scores.cache_info()._asdict()})
                save_csv(OUT/'metrics.csv',rows);save_csv(OUT/'paired_mc.csv',mcrows);dump(OUT/'acquisition.json',diagnostics);dump(OUT/'configurations.json',configurations)
                model.uniform_risk.cache_clear();model.scores.cache_clear();model.choose.cache_clear();del model;gc.collect()
    except Exception as exc:
        dump(OUT/'INCOMPLETE.json',{'error':repr(exc),'completed_configurations':configurations,'seconds':time.perf_counter()-started})
        raise
    effects,summary=summarize(rows,records,protocol)
    summary.update({'seconds':time.perf_counter()-started,'configurations':len(configurations),'vectors':len(records),'consumed_vectors':96,'stress_vectors':6,
                    'method_vector_evaluations':len(rows),'max_mass_error':max(d['mass_error'] for d in diagnostics),
                    'paired_mc_max_abs_z':max(abs(r['z']) for r in mcrows if r['z'] is not None),
                    'paired_mc_fraction_abs_z_over_3':sum(r['z'] is not None and abs(r['z'])>3 for r in mcrows)/len(mcrows),
                    'paired_mc_zero_se_unresolved':sum(r['zero_se_unresolved'] for r in mcrows),
                    'paired_mc_all_exact_effects_inside_pointwise_hoeffding':all(r['exact_in_hoeffding_interval'] for r in mcrows),
                    'paired_mc_precision_status':'INCONCLUSIVE_FOR_RARE_DEVIATIONS'})
    save_csv(OUT/'effects.csv',effects);dump(OUT/'summary.json',summary);write_report(summary,effects,diagnostics,rows)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    OUT=args.output
    if args.smoke:
        out = Path('refine-logs/sequential-feasibility/PLACEHOLDER_smoke.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        # PLACEHOLDER: entry-point check only, no estimator or risk observations.
        out.write_text(json.dumps({'status': 'ENTRY_POINT_ONLY', 'GPU_cost': 0}))
        print(out)
        return
    evaluate()


if __name__ == '__main__':
    main()
