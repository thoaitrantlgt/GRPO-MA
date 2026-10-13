"""Prepare, enumerate and report the predeclared fixed-Gram CPU gate."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
from experiments.cpu_gate.run import dump,vectors,population_pair,save_csv
from . import core

HERE=Path(__file__).resolve().parent
OUT=Path('refine-logs/gradient-objective-gate/20261011_run01')


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve previous run; use a new directory')
    OUT.mkdir(parents=True)
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    protocol['prepared_utc']=datetime.now(timezone.utc).isoformat()
    old=json.loads(Path('refine-logs/pilot-information-gate/20261011_run01/probability_vectors.json').read_text(encoding='utf-8'))
    records=[{**r,'kind':'development' if r['kind']=='confirmation' else 'stress'} for r in old]
    population=json.loads(Path('experiments/cpu_gate/protocol.json').read_text(encoding='utf-8'))
    population['K']=[4]; population['heterogeneous']['seed']=protocol['confirmation']['seed']
    for r in vectors(population):
        if r['kind']=='heterogeneous':
            records.append({**r,'id':'confirm-gradient-'+r['id'],'kind':'confirmation'})
    dump(OUT/'protocol.json',protocol); dump(OUT/'probability_vectors.json',records)
    dump(OUT/'grams.json',{name:matrix.tolist() for name,matrix in zip(core.NAMES,core.grams())})
    print('Seven Grams, three settings, 96 fresh vectors and all development/stress examples committed before outcomes.')


def pair_tables(rows,protocol):
    fresh=[r for r in rows if r['kind']=='confirmation' and r['gram']!='aligned-null']
    candidates=['JADA','gradient-JADA','gradient-surrogate-oracle','fixed-gradient-oracle','adaptive-gradient-oracle']
    def pairs(subset):
        return {f'{method}-vs-{base}':population_pair(subset,base,'budget-precommitted',candidate=method)
                for method in candidates for base in ['uniform','variance-only','adaptive-scalar-oracle']}
    primary=[r for r in fresh if r['gram'] in protocol['primary_grams']]
    return {'primary':pairs(primary),'by_gram':{name:pairs([r for r in fresh if r['gram']==name]) for name in core.NAMES if name!='aligned-null'},
            'by_setting':{s['id']:pairs([r for r in primary if r['setting']==s['id']]) for s in protocol['settings']}}


def sanity():
    p=np.array([[.2,.4,.7,.9],[.5,.5,.5,.5]])
    start=time.perf_counter(); timings=[]
    for s in json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['settings']:
        begin=time.perf_counter(); grid=core.risk_grid(p,s['total'],s['m0']); result=core.solve(grid,s['total'],s['m0'])
        if np.any(result['mse']['adaptive-gradient-oracle']>result['mse']['uniform']+1e-12):
            raise ArithmeticError('Sanity oracle dominance failed')
        timings.append({'setting':s['id'],'seconds':time.perf_counter()-begin,'allocations':len(grid['allocations']),
                        'pilot_histories':len(grid['pilots'])})
    dump(OUT/'SANITY.json',{'status':'PASS','settings':timings,'elapsed_seconds':time.perf_counter()-start,
         'fresh_confirmation_accessed':False,'GPU_cost':0})
    print(timings,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--sanity-only',action='store_true'); args=parser.parse_args()
    if args.prepare_only:
        prepare(); return
    if args.sanity_only:
        sanity(); return
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    records=json.loads((OUT/'probability_vectors.json').read_text(encoding='utf-8'))
    if (OUT/'results.jsonl').exists():
        raise FileExistsError('Do not overwrite results')
    dump(OUT/'GEOMETRY_FREEZE.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'evaluation_started':False,
         'grams':list(core.NAMES),'primary_grams':protocol['primary_grams'],'methods':list(core.METHODS),
         'information':'fixed G is supplied by synthetic model; gradient-JADA runtime does not know current true p'})
    start=time.perf_counter(); rows=[]; p=np.array([r['p'] for r in records])
    with (OUT/'results.jsonl').open('w',encoding='utf-8') as f:
        for s in protocol['settings']:
            pieces=[]
            for begin in range(0,len(p),32):
                grid=core.risk_grid(p[begin:begin+32],s['total'],s['m0'])
                result=core.solve(grid,s['total'],s['m0']); pieces.append(result)
                print(f'{s["id"]}: exact vectors {begin+1}-{min(begin+32,len(p))}/{len(p)}',flush=True)
            array_keys=['p','pilot_probabilities','conditional_risks','unconditional_risks']
            saved={key:np.concatenate([piece[key] for piece in pieces]) for key in array_keys}
            saved.update({key:pieces[0][key] for key in ['matrices','effective_matrices','null_grams','allocations','pilots']})
            saved.update({f'selected_{name}':np.concatenate([piece['selected'][name] for piece in pieces]) for name in core.METHODS})
            np.savez_compressed(OUT/f'exact_{s["id"]}.npz',**saved)
            mse={name:np.concatenate([piece['mse'][name] for piece in pieces]) for name in core.METHODS}
            for i,r in enumerate(records):
                z=p[i]-p[i].mean(); dispersion=np.mean(z*z)
                for gg,name in enumerate(core.NAMES):
                    row={**r,'setting':s['id'],'total':s['total'],'m0':s['m0'],'gram':name,
                         'pilot_fraction':4*s['m0']/s['total'],'dispersion':float(dispersion),
                         'max_leverage':float(np.max(z*z)/(4*dispersion)) if dispersion else 0,
                         'methods':{method+'/budget-precommitted':{'mse':float(value[i,gg])} for method,value in mse.items()}}
                    rows.append(row); f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush(); del pieces,saved,mse
    pairs=pair_tables(rows,protocol)
    oracle=pairs['primary']['adaptive-gradient-oracle-vs-uniform']
    objective=pairs['primary']['adaptive-gradient-oracle-vs-adaptive-scalar-oracle']
    passed=oracle['relative_ci95'][0]>.05 and objective['relative_ci95'][0]>.05
    summary={'decision':'HEADROOM_FOR_GRADIENT_CPU' if passed else 'NO_OBJECTIVE_SPECIFIC_SIGNAL',
             'comparisons':pairs,'cells':len(rows),'method_evaluations':len(rows)*len(core.METHODS),
             'elapsed_seconds':time.perf_counter()-start,'GPU_cost':0,'confirmation_consumed':True,
             'review_status':json.loads((OUT/'REVIEW_STATUS.json').read_text(encoding='utf-8')),
             'scope':'synthetic fixed-G information controls only; no practical gradient benefit or training GO'}
    dump(OUT/'summary.json',summary)
    save_csv(OUT/'metrics.csv',[{'id':r['id'],'kind':r['kind'],'component':r['component'],'setting':r['setting'],
                              'gram':r['gram'],'method':name,'gradient_mse':v['mse']} for r in rows for name,v in r['methods'].items()])
    from .report import report
    report(OUT,protocol,rows,summary)
    from .audit import audit
    audit(OUT)
    print(json.dumps({'decision':summary['decision'],'oracle_vs_uniform':oracle,'objective_specific':objective},indent=2),flush=True)


if __name__=='__main__':
    main()
