"""CPU pilot-information gate: fixed-main/fixed-reuse and matched total cost."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from experiments.cpu_gate import estimator as e
from experiments.cpu_gate.run import dump, vectors, population_pair, save_csv
from experiments.nonlinear_oracle.oracle import conditional_risk_grid
from .fit import PRIORS, prior_points
from .policy import make_policy, joint_integrals, symmetrize_integrals, PilotPolicy, evaluate
from .profile_information import SETTINGS

OUT = Path('refine-logs/pilot-information-gate/20261011_run01')


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    feasibility = json.loads((OUT/'FEASIBILITY.json').read_text(encoding='utf-8'))
    protocol = {'settings': SETTINGS, 'offline_integration': feasibility['offline_integration'],
                'confirmation_seed': 1618032026, 'confirmation_vectors':96, 'primary': PRIORS[0],
                'secondary': PRIORS[1], 'engine':'exact all positive main compositions and reward count states',
                'tie_tolerance':1e-12, 'dispersion_edges':[0,.01,.05,.25], 'leverage_edges':[.25,.4,.6,.75],
                'primary_gate':'fixed-main-m0-4 primary lower CI95 >5%',
                'practical_gate':'primary gate AND fixed-total-m0-4 lower CI95 >5% AND positive primary points in all fixed-total settings',
                'intervals':'stratified paired vector-cluster t delta intervals conditional on frozen approximate prior policies',
                'trend':'paired absolute MSE gain difference m0=4 minus m0=1 at fixed main/lambda',
                'GPU_cost':0, 'prepared_utc':utc()}
    old = json.loads(Path('experiments/cpu_gate/protocol.json').read_text(encoding='utf-8'))
    old['K']=[4]; old['heterogeneous']['seed']=protocol['confirmation_seed']
    records = [{**r,'kind':'confirmation' if r['kind']=='heterogeneous' else 'stress',
                'id':'confirm4-'+r['id'] if r['kind']=='heterogeneous' else r['id']} for r in vectors(old)]
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Do not overwrite a prepared run')
    dump(OUT/'protocol.json',protocol); dump(OUT/'probability_vectors.json',records)
    return protocol,records


def fit(protocol):
    config = protocol['offline_integration']
    policies, diagnostics = {}, []
    for prior in PRIORS:
        tables = {s['id']: {'nums':[], 'masses':[]} for s in SETTINGS}
        saved_points, labels = [], []
        for rep in range(config['replicates']):
            p, components = prior_points(prior,rep,config)
            saved_points.append(p); labels.append(components)
            for s in SETTINGS:
                start = time.perf_counter()
                g = conditional_risk_grid(p,s['total'],s['m0'],s['lambda'])
                num,mass = joint_integrals(g)
                num,mass = symmetrize_integrals(g['pilots'],g['allocations'],num,mass)
                t = tables[s['id']]; t['nums'].append(num); t['masses'].append(mass)
                t['pilots'],t['allocations'] = g['pilots'],g['allocations']
                print(f'FIT {prior} scramble {rep+1}/{config["replicates"]} {s["id"]}: {time.perf_counter()-start:.1f}s',flush=True)
                del g
        np.savez_compressed(OUT/f'prior_points_{prior}.npz',points=np.array(saved_points),components=np.array(labels))
        policies[prior]={}
        for s in SETTINGS:
            t=tables[s['id']]; nums,masses=np.array(t['nums']),np.array(t['masses'])
            def construct(a,b):
                return make_policy(t['pilots'],t['allocations'],nums[a:b].mean(0),masses[a:b].mean(0),s['total'],s['m0'],prior)
            policy=construct(0,len(nums)); half=len(nums)//2
            first,last=construct(0,half),construct(half,len(nums))
            tv=.5*np.abs(first.probabilities-last.probabilities).sum(1)
            uni=np.flatnonzero(np.all(policy.allocations==(s['total']-4*s['m0'])//4,axis=1))[0]
            replicate_risk=nums/masses[:,:,None]
            difference=replicate_risk[:,:,uni]-np.sum(replicate_risk*policy.probabilities[None],axis=2)
            d={'prior':prior,'setting':s['id'],'fit_completed_utc':utc(),
               'pilot_mass_weighted_half_policy_TV':float(masses.mean(0)@tv),
               'half_policy_changed_histories':int(np.sum(tv>1e-12)),
               'max_selected_vs_uniform_paired_scramble_SE':float(np.max(difference.std(0,ddof=1)/np.sqrt(len(nums)))),
               'prior_expected_uniform_mse':float(nums.mean(0)[:,uni].sum()),
               'prior_expected_policy_mse':float(np.sum(nums.mean(0)*policy.probabilities))}
            policies[prior][s['id']]=policy; diagnostics.append(d)
            dump(OUT/f'policy_{prior}_{s["id"]}.json',{**policy.to_dict(),'reuse_lambda':s['lambda'],'construction':d})
            np.savez_compressed(OUT/f'integration_{prior}_{s["id"]}.npz',numerators=nums,masses=masses,
                                pilots=t['pilots'],allocations=t['allocations'])
    dump(OUT/'integration_diagnostics.json',diagnostics)
    dump(OUT/'POLICY_FREEZE.json',{'frozen_utc':utc(),'evaluation_started':False,'methods':list(PRIORS)})
    return policies,diagnostics


def comparator_risks(g,s):
    risk=g['conditional_risks']; pilots=g['pilots']; p=g['p']; weights=g['pilot_probabilities']
    lookup={tuple(m):i for i,m in enumerate(g['allocations'])}
    methods={name:np.empty(len(p)) for name in ['uniform','variance-only','JADA']}
    for v,q in enumerate(p):
        counts=e.allocations(pilots,q,s['m0'],s['total'])
        for name in methods:
            idx=np.array([lookup[tuple(m)] for m in counts[name]])
            methods[name][v]=weights[v]@risk[v,np.arange(len(pilots)),idx]
    methods['adaptive-nonlinear-oracle']=np.sum(weights*risk.min(2),axis=1)
    return methods


def comparisons(rows):
    fresh=[r for r in rows if r['kind']=='confirmation']
    return {s['id']:{f'{method}-vs-{base}':population_pair([r for r in fresh if r['setting']==s['id']],base,'precommitted',candidate=method)
                    for method in ['JADA',*PRIORS,'adaptive-nonlinear-oracle'] for base in ['uniform','variance-only']} for s in SETTINGS}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args()
    if args.prepare_only:
        prepare(); print('Protocol and 96 fresh plus six stress vectors saved before fitting.'); return
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    records=json.loads((OUT/'probability_vectors.json').read_text(encoding='utf-8'))
    if (OUT/'POLICY_FREEZE.json').exists() or (OUT/'results.jsonl').exists():
        raise FileExistsError('Do not overwrite a completed or partially evaluated run')
    start=time.perf_counter(); policies,diagnostics=fit(protocol)
    rows=[]; p=np.array([r['p'] for r in records])
    with (OUT/'results.jsonl').open('w',encoding='utf-8') as f:
        for s in SETTINGS:
            g=conditional_risk_grid(p,s['total'],s['m0'],s['lambda'])
            np.savez_compressed(OUT/f'exact_{s["id"]}.npz',**g)
            methods=comparator_risks(g,s)
            for prior in PRIORS:
                methods[prior]=evaluate(g,policies[prior][s['id']])
            if any(np.any(methods['adaptive-nonlinear-oracle']>v+1e-12) for v in methods.values()):
                raise ArithmeticError('Oracle dominance failed')
            for i,r in enumerate(records):
                z=p[i]-p[i].mean(); dispersion=np.mean(z*z)
                row={**r,'setting':s['id'],'arm':s['arm'],'m0':s['m0'],'total':s['total'],'lambda':s['lambda'],
                     'pilot_fraction':4*s['m0']/s['total'],'dispersion':float(dispersion),
                     'max_leverage':float(np.max(z*z)/(4*dispersion)) if dispersion>0 else 0,
                     'methods':{name+'/precommitted':{'mse':float(v[i])} for name,v in methods.items()}}
                rows.append(row); f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush(); print(f'EVALUATED {s["id"]}: exact {len(p)} vectors',flush=True)
    pairs=comparisons(rows)
    key='posterior-Jeffreys-vs-uniform'
    primary=pairs['fixed-main-m0-4'][key]['relative_ci95'][0]>.05
    practical=primary and pairs['fixed-total-m0-4'][key]['relative_ci95'][0]>.05 and all(
        pairs[s['id']][key]['relative_reduction']>0 for s in SETTINGS if s['arm']=='fixed-total')
    summary={'decision':'PASS_EQUAL_COST_CPU' if practical else 'PASS_INFORMATION_ONLY' if primary else 'NO_CLEAR_SIGNAL',
             'comparisons':pairs,'integration_diagnostics':diagnostics,'exact_cells':len(rows),'method_evaluations':6*len(rows),
             'elapsed_seconds':time.perf_counter()-start,'GPU_cost':0,'review_status':'local-only; delegated reviewer unavailable (usage limit)',
             'confirmation_consumed':True,'primary_pass':bool(primary),'equal_cost_pass':bool(practical)}
    # Paired absolute-gain trend represented as a difference of two risks for each vector.
    trend=[]
    for r in rows:
        if r['kind']=='confirmation' and r['setting']=='fixed-main-m0-4':
            first=next(x for x in rows if x['id']==r['id'] and x['setting']=='fixed-main-m0-1')
            gain1=first['methods']['uniform/precommitted']['mse']-first['methods']['posterior-Jeffreys/precommitted']['mse']
            gain4=r['methods']['uniform/precommitted']['mse']-r['methods']['posterior-Jeffreys/precommitted']['mse']
            trend.append({**r,'methods':{'baseline/precommitted':{'mse':4+gain4},'candidate/precommitted':{'mse':4+gain1}}})
    trend_metric=population_pair(trend,'baseline','precommitted',candidate='candidate')
    summary['paired_gain_trend_m0_4_minus_1']={k:trend_metric[k] for k in
        ['difference','difference_ci95','df_difference','vector_clusters','strata']}
    dump(OUT/'summary.json',summary)
    save_csv(OUT/'metrics.csv',[{'id':r['id'],'kind':r['kind'],'component':r['component'],'setting':r['setting'],
                              'method':method,'mse':v['mse']} for r in rows for method,v in r['methods'].items()])
    from .information_report import report_and_audit
    report_and_audit(OUT,protocol,records,rows,summary)
    print(json.dumps({'decision':summary['decision'],'primary':pairs['fixed-main-m0-4'][key],
                     'equal_cost':pairs['fixed-total-m0-4'][key]},indent=2),flush=True)


if __name__=='__main__':
    main()
