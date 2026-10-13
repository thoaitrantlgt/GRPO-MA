"""Numerical convergence on consumed development only; no fresh confirmation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from experiments.cpu_gate.run import dump, save_csv
from experiments.nonlinear_oracle.oracle import conditional_risk_grid
from .fit import PRIORS, prior_points
from .policy import make_policy, symmetrize_integrals, evaluate, PilotPolicy

SOURCE=Path('refine-logs/pilot-information-gate/20261011_run01')
OUT=Path('refine-logs/posterior-convergence/20261011_run01')
CONFIG={'seed':1414212026,'replicates':8,'Jeffreys_points_per_replicate':4096,
        'mixture_points_per_component_per_replicate':1024}
LEVELS={'posterior-Jeffreys':[1024,2048,4096],'posterior-mixture':[256,512,1024]}


def nested_sums(grid,positions,limits):
    """Unnormalized joint integrals; component prefixes keep equal strata."""
    outputs=[]
    for limit in limits:
        selected=positions<limit
        likelihood=grid['pilot_probabilities'][selected]
        numerator=np.einsum('vp,vpc->pc',likelihood,grid['conditional_risks'][selected])
        outputs.append((numerator,likelihood.sum(0),int(selected.sum())))
    return outputs


def prepare():
    if OUT.exists():
        raise FileExistsError('Use a new run directory; preserve previous diagnostics')
    OUT.mkdir(parents=True)
    source_protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    settings=[s for s in source_protocol['settings'] if s['m0']==4]
    protocol={'prepared_utc':datetime.now(timezone.utc).isoformat(),'settings':settings,
              'offline_integration':CONFIG,'nested_levels_per_component':LEVELS,'chunk_points':256,
              'evaluation_source':str(SOURCE),'evaluation_status':'consumed confirmation reclassified as development',
              'new_confirmation':False,'new_outcome_sampling':False,'population_distribution_changed':False,
              'policy_inputs':'committed prior and pilot counts only; evaluation vectors never fit inputs',
              'decision':'recommend stop current allocation family if all largest-level half-policy TV<=.01 and every largest-level consumed-development reduction<5%; otherwise numerical diagnosis inconclusive. Not an impossibility certificate or new GO.',
              'limits':'exact finite reward risks; approximate continuous prior integration; development comparisons descriptive without fresh confidence claims',
              'GPU_cost':0}
    dump(OUT/'protocol.json',protocol)
    return protocol


def integrate(protocol):
    all_policies={}; diagnostics=[]
    for prior in PRIORS:
        levels=LEVELS[prior]; components=1 if prior==PRIORS[0] else 6
        tables={s['id']:{limit:{'nums':[],'masses':[]} for limit in levels} for s in protocol['settings']}
        saved_points=[]; saved_labels=[]
        for rep in range(CONFIG['replicates']):
            start=time.perf_counter()
            points,labels=prior_points(prior,rep,CONFIG)
            saved_points.append(points); saved_labels.append(labels)
            positions=np.tile(np.arange(levels[-1]),components)
            for s in protocol['settings']:
                sums={limit:{'num':0,'mass':0,'count':0} for limit in levels}
                for begin in range(0,len(points),protocol['chunk_points']):
                    end=min(begin+protocol['chunk_points'],len(points))
                    grid=conditional_risk_grid(points[begin:end],s['total'],4,s['lambda'])
                    for limit,(num,mass,count) in zip(levels,nested_sums(grid,positions[begin:end],levels)):
                        sums[limit]['num']+=num; sums[limit]['mass']+=mass; sums[limit]['count']+=count
                    pilots,allocations=grid['pilots'],grid['allocations']
                    del grid
                for limit in levels:
                    value=sums[limit]
                    if value['count']!=components*limit:
                        raise ArithmeticError('Nested component prefixes are unbalanced')
                    num,mass=symmetrize_integrals(pilots,allocations,value['num']/value['count'],value['mass']/value['count'])
                    table=tables[s['id']][limit]; table['nums'].append(num); table['masses'].append(mass)
                    table['pilots'],table['allocations']=pilots,allocations
            print(f'{prior}: scramble {rep+1}/8, {len(points)} prior points, both settings; {time.perf_counter()-start:.1f}s',flush=True)
        np.savez_compressed(OUT/f'prior_points_{prior}.npz',points=np.array(saved_points),components=np.array(saved_labels))
        for s in protocol['settings']:
            previous=None
            old=PilotPolicy.from_dict(json.loads((SOURCE/f'policy_{prior}_{s["id"]}.json').read_text(encoding='utf-8')))
            for limit in levels:
                table=tables[s['id']][limit]; nums,masses=np.array(table['nums']),np.array(table['masses'])
                def construct(a,b):
                    return make_policy(table['pilots'],table['allocations'],nums[a:b].mean(0),masses[a:b].mean(0),s['total'],4,prior)
                policy=construct(0,8); first,last=construct(0,4),construct(4,8)
                mass=masses.mean(0); uni=np.flatnonzero(np.all(policy.allocations==2,axis=1))[0]
                def tv(other):
                    return float(mass@(.5*np.abs(policy.probabilities-other.probabilities).sum(1)))
                half_tv=float(mass@(.5*np.abs(first.probabilities-last.probabilities).sum(1)))
                replicate_gains=nums[:,:,uni].sum(1)-np.sum(nums*policy.probabilities[None],axis=(1,2))
                d={'prior':prior,'setting':s['id'],'points_per_component_per_scramble':limit,
                   'total_prior_points':limit*components*8,'half_policy_TV':half_tv,
                   'policy_TV_vs_original':tv(old),'policy_TV_vs_previous_level':None if previous is None else tv(previous),
                   'prior_uniform_mse':float(nums.mean(0)[:,uni].sum()),
                   'prior_policy_mse':float(np.sum(nums.mean(0)*policy.probabilities)),
                   'prior_gain_scramble_SE':float(replicate_gains.std(ddof=1)/np.sqrt(8)),
                   'prior_gain_SE_scope':'diagnostic conditional on pooled policy selected from these same scrambles; not an independent optimal-risk interval'}
                name=f'{prior}_{s["id"]}_n{limit}'
                dump(OUT/f'policy_{name}.json',{**policy.to_dict(),'reuse_lambda':s['lambda'],'diagnostics':d})
                np.savez_compressed(OUT/f'integration_{name}.npz',numerators=nums,masses=masses,
                                    pilots=table['pilots'],allocations=table['allocations'])
                diagnostics.append(d); all_policies[(prior,s['id'],limit)]=policy; previous=policy
    dump(OUT/'POLICY_FREEZE.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
         'development_evaluation_started':False,'new_confirmation':False,'priors':list(PRIORS)})
    return all_policies,diagnostics


def summarize(protocol,policies,diagnostics,start):
    for prior in PRIORS:
        with np.load(OUT/f'prior_points_{prior}.npz') as a:
            for rep in range(protocol['offline_integration']['replicates']):
                p,labels=prior_points(prior,rep,protocol['offline_integration'])
                np.testing.assert_array_equal(p,a['points'][rep])
                np.testing.assert_array_equal(labels,a['components'][rep])
    records=json.loads((SOURCE/'probability_vectors.json').read_text(encoding='utf-8'))
    selected=np.array([i for i,r in enumerate(records) if r['kind']=='confirmation'])
    scores=[]; max_saved_error=0.
    for s in protocol['settings']:
        with np.load(SOURCE/f'exact_{s["id"]}.npz') as a:
            grid=dict(a)
        uni=np.flatnonzero(np.all(grid['allocations']==2,axis=1))[0]
        base=np.sum(grid['pilot_probabilities']*grid['conditional_risks'][:,:,uni],axis=1)
        oracle=np.sum(grid['pilot_probabilities']*grid['conditional_risks'].min(2),axis=1)
        for prior in PRIORS:
            for limit in LEVELS[prior]:
                policy=policies[(prior,s['id'],limit)]; risk=evaluate(grid,policy)
                if np.any(risk<oracle-1e-12):
                    raise ArithmeticError('Full-information lower comparator violated')
                name=f'{prior}_{s["id"]}_n{limit}'
                with np.load(OUT/f'integration_{name}.npz') as a:
                    rebuilt=make_policy(a['pilots'],a['allocations'],a['numerators'].mean(0),a['masses'].mean(0),s['total'],4,prior)
                np.testing.assert_array_equal(rebuilt.probabilities,policy.probabilities)
                max_saved_error=max(max_saved_error,float(np.max(np.abs(rebuilt.posterior_risk-policy.posterior_risk))))
                matching=next(d for d in diagnostics if d['prior']==prior and d['setting']==s['id'] and d['points_per_component_per_scramble']==limit)
                matching['consumed_development_reduction_pct']=float(100*(1-risk[selected].mean()/base[selected].mean()))
                matching['consumed_development_policy_mse']=float(risk[selected].mean())
                for i,r in enumerate(records):
                    scores.append({'id':r['id'],'kind':'development' if r['kind']=='confirmation' else 'stress',
                                   'component':r['component'],'setting':s['id'],'prior':prior,'level':limit,
                                   'policy_mse':float(risk[i]),'uniform_mse':float(base[i]),'oracle_mse':float(oracle[i])})
    largest=[d for d in diagnostics if d['points_per_component_per_scramble']==LEVELS[d['prior']][-1]]
    stable=all(d['half_policy_TV']<=.01 for d in largest)
    small=all(d['consumed_development_reduction_pct']<5 for d in largest)
    decision='RECOMMEND_STOP_CURRENT_ALLOCATOR_FAMILY' if stable and small else 'NUMERICAL_DIAGNOSIS_INCONCLUSIVE'
    summary={'decision':decision,'largest_level_policy_stable':stable,'diagnostics':diagnostics,
             'elapsed_seconds':time.perf_counter()-start,'new_confirmation':False,'GPU_cost':0,
             'original_gate_unchanged':'NO_CLEAR_SIGNAL','review':'local diagnostic; no fresh reviewer clearance'}
    dump(OUT/'summary.json',summary); save_csv(OUT/'metrics.csv',scores)
    dump(OUT/'AUDIT.json',{'status':'PASS','policy_reconstruction_residual':max_saved_error,
         'positive_integer_allocations':'PASS via PilotPolicy validation','nested_component_balance':'PASS',
         'oracle_dominance':'PASS','prior_samples_match_committed_rule':'PASS',
         'consumed_development_vectors':len(selected),'methods_times_cells':len(scores),
         'new_confirmation':False,'GPU_cost':0})
    from .convergence_report import report
    report(OUT,summary)
    print(json.dumps({'decision':decision,'largest_levels':largest},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args()
    if args.prepare_only:
        prepare(); print('Three nested levels, eight scrambles and two m0=4 settings committed; no fit/evaluation.'); return
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if (OUT/'POLICY_FREEZE.json').exists() or (OUT/'summary.json').exists():
        raise FileExistsError('Preserve previous fitted run')
    start=time.perf_counter(); policies,diagnostics=integrate(protocol)
    summarize(protocol,policies,diagnostics,start)


if __name__=='__main__':
    main()
