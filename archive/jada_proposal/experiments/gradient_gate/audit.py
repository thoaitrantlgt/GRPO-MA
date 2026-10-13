"""Saved-table integrity and global weighted-optimum audit, CPU only."""
import json
import math
import itertools
import numpy as np
from experiments.cpu_gate import estimator as e
from experiments.cpu_gate.run import dump,vectors
from experiments.nonlinear_oracle.oracle import conditional_risk_grid,state_probabilities
from . import core
from .run import OUT,pair_tables


def audit(out=OUT):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    records=json.loads((out/'probability_vectors.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (out/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    lookup={(r['id'],r['setting'],r['gram']):r for r in rows}
    expected={(r['id'],s['id'],g) for r in records for s in protocol['settings'] for g in core.NAMES}
    if set(lookup)!=expected or len(rows)!=len(lookup):
        raise ValueError('Incomplete or duplicate evaluation coverage')
    population=json.loads(open('experiments/cpu_gate/protocol.json',encoding='utf-8').read())
    population['K']=[4]; population['heterogeneous']['seed']=protocol['confirmation']['seed']
    fresh=[r for r in records if r['kind']=='confirmation']
    declared=[{**r,'id':'confirm-gradient-'+r['id'],'kind':'confirmation'} for r in vectors(population) if r['kind']=='heterogeneous']
    if fresh!=declared:
        raise ValueError('Fresh benchmark differs from predeclared seed/distribution')
    freeze=json.loads((out/'GEOMETRY_FREEZE.json').read_text(encoding='utf-8'))
    if freeze['evaluation_started'] is not False or freeze['primary_grams']!=protocol['primary_grams']:
        raise ValueError('Missing pre-evaluation geometry freeze')
    saved_grams=json.loads((out/'grams.json').read_text(encoding='utf-8'))
    np.testing.assert_array_equal(np.array([saved_grams[n] for n in core.NAMES]),core.grams())
    p=np.array([r['p'] for r in records]); maxdirect=maxscalar=maxmetric=maxbias=0.
    for s in protocol['settings']:
        with np.load(out/f'exact_{s["id"]}.npz') as a:
            grid=dict(a)
        risk=grid['conditional_risks']; weights=grid['pilot_probabilities']; pilots=grid['pilots']; counts=grid['allocations']
        np.testing.assert_array_equal(grid['p'],p); np.testing.assert_array_equal(grid['matrices'],core.grams())
        if len(counts)!=math.comb(s['total']-4*s['m0']-1,3) or len(set(map(tuple,counts)))!=len(counts) or np.any(counts<1) or np.any(counts.sum(1)!=s['total']-4*s['m0']):
            raise ValueError('Allocation space incomplete')
        if set(map(tuple,pilots))!=set(itertools.product(range(s['m0']+1),repeat=4)):
            raise ValueError('Pilot space incomplete')
        np.testing.assert_allclose(weights,state_probabilities(pilots,np.full(4,s['m0']),p),atol=1e-14)
        scalar=conditional_risk_grid(p,s['total'],s['m0'])
        maxscalar=max(maxscalar,float(np.max(np.abs(risk[:,0]-scalar['conditional_risks']))))
        np.testing.assert_allclose(risk[:,2],.5*risk[:,0],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(risk[:,1],0)
        unconditional=np.einsum('vh,vghc->vgc',weights,risk)
        np.testing.assert_allclose(unconditional,grid['unconditional_risks'],atol=1e-13)
        weighted_opt=np.einsum('vh,vgh->vg',weights,risk.min(3))
        for name in core.METHODS:
            idx=grid['selected_'+name]
            if np.any(idx<0) or np.any(idx>=len(counts)):
                raise ValueError('Infeasible selected allocation')
            chosen=np.take_along_axis(risk,idx[:,:,:,None],axis=3)[:,:,:,0]
            values=np.einsum('vh,vgh->vg',weights,chosen)
            if np.any(values<weighted_opt-1e-12):
                raise ValueError('Weighted full-information lower comparator violated')
            if name=='adaptive-gradient-oracle':
                np.testing.assert_allclose(chosen,risk.min(3),atol=1e-13)
            if name=='fixed-gradient-oracle':
                np.testing.assert_array_equal(idx, np.broadcast_to(idx[:,:,:1],idx.shape))
                np.testing.assert_allclose(values,unconditional.min(2),atol=1e-13)
            if name=='adaptive-scalar-oracle':
                scalar_chosen=np.take_along_axis(risk[:,0],idx[:,0,:,None],axis=2)[:,:,0]
                np.testing.assert_allclose(scalar_chosen,risk[:,0].min(2),atol=1e-13)
            if name=='fixed-scalar-oracle':
                np.testing.assert_array_equal(idx,np.broadcast_to(idx[:,:,:1],idx.shape))
                scalar_score=np.sum(weights*np.take_along_axis(risk[:,0],idx[:,0,:,None],axis=2)[:,:,0],axis=1)
                np.testing.assert_allclose(scalar_score,unconditional[:,0].min(1),atol=1e-13)
            if name=='gradient-JADA':
                np.testing.assert_array_equal(idx,np.broadcast_to(idx[:1],idx.shape))
            for i,r in enumerate(records):
                for gg,g in enumerate(core.NAMES):
                    row=lookup[(r['id'],s['id'],g)]
                    maxmetric=max(maxmetric,abs(values[i,gg]-row['methods'][name+'/budget-precommitted']['mse']))
        for v,h,c in [(0,0,0),(6,len(pilots)//2,len(counts)//2),(len(p)-1,len(pilots)-1,len(counts)-1)]:
            m=counts[c]; main,prob=e.binomial_states(m,p[v]); lam=4*s['m0']/s['total']
            value=lam*pilots[h]/s['m0']+(1-lam)*main/m
            delta=e.advantage(value)-e.advantage(p[v])
            direct=prob@np.einsum('ti,gij,tj->tg',delta,core.grams(),delta)
            maxdirect=max(maxdirect,float(np.max(np.abs(direct-risk[v,:,h,c]))))
        lam=4*s['m0']/s['total']; maxbias=max(maxbias,float(np.max(np.abs(lam*(weights@(pilots/s['m0'])-p)))))
    if max(maxdirect,maxscalar,maxmetric,maxbias)>1e-11:
        raise ValueError('Saved risk/moment audit failed')
    if pair_tables(rows,protocol)!=summary['comparisons']:
        raise ValueError('Intervals differ from saved rows')
    result={'status':'PASS','cells':len(rows),'method_evaluations':len(rows)*len(core.METHODS),
            'exact_global_weighted_oracles':'PASS','scalar_and_null_controls':'PASS',
            'gradient_JADA_pilot_only_given_fixed_G':'PASS','unfiltered_predeclared_confirmation':'PASS',
            'max_direct_weighted_residual':maxdirect,'max_scalar_engine_residual':maxscalar,
            'max_saved_metric_residual':maxmetric,'max_raw_value_bias':maxbias,'GPU_cost':0}
    dump(out/'AUDIT.json',result); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    audit()
