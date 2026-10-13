"""Audit numerical convergence from frozen files; no refitting or new data."""
import csv
import itertools
import json
import numpy as np
from .convergence import OUT,SOURCE
from .fit import prior_points,PRIORS
from .policy import PilotPolicy,make_policy,evaluate
from experiments.cpu_gate.run import dump


def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    freeze=json.loads((OUT/'POLICY_FREEZE.json').read_text(encoding='utf-8'))
    if freeze['development_evaluation_started'] is not False or freeze['new_confirmation'] is not False:
        raise ValueError('Missing development-only freeze')
    records=json.loads((SOURCE/'probability_vectors.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8',newline='')))
    lookup={(r['id'],r['setting'],r['prior'],int(r['level'])):r for r in rows}
    if len(lookup)!=12*len(records) or len(rows)!=len(lookup):
        raise ValueError('Coverage or duplication failure')
    max_risk=max_metric=0.
    for prior in PRIORS:
        with np.load(OUT/f'prior_points_{prior}.npz') as a:
            for rep in range(protocol['offline_integration']['replicates']):
                p,c=prior_points(prior,rep,protocol['offline_integration'])
                np.testing.assert_array_equal(p,a['points'][rep]); np.testing.assert_array_equal(c,a['components'][rep])
        for s in protocol['settings']:
            with np.load(SOURCE/f'exact_{s["id"]}.npz') as a:
                grid=dict(a)
            np.testing.assert_array_equal(grid['p'],np.array([r['p'] for r in records]))
            oracle=np.sum(grid['pilot_probabilities']*grid['conditional_risks'].min(2),axis=1)
            for level in protocol['nested_levels_per_component'][prior]:
                name=f'{prior}_{s["id"]}_n{level}'
                data=json.loads((OUT/f'policy_{name}.json').read_text(encoding='utf-8'))
                if data['reuse_lambda']!=s['lambda']:
                    raise ValueError('Wrong fixed reuse')
                policy=PilotPolicy.from_dict(data)
                with np.load(OUT/f'integration_{name}.npz') as a:
                    rebuilt=make_policy(a['pilots'],a['allocations'],a['numerators'].mean(0),a['masses'].mean(0),s['total'],4,prior)
                np.testing.assert_array_equal(policy.probabilities,rebuilt.probabilities)
                max_risk=max(max_risk,float(np.max(np.abs(policy.posterior_risk-rebuilt.posterior_risk))))
                hm={tuple(h):i for i,h in enumerate(policy.pilots)}; mm={tuple(m):i for i,m in enumerate(policy.allocations)}
                for perm in itertools.permutations(range(4)):
                    hi=[hm[tuple(h[list(perm)])] for h in policy.pilots]; mi=[mm[tuple(m[list(perm)])] for m in policy.allocations]
                    np.testing.assert_allclose(policy.probabilities,policy.probabilities[np.ix_(hi,mi)],atol=1e-13,rtol=0)
                risk=evaluate(grid,policy)
                if np.any(risk<oracle-1e-12):
                    raise ValueError('Oracle lower comparator violated')
                for i,r in enumerate(records):
                    row=lookup[(r['id'],s['id'],prior,level)]
                    if row['kind']!=('development' if r['kind']=='confirmation' else 'stress'):
                        raise ValueError('Consumed data relabeled as new confirmation')
                    max_metric=max(max_metric,abs(risk[i]-float(row['policy_mse'])))
                idx=[i for i,r in enumerate(records) if r['kind']=='confirmation']
                baseline=np.mean([float(lookup[(r['id'],s['id'],prior,level)]['uniform_mse']) for r in records if r['kind']=='confirmation'])
                reduction=100*(1-risk[idx].mean()/baseline)
                d=next(d for d in summary['diagnostics'] if d['prior']==prior and d['setting']==s['id'] and d['points_per_component_per_scramble']==level)
                if abs(reduction-d['consumed_development_reduction_pct'])>1e-10:
                    raise ValueError('Summary reduction mismatch')
    if max_risk>1e-12 or max_metric>1e-12:
        raise ValueError('Reconstruction mismatch')
    result={'status':'PASS','prior_samples_match_committed_rule':'PASS','permutation_equivariance':'PASS',
            'policy_reconstruction_residual':max_risk,'max_saved_metric_residual':max_metric,
            'method_times_cells':len(rows),'development_only':'PASS','new_confirmation':False,'GPU_cost':0}
    dump(OUT/'SAVED_AUDIT.json',result); print(result)


if __name__=='__main__':
    main()
