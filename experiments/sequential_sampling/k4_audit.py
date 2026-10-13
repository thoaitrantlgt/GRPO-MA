"""Reconstruct saved K4 policies using independent direct-error outcome sums."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .core import target
from .evaluation import exact_sequential,average_permutations
from .symmetry_control import SavedControl,build_graph,graph_metrics,canonical,next_key


BASE=Path('refine-logs/information-control-k4/20261011_run01')


def main():
    protocol=json.loads((BASE/'protocol.json').read_text());records=json.loads((BASE/'probability_vectors.json').read_text())
    highest=protocol['quadrature']['nested_base_points_per_scramble'][-1]
    model=SavedControl(BASE/f'policy_n{highest}.npz');points=np.load(BASE/f'prior_support_n{highest}.npy')
    stress=np.array([r['kind']=='stress' for r in records]);p=np.array([r['p'] for r in records])[stress]
    permutations=np.array(list(itertools.permutations(range(4))));expanded=p[:,permutations].reshape(-1,4)
    maximum=0.;prior_error=0.;budget_ok=True
    metadata=json.loads((BASE/'levels.json').read_text())[-1]
    with threadpool_limits(limits=1):
        for m0 in [1,2]:
            result=average_permutations(exact_sequential(model,expanded,m0,'bayes-control'),permutations)
            with np.load(BASE/f'exact_n{highest}_m{m0}.npz') as saved:
                for key in ['risk','bias_squared','variance','mean_error','second','expected_counts']:
                    maximum=max(maximum,float(np.max(np.abs(result[key]-saved[key][stress]))))
            graph=build_graph(model,m0)
            budget_ok &= all(sum(n)==16 and min(n)>=m0 and all(0<=v<=c for v,c in zip(s,n)) for n,s in graph['terminal'])
            prior=graph_metrics(model,graph,points)['risk'].mean()
            prior_error=max(prior_error,abs(float(prior)-metadata['prior_values'][str(m0)]))
    with np.load(BASE/f'policy_n{highest}.npz') as saved:
        keys=[canonical(tuple(n),tuple(s)) for n,s in zip(saved['n'],saved['s'])]
        values=dict(zip(keys,saved['value']));actions=dict(zip(keys,saved['action']))
    rng=np.random.default_rng(3464102027);chosen=rng.choice(len(keys),512,replace=False)
    residual=0.;terminal_error=0.;truth=target(points)
    # Direct products, independently of solver log-weight normalization.
    for ix in chosen:
        key=keys[int(ix)];n=np.array([x[0] for x in key]);s=np.array([x[1] for x in key])
        w=np.prod(points**s*(1-points)**(n-s),axis=1);w/=w.sum()
        if sum(n)==16:
            credit=target(s/n);cost=float(w@np.sum((credit-truth)**2,axis=1))
            terminal_error=max(terminal_error,abs(cost-values[key]))
        else:
            predictive=w@points
            q=np.array([predictive[j]*values[next_key(key,j,1)]+(1-predictive[j])*values[next_key(key,j,0)] for j in range(4)])
            residual=max(residual,abs(values[key]-q.min()),abs(values[key]-q[actions[key]]))
    mismatches=[]
    for name,digest in protocol['code_sha256'].items():
        for file in [BASE/'code'/name,Path(__file__).with_name(name)]:
            if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:mismatches.append(str(file))
    checks={'saved_policy_direct_error_reconstruction':maximum<1e-10,'finite_prior_Bellman_value_equals_outcome_mean':prior_error<1e-10,
            'saved_policy_Bellman_check_independent_weights':residual<1e-10 and terminal_error<1e-10,'exact_budget_and_coverage':bool(budget_ok),
            'frozen_evaluator_code_matches':not mismatches}
    checks={name:bool(value) for name,value in checks.items()}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'maximum_saved_metric_discrepancy':maximum,
            'maximum_prior_value_discrepancy':prior_error,'sampled_saved_Bellman_states':512,'maximum_independent_Bellman_residual':residual,
            'maximum_independent_terminal_error':terminal_error,'code_mismatches':mismatches,'implementation_tests':15,
            'scope':'local deterministic saved-table reconstruction; no semantic or continuous-prior certification','GPU_cost':0}
    (BASE/'SAVED_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert all(checks.values())


if __name__=='__main__':main()
