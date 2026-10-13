"""Posthoc mechanism audit and the proposal's exploratory K2 information control."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from .core import RiskModel, child
from .information_control import BayesControl
from .evaluation import exact_sequential, average_permutations


BASE=Path('refine-logs/sequential-feasibility/20261011_run02')
OUT=BASE/'mechanism-diagnostic'


def main():
    if not (BASE/'summary.json').exists():raise RuntimeError('Wait for the frozen primary trial to complete')
    if OUT.exists():raise FileExistsError('Preserve completed diagnostics')
    OUT.mkdir()
    declaration={'stage':'posthoc exploratory diagnosis after candidate failure; cannot rescue primary verdict',
                 'one_action_identity':'if B=K*t, all n_i<=t and n_j+1<=t, one step j followed by balanced completion gives the same terminal counts t and identical terminal-credit law',
                 'control':'full Bayes Bellman recursion, K2/B16/m0=1,2, unchanged frozen finite prior and vectors; no reliability gate',
                 'GPU_cost':0}
    (OUT/'protocol.json').write_text(json.dumps(declaration,indent=2))
    records=json.loads((BASE/'probability_vectors.json').read_text())
    audit=[]
    for k in [2,4]:
        with np.load(BASE/f'prior_support_K{k}.npz') as data:points=data['points']
        model=RiskModel(points,16,deadline_seconds=600);quota=16//k
        for nval in [1,quota-1]:
            n=(nval,)*k;maximum=0.;histories=0
            for s in itertools.product(range(nval+1),repeat=k):
                base=model.uniform_risk(n,s)
                for j in range(k):
                    probability=points[:,j]
                    q=probability*model.uniform_risk(*child(n,s,j,1))+(1-probability)*model.uniform_risk(*child(n,s,j,0))
                    maximum=max(maximum,float(np.max(np.abs(base-q))))
                histories+=1
            audit.append({'K':k,'n_per_thought':nval,'histories':histories,'max_pointwise_absolute_one_action_gain':maximum})
        model.uniform_risk.cache_clear();model.scores.cache_clear();model.choose.cache_clear()
    (OUT/'one_action_identity.json').write_text(json.dumps(audit,indent=2))
    with np.load(BASE/'prior_support_K2.npz') as data:points=data['points']
    p=np.array([r['p'] for r in records])[:,[0,3]]
    permutations=np.array([[0,1],[1,0]]);expanded=p[:,permutations].reshape(-1,2)
    controls=[];allrisk={}
    for m0 in [1,2]:
        started=time.perf_counter();model=BayesControl(points,16)
        result=average_permutations(exact_sequential(model,expanded,m0,'bayes-control'),permutations)
        with np.load(BASE/f'exact_K2_m{m0}_uniform.npz') as data:baseline=data['risk']
        consumed=np.array([r['kind']=='consumed' for r in records]);stress=~consumed
        prior_pilot_states=list(itertools.product(range(m0+1),repeat=2));initial=(m0,)*2
        # Conditional diagnostic bound checked for every possible pilot state.
        gaps=[float(model.posterior(initial,s)@model.uniform_risk(initial,s)-model.value(initial,s))
              for s in prior_pilot_states]
        controls.append({'K':2,'B':16,'m0':m0,'development_reduction':float(1-result['risk'][consumed].mean()/baseline[consumed].mean()),
                         'maximum_projected_stress_regression':float(np.max(result['risk'][stress]/baseline[stress]-1)),
                         'minimum_conditional_finite_prior_gain':min(gaps),'bellman_states':model.value.cache_info().currsize,
                         'seconds':time.perf_counter()-started,'mass_error':result['mass_error']})
        np.savez_compressed(OUT/f'bayes_control_K2_m{m0}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        model.value.cache_clear();model.uniform_risk.cache_clear();model.scores.cache_clear();model.choose.cache_clear()
    (OUT/'information_control.json').write_text(json.dumps(controls,indent=2))
    text=['# One-action mechanism diagnosis','',
          '**Exploratory diagnosis; primary candidate verdict remains unchanged.**','',
          '## Terminal-count identity','',
          'Suppose B=K*t, every current count n_i<=t, and the contemplated action satisfies n_j+1<=t. Balanced completion fills all thoughts to t: exactly K*t observations are available and no thought starts above t. Independent Bernoulli reward streams therefore produce the same conditional distribution of final success counts regardless of which admissible thought is sampled first. The terminal estimator depends only on final counts and successes, so Q_j(p,D,q)=Q_U(p,D,q) for every p. This is an identity, independent of the prior or quadrature.','',
          'Starting from balanced coverage, this one-action score cannot prefer a deviation before a thought reaches terminal quota. For B=16, preferences can first arise during the last one observation at K2 or the last three observations at K4. It cannot value early information that would support later adaptive allocation. Simply tightening posterior integration or raising pilot counts cannot remove this structural limitation.','',
          '| K | Current count per thought | Histories checked | Maximum absolute pointwise gain |',
          '|---:|---:|---:|---:|']
    for r in audit:text.append(f"| {r['K']} | {r['n_per_thought']} | {r['histories']} | {r['max_pointwise_absolute_one_action_gain']:.3g} |")
    text+=['','## Full Bayes information control','',
           'This control uses only observed counts and the frozen finite prior, but solves the entire adaptive Bellman problem without the proposed reliability gate. It is an exploratory small-case diagnostic, not a successful replacement method or a true-p oracle.','',
           '| K | m0 | Development reduction | Maximum projected stress regression | Bellman states | CPU seconds |',
           '|---:|---:|---:|---:|---:|---:|']
    for r in controls:text.append(f"| 2 | {r['m0']} | {100*r['development_reduction']:.4f}% | {100*r['maximum_projected_stress_regression']:.4f}% | {r['bellman_states']} | {r['seconds']:.3f} |")
    text+=['','The finite-prior optimization bound applies under its own support distribution; gains on the consumed heterogeneous population are descriptive. No thresholds, prior or evaluation vectors were changed. The one-action acquisition must be redesigned to account for future adaptive allocation before spending on a larger sweep.']
    (OUT/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'identity':audit,'controls':controls},indent=2),flush=True)


if __name__=='__main__':main()
