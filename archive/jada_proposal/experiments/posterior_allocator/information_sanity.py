"""Small end-to-end spine on a finite exchangeable prior, no holdout access."""
import itertools
import numpy as np
from experiments.nonlinear_oracle.oracle import conditional_risk_grid
from .policy import joint_integrals, make_policy, evaluate, PilotPolicy
from .information import comparator_risks
from experiments.cpu_gate.run import dump
from .information import OUT


def main():
    p=np.array(list(set(itertools.permutations([.1,.4,.6,.9]))))
    results=[]
    for m0 in [1,4]:
        s={'id':f'sanity-m0-{m0}','m0':m0,'total':8+4*m0,'lambda':.5}
        g=conditional_risk_grid(p,s['total'],m0,.5)
        num,mass=joint_integrals(g)
        policy=make_policy(g['pilots'],g['allocations'],num,mass,s['total'],m0,'finite-sanity')
        rebuilt=PilotPolicy.from_dict(policy.to_dict())
        counts=rebuilt.allocate(np.zeros(4,dtype=int),np.random.default_rng(17))
        assert counts.sum()==8 and np.all(counts>=1)
        methods=comparator_risks(g,s)
        score=evaluate(g,rebuilt)
        assert np.all(score>=methods['adaptive-nonlinear-oracle']-1e-12)
        assert score.mean()<=methods['uniform'].mean()+1e-12
        results.append({'m0':m0,'uniform_mse':float(methods['uniform'].mean()),'policy_mse':float(score.mean()),
                        'oracle_mse':float(methods['adaptive-nonlinear-oracle'].mean())})
    dump(OUT/'SANITY.json',{'status':'PASS','finite_exchangeable_prior_vectors':len(p),
         'settings':results,'holdout_accessed':False,'GPU_cost':0})
    print(results,flush=True)


if __name__=='__main__':
    main()
