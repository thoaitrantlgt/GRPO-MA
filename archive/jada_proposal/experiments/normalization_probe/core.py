"""Exact scalar-shrinkage risk for uniformly pooled Bernoulli estimates."""
import numpy as np
from experiments.cpu_gate.estimator import advantage, binomial_states
from experiments.nonlinear_oracle.oracle import state_probabilities


def moments(p, total):
    p=np.atleast_2d(np.asarray(p,dtype=float))
    if p.shape[1]!=4 or total not in [8,16,32] or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Finite K4 probabilities and total in {8,16,32} required')
    counts=np.full(4,total//4)
    states,_=binomial_states(counts,np.full(4,.5))
    prob=state_probabilities(states,counts,p)
    adv=advantage(states/counts); target=advantage(p)
    energy=prob@np.mean(adv**2,axis=1)
    mean=prob@adv
    cross=np.mean(mean*target,axis=1)
    target_energy=np.mean(target**2,axis=1)
    optimal=np.clip(np.divide(cross,energy,out=np.zeros_like(cross),where=energy>0),0,1)
    out={}
    for name,c in [('plugin',np.ones(len(p))),('fixed-half',np.full(len(p),.5)),('zero',np.zeros(len(p))),('oracle-scalar',optimal)]:
        bias=np.mean((c[:,None]*mean-target)**2,axis=1)
        variance=c*c*(energy-np.mean(mean**2,axis=1))
        risk=c*c*energy-2*c*cross+target_energy
        out[name]={'mse':np.maximum(risk,0),'bias_squared':bias,'variance':variance,'coefficient':c}
    return {'methods':out,'energy':energy,'alignment':cross,'target_energy':target_energy,'states':states,'probabilities':prob}
