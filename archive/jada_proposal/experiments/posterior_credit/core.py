"""Frozen finite-prior conditional mean credit and exact fixed-Gram risk."""
import itertools
import numpy as np
from scipy.stats import qmc
from scipy.special import betaincinv
from experiments.nonlinear_oracle.oracle import state_probabilities
from experiments.gradient_gate.core import grams


def target(p,tau=1e-4):
    p=np.asarray(p,dtype=float)
    if p.shape[-1]!=4 or np.any(~np.isfinite(p)):
        raise ValueError('Finite K4 values required')
    z=p-p.mean(axis=-1,keepdims=True)
    return z/np.maximum(np.std(p,axis=-1,ddof=1,keepdims=True),tau)


def table_from_support(points,n=4):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=4 or not len(points) or np.any((points<0)|(points>1)):
        raise ValueError('Nonempty finite-prior Bernoulli support required')
    states=np.array(list(itertools.product(range(n+1),repeat=4)),dtype=int)
    mass=np.zeros(len(states)); numerator=np.zeros((len(states),4)); second=np.zeros((4,4))
    for start in range(0,len(points),1024):
        p=points[start:start+1024]; a=target(p)
        prob=state_probabilities(states,np.full(4,n),p)
        mass+=prob.sum(0)/len(points); numerator+=prob.T@a/len(points)
        second+=a.T@a/len(points)
    if np.any(mass<=0):
        raise ValueError('All count histories require positive prior predictive mass')
    return {'states':states,'mass':mass,'numerator':numerator,'second':second,
            'credit':numerator/mass[:,None]}


def fit(config):
    permutations=np.array(list(itertools.permutations(range(4))))
    supports=[]; pieces=[]
    for replicate in range(config['scrambles']):
        u=qmc.Sobol(4,scramble=True,seed=config['seed']+replicate).random_base2(int(np.log2(config['points_per_scramble'])))
        p=betaincinv(.5,.5,u)
        support=p[:,permutations].reshape(-1,4); supports.append(support)
        pieces.append(table_from_support(support))
    mass=np.mean([x['mass'] for x in pieces],axis=0)
    numerator=np.mean([x['numerator'] for x in pieces],axis=0)
    credit=numerator/mass[:,None]; plugin=target(pieces[0]['states']/4)
    matrices=grams(); primary=matrices[3:].mean(axis=0)
    energy=np.sum(mass*np.einsum('hi,ij,hj->h',plugin,primary,plugin))
    alignment=np.sum(np.einsum('hi,ij,hj->h',plugin,primary,numerator))
    scalar=float(np.clip(alignment/energy,0,1))
    half=config['scrambles']//2
    first=np.sum([x['numerator'] for x in pieces[:half]],axis=0)/np.sum([x['mass'] for x in pieces[:half]],axis=0)[:,None]
    last=np.sum([x['numerator'] for x in pieces[half:]],axis=0)/np.sum([x['mass'] for x in pieces[half:]],axis=0)[:,None]
    return {'states':pieces[0]['states'],'mass':mass,'numerator':numerator,
            'second':np.mean([x['second'] for x in pieces],axis=0),'credit':credit,'scalar':scalar,
            'support':np.concatenate(supports),'half_difference':float(np.max(np.abs(first-last))),
            'plugin':plugin}


def exact_metrics(p,table,matrices):
    p=np.atleast_2d(np.asarray(p,dtype=float)); counts=np.full(4,4)
    prob=state_probabilities(table['states'],counts,p); truth=target(p)
    methods={'plugin':table['plugin'],'prior-scale':table['scalar']*table['plugin'],'posterior-credit':table['credit']}
    out={}
    for name,credit in methods.items():
        delta=credit[None,:,:]-truth[:,None,:]
        mean=prob@credit-truth
        mse=np.einsum('nh,nhi,gij,nhj->ng',prob,delta,matrices,delta)
        bias=np.einsum('ni,gij,nj->ng',mean,matrices,mean)
        truth_energy=np.einsum('ni,gij,nj->ng',truth,matrices,truth)
        state_energy=np.maximum(np.einsum('hi,gij,hj->hg',credit,matrices,credit),0)
        alignment=np.einsum('hi,gij,nj->nhg',credit,matrices,truth)
        denom=np.sqrt(np.maximum(truth_energy,0)[:,None,:]*state_energy[None,:,:])
        valid=denom>1e-12
        cosine=np.divide(alignment,denom,out=np.zeros_like(alignment),where=valid)
        out[name]={'mse':np.maximum(mse,0),'bias_squared':np.maximum(bias,0),'variance':np.maximum(mse-bias,0),
                   'target_update_energy':np.maximum(truth_energy,0),
                   'estimated_update_energy':prob@state_energy,
                   'expected_target_alignment':np.einsum('nh,nhg->ng',prob,alignment),
                   'direction_defined_mass':np.einsum('nh,nhg->ng',prob,valid),
                   'direction_cosine_numerator':np.einsum('nh,nhg->ng',prob,np.clip(cosine,-1,1))}
    return out
