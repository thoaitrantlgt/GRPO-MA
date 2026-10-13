"""Exact nonlinear credit second moments for precommitted count allocations."""
import itertools
import numpy as np
from scipy.stats import binom
from experiments.sequential_sampling.core import target


def allocations(k,budget,m0):
    if k<2 or m0<1 or k*m0>budget:raise ValueError('Infeasible covered budget')
    def recurse(prefix,left):
        if len(prefix)==k-1:
            if left>=m0:yield prefix+[left]
            return
        for count in range(m0,left-m0*(k-len(prefix)-1)+1):
            yield from recurse(prefix+[count],left-count)
    return np.array(list(recurse([],budget)),dtype=int)


def geometries():
    v=np.array([1.,1.,-1.,-1.]);scale=np.array([.5,1,2,4])**2
    names=['orthogonal','aligned-null','opposed','unequal-norm']
    return names,np.array([np.eye(4),np.ones((4,4)),np.outer(v,v),np.diag(4*scale/scale.sum())])


def moments(n,p):
    n=np.asarray(n,dtype=int);p=np.asarray(p,dtype=float)
    if p.ndim!=2 or len(n)!=p.shape[1] or np.any(n<1) or np.any((p<0)|(p>1)):
        raise ValueError('Invalid counts or probability vectors')
    successes=np.array(list(itertools.product(*(range(int(v)+1) for v in n))))
    credit=target(successes/n);truth=target(p)
    weights=np.ones((len(p),len(successes)))
    for j in range(len(n)):
        pmf=binom.pmf(np.arange(n[j]+1)[None,:],n[j],p[:,j,None])
        weights*=pmf[:,successes[:,j]]
    features=np.concatenate([credit,np.einsum('si,sj->sij',credit,credit).reshape(len(credit),-1)],axis=1)
    expectation=weights@features
    mean_credit=expectation[:,:len(n)]
    second=expectation[:,len(n):].reshape(len(p),len(n),len(n))
    matrix=second-np.einsum('vi,vj->vij',mean_credit,truth)-np.einsum('vi,vj->vij',truth,mean_credit)+np.einsum('vi,vj->vij',truth,truth)
    return matrix,mean_credit-truth


def update_risks(matrix,grams):
    return np.einsum('vij,gij->vg',matrix,grams)
