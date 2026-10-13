"""Nonlinear one-action risk acquisition; no evaluator probabilities in policy API."""
from functools import lru_cache
import itertools
import time
import numpy as np
from scipy.special import betaincinv, logsumexp
from scipy.stats import qmc


def target(p, tau=1e-4):
    p=np.asarray(p,dtype=float)
    if p.ndim<1 or p.shape[-1]<2 or np.any(~np.isfinite(p)):
        raise ValueError('Finite vectors with K>=2 required')
    return (p-p.mean(axis=-1,keepdims=True))/np.maximum(np.std(p,axis=-1,ddof=1,keepdims=True),tau)


def jacobian(p,tau=1e-4):
    p=np.asarray(p,dtype=float);k=len(p);z=p-p.mean();r=np.std(p,ddof=1)
    project=np.eye(k)-np.ones((k,k))/k
    if r<tau:return project/tau
    if r==tau:raise ValueError('Jacobian undefined at floor kink')
    return project/r-np.outer(z,z)/((k-1)*r**3)


def support(k, seed=2828422027):
    """Nested exchangeable support. Every support decision is frozen before evaluation."""
    count=128 if k==2 else 16
    permutations=np.array(list(itertools.permutations(range(k))))
    points=[];low=[];halves=[]
    for rep in range(2):
        u=qmc.Sobol(k,scramble=True,seed=seed+rep).random_base2(int(np.log2(count)))
        p=betaincinv(.5,.5,u)[:,permutations].reshape(-1,k)
        start=sum(len(x) for x in points);points.append(p)
        halves.append(np.arange(start,start+len(p)))
        low.extend(range(start,start+len(p)//2))
    p=np.concatenate(points)
    return p,[np.arange(len(p)),np.array(low),*halves]


def child(n,s,j,x):
    nn=list(n);ss=list(s);nn[j]+=1;ss[j]+=x
    return tuple(nn),tuple(ss)


class RiskModel:
    def __init__(self,points,budget,subsets=None,deadline_seconds=None):
        self.points=np.asarray(points,dtype=float)
        if self.points.ndim!=2 or np.any((self.points<=0)|(self.points>=1)):
            raise ValueError('Prior support must be strictly interior')
        self.k=self.points.shape[1];self.budget=int(budget)
        self.truth=target(self.points);self.logp=np.log(self.points);self.log1p=np.log1p(-self.points)
        self.subsets=subsets if subsets is not None else [np.arange(len(points))]
        self.deadline=time.perf_counter()+deadline_seconds if deadline_seconds else float('inf')
        self.uniform_risk=lru_cache(maxsize=30000)(self._uniform_risk)
        self.scores=lru_cache(maxsize=30000)(self._scores)
        self.choose=lru_cache(maxsize=60000)(self.choose)
        self.policy_calls=0

    def check_deadline(self):
        if time.perf_counter()>self.deadline:raise TimeoutError('Declared configuration CPU ceiling exceeded')

    def posterior(self,n,s):
        n=np.asarray(n);s=np.asarray(s)
        if np.any(n<1) or np.any(s<0) or np.any(s>n):raise ValueError('Invalid covered history')
        logw=self.logp@s+self.log1p@(n-s)
        return np.exp(logw-logsumexp(logw))

    def _uniform_risk(self,n,s):
        self.check_deadline()
        if sum(n)>self.budget:raise ValueError('Budget exceeded')
        if sum(n)==self.budget:
            delta=target(np.asarray(s)/np.asarray(n))-self.truth
            return np.sum(delta*delta,axis=1)
        j=int(np.argmin(n));p=self.points[:,j]
        return p*self.uniform_risk(*child(n,s,j,1))+(1-p)*self.uniform_risk(*child(n,s,j,0))

    def _scores(self,n,s):
        if sum(n)>=self.budget:raise ValueError('No acquisition budget remains')
        base=self.uniform_risk(n,s);weights=self.posterior(n,s)
        reference=[];gains=[];confidence=[]
        delta=np.zeros((len(self.points),self.k));uj=int(np.argmin(n))
        for j in range(self.k):
            if j==uj:continue  # Exact baseline identity; do not prioritize roundoff.
            p=self.points[:,j]
            q=p*self.uniform_risk(*child(n,s,j,1))+(1-p)*self.uniform_risk(*child(n,s,j,0))
            delta[:,j]=base-q
        for ix in self.subsets:
            w=weights[ix];w=w/w.sum()
            reference.append(float(w@base[ix]));gains.append(w@delta[ix])
            confidence.append(w@(delta[ix]>1e-12))
        reference=np.array(reference);gain=np.array(gains);confidence=np.array(confidence)
        numerical_change=float(np.max(np.abs(reference-reference[0]))/max(reference[0],1e-12))
        return {'gain':gain,'confidence':confidence,'reference':reference,
                'numerical_change':numerical_change,'ess':float(1/(weights@weights))}

    def choose(self,n,s,method):
        self.policy_calls+=1;uj=int(np.argmin(n))
        if method=='uniform':return uj,'uniform'
        if method not in ['ungated','gated']:raise ValueError('Unknown method')
        scores=self.scores(n,s)
        if scores['numerical_change']>.01:return uj,'numerical-fallback'
        eligible=np.all(scores['gain']>1e-12,axis=0)
        if method=='gated':eligible &= np.all(scores['confidence']>=.90,axis=0)
        if not np.any(eligible):return uj,'reliability-fallback' if method=='gated' else 'no-positive-gain'
        j=int(np.argmax(np.where(eligible,scores['gain'][0],-np.inf)))
        return j,'preferred'


def simulate(model,streams,m0,method):
    """Streams are observed Bernoulli reward arrays; policy never receives true p."""
    streams=np.asarray(streams,dtype=bool)
    if streams.shape!=(model.k,model.budget) or model.k*m0>=model.budget:
        raise ValueError('Streams or initial coverage invalid')
    n=[m0]*model.k;s=streams[:,:m0].sum(axis=1).astype(int).tolist();reasons={}
    for _ in range(model.budget-model.k*m0):
        j,reason=model.choose(tuple(n),tuple(s),method)
        reasons[reason]=reasons.get(reason,0)+1
        s[j]+=int(streams[j,n[j]]);n[j]+=1
    return {'n':n,'s':s,'credit':target(np.asarray(s)/np.asarray(n)),'reasons':reasons}
