"""Exact exchangeable finite-prior Bellman quotient and labelled outcome graph."""
import ctypes
import itertools
import time
import numpy as np
from scipy.special import betaincinv
from scipy.stats import qmc
from .core import target
from .evaluation import outcome_mass


def ladder_support(base_points,seed=2828422027):
    permutations=np.array(list(itertools.permutations(range(4))))
    result=[]
    for rep in range(2):
        u=qmc.Sobol(4,scramble=True,seed=seed+rep).random_base2(int(np.log2(base_points)))
        result.append(betaincinv(.5,.5,u)[:,permutations].reshape(-1,4))
    return np.concatenate(result)


def process_memory():
    """Windows working set, using the standard process-memory API."""
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in
                  ['peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
    if not hasattr(ctypes,'windll'):return None
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    getprocess=ctypes.windll.kernel32.GetCurrentProcess;getprocess.restype=ctypes.c_void_p
    getmemory=ctypes.windll.psapi.GetProcessMemoryInfo
    getmemory.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not getmemory(getprocess(),ctypes.byref(counters),counters.cb):raise OSError('GetProcessMemoryInfo failed')
    return {'rss':int(counters.rss),'peak_rss':int(counters.peak_rss)}


def canonical(n,s):
    return tuple(sorted(zip(n,s)))


def next_key(key,j,x):
    pairs=list(key);n,s=pairs[j];pairs[j]=(n+1,s+x)
    return tuple(sorted(pairs))


class SymmetricControl:
    def __init__(self,points,budget,seconds=600,rss_limit=2*1024**3):
        self.points=np.asarray(points,dtype=float);self.budget=int(budget);self.k=self.points.shape[1]
        if np.any((self.points<=0)|(self.points>=1)):raise ValueError('Interior support required')
        sorted_points=self.points[np.lexsort(self.points.T[::-1])]
        for j in range(self.k-1):
            perm=np.arange(self.k);perm[j:j+2]=perm[j:j+2][::-1];other=self.points[:,perm]
            if not np.array_equal(sorted_points,other[np.lexsort(other.T[::-1])]):raise ValueError('Exchangeable support required')
        self.truth=target(self.points);self.energy=np.sum(self.truth**2,axis=1)
        self.logp=np.log(self.points);self.log1p=np.log1p(-self.points)
        self.values={};self.actions={};self.started=time.perf_counter();self.deadline=self.started+seconds
        self.rss_limit=rss_limit;self.peak_rss=0;self.last_print=self.started;self.calls=0

    def check_deadline(self):
        if time.perf_counter()>self.deadline:raise TimeoutError('Declared level CPU wall ceiling exceeded')
        self.calls+=1
        if self.calls%512==0:
            memory=process_memory()
            if memory:
                self.peak_rss=max(self.peak_rss,memory['peak_rss'])
                if memory['rss']>self.rss_limit:raise MemoryError('Declared process RSS ceiling exceeded')
            now=time.perf_counter()
            if now-self.last_print>5:
                print(f'  Bellman: {len(self.values)} quotient states solved; {now-self.started:.1f}s',flush=True);self.last_print=now

    def posterior_key(self,key):
        n=np.array([x[0] for x in key]);s=np.array([x[1] for x in key])
        logw=np.einsum('ij,j->i',self.logp,s)+np.einsum('ij,j->i',self.log1p,n-s)
        weights=np.exp(logw-logw.max());return weights/weights.sum()

    def terminal_cost(self,key,weights=None):
        weights=self.posterior_key(key) if weights is None else weights
        credit=target(np.array([x[1]/x[0] for x in key]))
        cost=float(credit@credit-2*credit@(weights@self.truth)+weights@self.energy)
        if cost < -1e-10:raise ArithmeticError('Negative posterior MSE')
        return max(cost,0.)

    def _value(self,key):
        if key in self.values:return self.values[key]
        self.check_deadline();total=sum(x[0] for x in key)
        if total>self.budget or any(n<1 or s<0 or s>n for n,s in key):raise ValueError('Invalid Bellman state')
        weights=self.posterior_key(key)
        if total==self.budget:
            value=self.terminal_cost(key,weights);action=-1
        else:
            probabilities=weights@self.points
            choices=[probabilities[j]*self._value(next_key(key,j,1))
                     +(1-probabilities[j])*self._value(next_key(key,j,0)) for j in range(self.k)]
            action=int(np.argmin(choices));value=float(choices[action])
        self.values[key]=value;self.actions[key]=action
        return value

    def value(self,n,s):return self._value(canonical(n,s))

    def choose(self,n,s,method='bayes-control'):
        if method=='uniform':return int(np.argmin(n)),'uniform'
        if method!='bayes-control':raise ValueError('Unknown control')
        key=canonical(n,s);self._value(key);pair=key[self.actions[key]]
        return next(j for j,item in enumerate(zip(n,s)) if item==pair),'bayes-control'

    def save(self,path):
        keys=sorted(self.values)
        np.savez_compressed(path,n=np.array([[x[0] for x in key] for key in keys]),
                            s=np.array([[x[1] for x in key] for key in keys]),
                            value=np.array([self.values[key] for key in keys]),
                            action=np.array([self.actions[key] for key in keys]),budget=self.budget)


class SavedControl:
    def __init__(self,path):
        with np.load(path) as saved:
            self.k=saved['n'].shape[1];self.budget=int(saved['budget'])
            self.actions={canonical(tuple(n),tuple(s)):int(a) for n,s,a in zip(saved['n'],saved['s'],saved['action'])}

    def check_deadline(self):pass

    def choose(self,n,s,method='bayes-control'):
        key=canonical(n,s);pair=key[self.actions[key]]
        return next(j for j,item in enumerate(zip(n,s)) if item==pair),'bayes-control'


def build_graph(model,m0,other=None):
    initial_n=(m0,)*model.k
    states=[(initial_n,s) for s in itertools.product(range(m0+1),repeat=model.k)]
    initial=states.copy();edges=[]
    for depth in range(model.k*m0,model.budget):
        model.check_deadline();actions=[];children=[];following=set();disagrees=[]
        for n,s in states:
            j,_=model.choose(n,s);actions.append(j)
            disagrees.append(other is not None and other.choose(n,s)[0]!=j)
            pair=[]
            for x in [0,1]:
                nn=list(n);ss=list(s);nn[j]+=1;ss[j]+=x;next_state=(tuple(nn),tuple(ss))
                pair.append(next_state);following.add(next_state)
            children.append(pair)
        following=sorted(following);lookup={state:i for i,state in enumerate(following)}
        edges.append({'states':states,'actions':np.array(actions),'zero':np.array([lookup[a[0]] for a in children]),
                      'one':np.array([lookup[a[1]] for a in children]),'next_count':len(following),'disagree':np.array(disagrees)})
        states=following
    return {'initial':initial,'edges':edges,'terminal':states,'m0':m0}


def graph_metrics(model,graph,p,block=256):
    pieces=[];mass_error=0.
    terminal_n=np.array([n for n,s in graph['terminal']]);terminal_s=np.array([s for n,s in graph['terminal']])
    credits=target(terminal_s/terminal_n)
    outer=np.einsum('hi,hj->hij',credits,credits).reshape(len(credits),-1)
    for start in range(0,len(p),block):
        model.check_deadline();prob=np.asarray(p[start:start+block]);truth=target(prob)
        mass=outcome_mass([s for n,s in graph['initial']],[graph['m0']]*model.k,prob)
        disagreement=np.zeros(len(prob))
        for edge in graph['edges']:
            following=np.zeros((len(prob),edge['next_count']))
            probabilities=prob[:,edge['actions']]
            disagreement+=mass[:,edge['disagree']].sum(axis=1)
            np.add.at(following.T,edge['zero'],(mass*(1-probabilities)).T)
            np.add.at(following.T,edge['one'],(mass*probabilities).T)
            mass=following
        mean=mass@credits;second=(mass@outer).reshape(-1,model.k,model.k)
        second=second-np.einsum('vi,vj->vij',mean,truth)-np.einsum('vi,vj->vij',truth,mean)+np.einsum('vi,vj->vij',truth,truth)
        error=mean-truth;risk=np.maximum(np.trace(second,axis1=1,axis2=2),0);bias=np.sum(error**2,axis=1)
        mass_error=max(mass_error,float(np.max(np.abs(mass.sum(axis=1)-1))))
        pieces.append({'risk':risk,'bias_squared':bias,'variance':np.maximum(risk-bias,0),'mean_error':error,'second':second,
                       'expected_counts':mass@terminal_n,'action_disagreement':disagreement/len(graph['edges'])})
    result={key:np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
    result['mass_error']=mass_error;result['terminal_states']=len(credits)
    return result
