"""Finite reward-state propagation and paired validation; policy only sees counts."""
import itertools
from collections import Counter
import time
import numpy as np
from scipy.special import comb
from .core import target, jacobian, child, simulate


def outcome_mass(states,counts,p):
    states=np.asarray(states);counts=np.asarray(counts);p=np.asarray(p)
    return np.prod(comb(counts,states)[None,:,:]*p[:,None,:]**states[None,:,:]
                   *(1-p[:,None,:])**(counts-states)[None,:,:],axis=-1)


def initial_states(k,m0):
    return list(itertools.product(range(m0+1),repeat=k))


def metrics(mass,credits,truth,counts=None):
    mass=np.asarray(mass);credits=np.asarray(credits)
    delta=credits[None,:,:]-truth[:,None,:]
    second=np.einsum('vh,vhi,vhj->vij',mass,delta,delta)
    mean=mass@credits-truth
    risk=np.trace(second,axis1=1,axis2=2);bias=np.sum(mean*mean,axis=1)
    result={'risk':risk,'bias_squared':bias,'variance':np.maximum(risk-bias,0),'second':second,'mean_error':mean,
            'mass_error':float(np.max(np.abs(mass.sum(axis=1)-1)))}
    if counts is not None:result['expected_counts']=mass@np.asarray(counts)
    return result


def average_permutations(result,permutations):
    """Return metrics of the randomized policy in original thought coordinates."""
    count=len(permutations);v=len(result['risk'])//count;k=len(permutations[0]);out={}
    for key,value in result.items():
        if isinstance(value,np.ndarray) and value.shape[0]==v*count:
            shaped=value.reshape((v,count)+value.shape[1:])
            if key in ['mean_error','expected_counts','second']:
                rotated=[]
                for ix,perm in enumerate(permutations):
                    inv=np.argsort(perm)
                    piece=shaped[:,ix]
                    rotated.append(piece[:,inv,:][:,:,inv] if key=='second' else piece[:,inv])
                out[key]=np.mean(rotated,axis=0)
            else:out[key]=shaped.mean(axis=1)
        elif key=='reasons':out[key]={r:a.reshape(v,count).mean(axis=1) for r,a in value.items()}
        else:out[key]=value
    out['bias_squared']=np.sum(out['mean_error']**2,axis=1)
    out['variance']=np.maximum(out['risk']-out['bias_squared'],0)
    return out


def exact_sequential(model,p,m0,method):
    k=model.k;n=(m0,)*k;states=initial_states(k,m0)
    mass=outcome_mass(states,n,p)
    active={(n,s):mass[:,i] for i,s in enumerate(states)}
    reasons={};history_counts=Counter();start=time.perf_counter()
    for depth in range(k*m0,model.budget):
        model.check_deadline();following={}
        for (nn,ss),weight in active.items():
            j,reason=model.choose(nn,ss,method)
            if reason not in reasons:reasons[reason]=np.zeros(len(p))
            reasons[reason]+=weight;history_counts[reason]+=1
            for x in [0,1]:
                next_state=child(nn,ss,j,x)
                contribution=weight*(p[:,j] if x else 1-p[:,j])
                if next_state in following:following[next_state]+=contribution
                else:following[next_state]=contribution
        active=following
        print(f'  {method}: depth {depth+1}/{model.budget}, {len(active)} states, {time.perf_counter()-start:.1f}s',flush=True)
    mass=np.stack(list(active.values()),axis=1)
    credits=np.array([target(np.asarray(s)/np.asarray(n)) for n,s in active])
    counts=[n for n,s in active]
    result=metrics(mass,credits,target(p),counts)
    result['reasons']=reasons;result['history_counts']=dict(history_counts)
    result['terminal_states']=len(active);result['seconds']=time.perf_counter()-start
    return result


def allocate(coefficients,budget):
    counts=np.ones(len(coefficients),dtype=int)
    for _ in range(budget-len(counts)):
        marginal=coefficients/(counts*(counts+1))
        eligible=np.flatnonzero(marginal==marginal.max())
        j=eligible[np.argmin(counts[eligible])];counts[j]+=1
    return counts


def fixed_counts(s,m0,budget,method):
    s=np.asarray(s);k=len(s);remain=budget-k*m0
    if method=='variance-only' or method=='JADA':
        alpha=.5+s;beta=.5+m0-s
        coeff=alpha*beta/((m0+1)*(m0+2))
        if method=='JADA':coeff*=np.sum(jacobian(alpha/(m0+1))**2,axis=0)
        return m0+allocate(coeff,remain)
    raise ValueError('Unknown fixed-pilot comparator')


def _exact_fixed(p,m0,budget,method):
    k=p.shape[1];pilots=initial_states(k,m0)
    pilot_mass=outcome_mass(pilots,[m0]*k,p)
    masses=[];credits=[];allcounts=[]
    for ix,s in enumerate(pilots):
        n=fixed_counts(s,m0,budget,method);extra=n-m0
        main=list(itertools.product(*(range(c+1) for c in extra)))
        mass=outcome_mass(main,extra,p)*pilot_mass[:,ix,None]
        masses.append(mass);credits.extend(target((np.array(main)+s)/n))
        allcounts.extend([n]*len(main))
    return metrics(np.concatenate(masses,axis=1),np.array(credits),target(p),allcounts)


def exact_fixed(p,m0,budget,method):
    pieces=[_exact_fixed(p[start:start+128],m0,budget,method) for start in range(0,len(p),128)]
    return {key:np.concatenate([x[key] for x in pieces]) if isinstance(pieces[0][key],np.ndarray)
            else max(x[key] for x in pieces) for key in pieces[0]}


def paired_mc(model,p,m0,exact,episodes,seed):
    rng=np.random.default_rng(seed);truth=target(p)
    methods=['uniform','ungated','gated'];estimates={};checks=[]
    start=time.perf_counter()
    for ix,prob in enumerate(p):
        loss={name:[] for name in methods}
        for _ in range(episodes):
            # Independent thought streams, same stream prefixes for every policy.
            streams=rng.random((model.k,model.budget))<prob[:,None]
            permutation=rng.permutation(model.k);inverse=np.argsort(permutation)
            for name in methods:
                credit=simulate(model,streams[permutation],m0,name)['credit'][inverse]
                loss[name].append(float(np.sum((credit-truth[ix])**2)))
        baseline=np.array(loss['uniform'])
        for name in methods:
            delta=baseline-np.array(loss[name]);mean=float(delta.mean())
            se=float(delta.std(ddof=1)/np.sqrt(episodes))
            expected=float(exact['uniform']['risk'][ix]-exact[name]['risk'][ix])
            z=(mean-expected)/se if se>1e-15 else (0. if abs(mean-expected)<1e-12 else None)
            bound=4*(model.k-1)*np.sqrt(2*np.log(2/.05)/episodes)
            checks.append({'vector_index':ix,'method':name,'mc_gain':mean,'exact_gain':expected,
                           'se':se,'z':z,'zero_se_unresolved':se<=1e-15 and abs(mean-expected)>=1e-12,
                           'hoeffding_low':mean-bound,'hoeffding_high':mean+bound,
                           'exact_in_hoeffding_interval':abs(mean-expected)<=bound,
                           'absolute_discrepancy':abs(mean-expected)})
        if (ix+1)%24==0:print(f'  paired MC: {ix+1}/{len(p)} vectors',flush=True)
    return checks,time.perf_counter()-start
