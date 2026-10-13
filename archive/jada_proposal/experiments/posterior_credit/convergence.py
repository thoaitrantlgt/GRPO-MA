"""Nested exact joint-integral symmetrization; unchanged posterior-credit model."""
import itertools
import numpy as np
from scipy.stats import qmc
from scipy.special import betaincinv
from .core import table_from_support, target


def symmetrize(table):
    states=table['states']; lookup={tuple(s):i for i,s in enumerate(states)}
    mass=np.zeros_like(table['mass']); num=np.zeros_like(table['numerator']); second=np.zeros_like(table['second'])
    for perm in itertools.permutations(range(4)):
        inverse=np.argsort(perm)
        indexes=np.array([lookup[tuple(s[list(perm)])] for s in states])
        mass+=table['mass'][indexes]/24
        num+=table['numerator'][indexes][:,inverse]/24
        second+=table['second'][np.ix_(perm,perm)]/24
    return {'states':states,'mass':mass,'numerator':num,'second':second,'credit':num/mass[:,None]}


def nested_integrals(points,levels):
    levels=list(levels)
    if not levels or levels!=sorted(set(levels)) or levels[0]<1 or levels[-1]>len(points):
        raise ValueError('Increasing positive nested levels within support required')
    sums=None; previous=0; result={}
    for limit in levels:
        block=table_from_support(points[previous:limit]); count=limit-previous
        if sums is None:
            sums={key:count*block[key] for key in ['mass','numerator','second']}
        else:
            for key in sums:
                sums[key]+=count*block[key]
        result[limit]=symmetrize({'states':block['states'],**{key:value/limit for key,value in sums.items()}})
        previous=limit
    return result


def integrate(config,progress=None):
    levels=config['nested_points_per_scramble']; replicates={level:[] for level in levels}; points=[]
    for rep in range(config['scrambles']):
        u=qmc.Sobol(4,scramble=True,seed=config['seed']+rep).random_base2(int(np.log2(levels[-1])))
        support=betaincinv(.5,.5,u); points.append(support)
        nested=nested_integrals(support,levels)
        for level in levels:
            replicates[level].append(nested[level])
        if progress:
            progress(rep+1)
    tables={}
    for level,pieces in replicates.items():
        mass=np.array([x['mass'] for x in pieces]); num=np.array([x['numerator'] for x in pieces])
        pooled_mass=mass.mean(0); pooled_num=num.mean(0); half=len(pieces)//2
        first=num[:half].sum(0)/mass[:half].sum(0)[:,None]
        last=num[half:].sum(0)/mass[half:].sum(0)[:,None]
        tables[level]={'states':pieces[0]['states'],'mass':pooled_mass,'numerator':pooled_num,
                       'second':np.mean([x['second'] for x in pieces],axis=0),
                       'credit':pooled_num/pooled_mass[:,None],
                       'half_difference':float(np.max(np.abs(first-last))),
                       'replicate_mass':mass,'replicate_numerator':num,
                       'plugin':target(pieces[0]['states']/4)}
    return tables,np.array(points)
