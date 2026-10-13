"""Exact normalized-credit risk under fixed synthetic PSD score geometry."""
import itertools
import numpy as np
from experiments.cpu_gate import estimator as e
from experiments.nonlinear_oracle.oracle import compositions,state_probabilities

NAMES=('isotropic','aligned-null','equicorrelated-control','contrast-rank1',
       'heterogeneous-diagonal','paired-correlated','dense-correlated')
METHODS=('uniform','variance-only','JADA','gradient-JADA','gradient-surrogate-oracle',
         'fixed-gradient-oracle','adaptive-gradient-oracle','fixed-scalar-oracle','adaptive-scalar-oracle')


def grams():
    identity=np.eye(4)/4; ones=np.ones((4,4))/4
    contrast=np.array([1.,-1.,1.,-1.])/2
    pairs=np.array([[1,.7,-.2,-.2],[.7,1,-.2,-.2],[-.2,-.2,1,.7],[-.2,-.2,.7,1]])/4
    scores=np.array([[1,.2,.4,-.3],[.1,1,.2,.5],[.5,-.2,.8,.1]])
    dense=scores.T@scores; dense/=np.trace(dense)
    matrices=np.array([identity,ones,.5*identity+.5*ones,np.outer(contrast,contrast),
                       np.diag([.55,.25,.15,.05]),pairs,dense])
    if np.any(np.linalg.eigvalsh(matrices)<-1e-13) or not np.allclose(np.trace(matrices,axis1=1,axis2=2),1):
        raise ArithmeticError('Invalid declared PSD trace-one geometry')
    return matrices


def local_coefficients(mu,variance,matrices):
    mu=np.atleast_2d(mu); variance=np.atleast_2d(variance)
    j=np.array([e.jacobian(x) for x in mu])
    diagonal=np.einsum('hij,gik,hkj->ghj',j,matrices,j)
    return np.maximum(diagonal,0)*variance[None]


def risk_grid(p,total,m0,matrices=None):
    p=np.atleast_2d(np.asarray(p,dtype=float)); matrices=grams() if matrices is None else np.asarray(matrices,dtype=float)
    if p.shape[1]!=4 or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Finite K4 Bernoulli probabilities required')
    if not isinstance(m0,(int,np.integer)) or m0<1:
        raise ValueError('Positive integer pilot size required')
    if matrices.ndim!=3 or matrices.shape[1:]!=(4,4) or np.any(~np.isfinite(matrices)) or not np.allclose(matrices,matrices.transpose(0,2,1),atol=1e-13) or np.any(np.linalg.eigvalsh(matrices)<-1e-13):
        raise ValueError('Finite symmetric PSD Grams required')
    project=np.eye(4)-np.ones((4,4))/4
    effective=np.einsum('ij,gjk,kl->gil',project,matrices,project)
    null=np.max(np.abs(effective),axis=(1,2))<1e-14
    effective[null]=0
    allocations=compositions(total-4*m0,4)
    pilots=np.array(list(itertools.product(range(m0+1),repeat=4)),dtype=int)
    weights=state_probabilities(pilots,np.full(4,m0),p)
    target=e.advantage(p); gt=np.einsum('gij,vj->vgi',effective,target)
    target_energy=np.einsum('vi,vgi->vg',target,gt)
    risks=np.empty((len(p),len(matrices),len(pilots),len(allocations)))
    lam=4*m0/total
    for c,m in enumerate(allocations):
        main=np.array(list(itertools.product(*[range(int(n)+1) for n in m])),dtype=int)
        prob=state_probabilities(main,m,p)
        values=lam*pilots[:,None,:]/m0+(1-lam)*main[None,:,:]/m
        adv=e.advantage(values)
        energy=np.einsum('hti,gij,htj->htg',adv,effective,adv)
        expected_energy=(prob@energy.transpose(1,0,2).reshape(len(main),-1)).reshape(len(p),len(pilots),len(matrices)).transpose(0,2,1)
        expected_adv=(prob@adv.transpose(1,0,2).reshape(len(main),-1)).reshape(len(p),len(pilots),4)
        risks[:,:,:,c]=expected_energy+target_energy[:,:,None]-2*np.einsum('vhi,vgi->vgh',expected_adv,gt)
    if np.min(risks)<-1e-11:
        raise ArithmeticError('Negative exact gradient risk')
    risks=np.maximum(risks,0); risks[:,null]=0
    bound=16*np.linalg.eigvalsh(matrices).max(axis=1)
    if np.any(risks>bound[None,:,None,None]+1e-10):
        raise ArithmeticError('Risk exceeds fixed-Gram norm bound')
    return {'p':p,'matrices':matrices,'effective_matrices':effective,'null_grams':null,
            'allocations':allocations,'pilots':pilots,'pilot_probabilities':weights,'conditional_risks':risks}


def solve(grid,total,m0):
    p=grid['p']; matrices=grid['matrices']; risks=grid['conditional_risks']; weights=grid['pilot_probabilities']
    pilots=grid['pilots']; allocations=grid['allocations']; lookup={tuple(m):i for i,m in enumerate(allocations)}
    n,g,h,c=risks.shape
    selected={name:np.empty((n,g,h),dtype=int) for name in METHODS}
    mean=(pilots+.5)/(m0+1)
    variance=(pilots+.5)*(m0-pilots+.5)/((m0+1)*(m0+2))
    pilot_counts=e.allocate(local_coefficients(mean,variance,matrices),total-4*m0)
    for v,q in enumerate(p):
        old=e.allocations(pilots,q,m0,total)
        true_counts=e.allocate(local_coefficients(q,q*(1-q),matrices)[:,0,:],total-4*m0)
        for name in ['uniform','variance-only','JADA']:
            selected[name][v]=np.array([lookup[tuple(m)] for m in old[name]])[None]
        for gg in range(g):
            selected['gradient-JADA'][v,gg]=[lookup[tuple(m)] for m in pilot_counts[gg]]
            selected['gradient-surrogate-oracle'][v,gg]=lookup[tuple(true_counts[gg])]
    # Identical scalar geometry controls use original tie handling, avoiding a spurious tie-breaking ablation.
    for gg in [0,2]:
        if g==len(NAMES) and np.allclose(matrices[gg],grams()[gg]):
            selected['gradient-JADA'][:,gg]=selected['JADA'][:,gg]
            for v,q in enumerate(p):
                old=e.allocations(pilots,q,m0,total)
                selected['gradient-surrogate-oracle'][v,gg]=lookup[tuple(old['oracle'][0])]
    unconditional=np.einsum('vh,vghc->vgc',weights,risks)
    selected['fixed-gradient-oracle'][:]=unconditional.argmin(2)[:,:,None]
    selected['adaptive-gradient-oracle'][:]=risks.argmin(3)
    selected['fixed-scalar-oracle'][:]=unconditional[:,0].argmin(1)[:,None,None]
    selected['adaptive-scalar-oracle'][:]=risks[:,0].argmin(2)[:,None,:]
    mse={}
    for name,idx in selected.items():
        conditional=np.take_along_axis(risks,idx[:,:,:,None],axis=3)[:,:,:,0]
        mse[name]=np.einsum('vh,vgh->vg',weights,conditional)
    if any(np.any(mse['adaptive-gradient-oracle']>value+1e-12) for value in mse.values()):
        raise ArithmeticError('Global weighted oracle dominance failed')
    return {**grid,'unconditional_risks':unconditional,'selected':selected,'mse':mse}
