"""Exact nonlinear moments and first-order Taylor remainder decomposition."""
import itertools
import numpy as np
from experiments.cpu_gate import estimator as e
from experiments.nonlinear_oracle.oracle import solve, state_probabilities


def diagnose(p, total=16, m0=2):
    p = np.atleast_2d(np.asarray(p, dtype=float))
    result = solve(p, total, m0)
    pilots = result['pilots']; weights = result['pilot_probabilities']
    n, h = weights.shape; k = p.shape[1]; lam = k*m0/total
    target = e.advantage(p); jac = np.array([e.jacobian(q) for q in p])
    methods = ['uniform', 'JADA', 'local-surrogate-oracle', 'fixed-nonlinear-oracle', 'adaptive-nonlinear-oracle']
    moments = {name: {'delta': np.zeros((n,k)), 'raw_mean': np.zeros((n,k)),
                     'linear_mean': np.zeros((n,k)), **{key: np.zeros(n) for key in ['mse','linear','cross','remainder']}}
               for name in methods}
    for a, m in enumerate(result['allocations']):
        states = np.array(list(itertools.product(*[range(int(x)+1) for x in m])))
        main_prob = state_probabilities(states, m, p)
        val = lam*pilots[:,None,:]/m0+(1-lam)*states[None,:,:]/m
        delta = e.advantage(val)[None,:,:,:]-target[:,None,None,:]
        err = val[None,:,:,:]-p[:,None,None,:]
        linear = np.einsum('nhtj,nij->nhti',err,jac)
        rem = delta-linear
        conditional = {'delta': np.einsum('nt,nhtk->nhk',main_prob,delta),
                       'raw_mean': np.einsum('nt,nhtk->nhk',main_prob,err),
                       'linear_mean': np.einsum('nt,nhtk->nhk',main_prob,linear)}
        for key, value in [('mse',np.mean(delta**2,axis=-1)),('linear',np.mean(linear**2,axis=-1)),
                           ('cross',2*np.mean(linear*rem,axis=-1)),('remainder',np.mean(rem**2,axis=-1))]:
            conditional[key] = np.einsum('nt,nht->nh',main_prob,value)
        for name in methods:
            selected_weight = weights*(result['selected'][name]==a)
            for key, value in conditional.items():
                moments[name][key] += np.einsum('nh,nhk->nk',selected_weight,value) if value.ndim==3 else np.sum(selected_weight*value,axis=1)
    rows = []
    for name, moment in moments.items():
        selected_m = result['allocations'][result['selected'][name]]
        inv = np.sum(weights[:,:,None]/selected_m,axis=1)
        covariance = p*(1-p)*(lam**2/m0+(1-lam)**2*inv)
        surrogate = np.sum(e.column_norms(p)*covariance,axis=1)/k
        dispersion = np.mean((p-p.mean(axis=1,keepdims=True))**2,axis=1)
        for i in range(n):
            bias2 = np.mean(moment['delta'][i]**2)
            row = {key: float(moment[key][i]) for key in ['mse','linear','cross','remainder']}
            row.update({'index':i,'method':name,'bias_squared':float(bias2),'variance':float(moment['mse'][i]-bias2),
                        'surrogate':float(surrogate[i]),'raw_bias_max':float(np.max(np.abs(moment['raw_mean'][i]))),
                        'linear_mean_max':float(np.max(np.abs(moment['linear_mean'][i]))),
                        'dispersion':float(dispersion[i]),
                        'noise_to_dispersion':float((1-1/k)*np.mean(covariance[i])/(dispersion[i]+e.EPS)),
                        'calibration_error':float(abs(surrogate[i]-moment['mse'][i])/moment['mse'][i])})
            rows.append(row)
    return rows, result
