"""Globally optimal nonlinear-risk allocations by exact finite enumeration."""
import itertools
import math
import numpy as np
from experiments.cpu_gate import estimator as e

METHODS = ('uniform', 'variance-only', 'JADA', 'local-surrogate-oracle',
           'fixed-nonlinear-oracle', 'adaptive-nonlinear-oracle')


def compositions(total, k):
    if not isinstance(total, (int, np.integer)) or total < k or k < 1:
        raise ValueError('Integer total >= positive K is required')
    def recurse(left, dimensions):
        if dimensions == 1:
            yield (left,)
        else:
            for first in range(1, left-dimensions+2):
                for rest in recurse(left-first, dimensions-1):
                    yield (first,)+rest
    return np.array(list(recurse(total, k)), dtype=int)


def state_probabilities(states, counts, p):
    """Independent binomial count PMFs, vectorized over true p vectors."""
    weights = np.ones((len(p), len(states)))
    for j, n in enumerate(counts):
        successes = states[:, j]
        coefficient = np.array([math.comb(int(n), int(s)) for s in successes])
        weights *= coefficient[None, :]*p[:, j, None]**successes[None, :] \
            *(1-p[:, j, None])**(n-successes)[None, :]
    if not np.allclose(weights.sum(1), 1, rtol=1e-12, atol=1e-14):
        raise ArithmeticError('Binomial state probabilities do not sum to one')
    return weights


def conditional_risk_grid(p, total, m0, reuse_lambda=None):
    p = np.atleast_2d(np.asarray(p, dtype=float))
    if p.shape[1] != 4 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('This gate requires finite K4 Bernoulli probabilities')
    if m0 < 1 or not isinstance(m0, (int, np.integer)):
        raise ValueError('A positive integer pilot size is required')
    k = p.shape[1]
    lam = k*m0/total if reuse_lambda is None else float(reuse_lambda)
    if not np.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError('Precommitted reuse lambda must be in [0,1]')
    candidates = compositions(total-k*m0, k)
    pilots = np.array(list(itertools.product(range(m0+1), repeat=k)), dtype=int)
    pilot_weights = state_probabilities(pilots, np.full(k, m0), p)
    target = e.advantage(p)
    target_energy = np.mean(target**2, axis=1)
    risks = np.empty((len(p), len(pilots), len(candidates)))
    for index, m in enumerate(candidates):
        main = np.array(list(itertools.product(*[range(int(n)+1) for n in m])), dtype=int)
        probabilities = state_probabilities(main, m, p)
        val = lam*pilots[:, None, :]/m0+(1-lam)*main[None, :, :]/m
        adv = e.advantage(val)
        expected_energy = probabilities@np.mean(adv**2, axis=2).T
        expected_adv = (probabilities@adv.transpose(1, 0, 2).reshape(len(main), -1)).reshape(len(p), len(pilots), k)
        risks[:, :, index] = expected_energy+target_energy[:, None] \
            -2*np.mean(expected_adv*target[:, None, :], axis=2)
    if np.min(risks) < -1e-12 or np.max(risks) > 4+1e-12:
        raise ArithmeticError('Normalized-advantage risk outside its mathematical bounds')
    risks = np.maximum(risks, 0.)
    return {'p': p, 'allocations': candidates, 'pilots': pilots,
            'pilot_probabilities': pilot_weights, 'conditional_risks': risks}


def solve(p, total, m0):
    grid = conditional_risk_grid(p, total, m0)
    p, candidates, pilots, pilot_weights, risks = [grid[key] for key in
        ['p', 'allocations', 'pilots', 'pilot_probabilities', 'conditional_risks']]
    k = p.shape[1]
    unconditional_risks = np.einsum('vp,vpc->vc', pilot_weights, risks)
    fixed_index = unconditional_risks.argmin(1)
    adaptive_index = risks.argmin(2)
    lookup = {tuple(m): index for index, m in enumerate(candidates)}
    selected = {method: np.empty((len(p), len(pilots)), dtype=int) for method in METHODS}
    for i, q in enumerate(p):
        counts = e.allocations(pilots, q, m0, total)
        for method in ['uniform', 'variance-only', 'JADA', 'local-surrogate-oracle']:
            old_name = 'oracle' if method == 'local-surrogate-oracle' else method
            selected[method][i] = [lookup[tuple(m)] for m in counts[old_name]]
    selected['fixed-nonlinear-oracle'][:] = fixed_index[:, None]
    selected['adaptive-nonlinear-oracle'][:] = adaptive_index
    mse = {}
    for method, choices in selected.items():
        conditional = np.take_along_axis(risks, choices[:, :, None], axis=2)[:, :, 0]
        mse[method] = np.sum(pilot_weights*conditional, axis=1)
    oracle_risk = mse['adaptive-nonlinear-oracle']
    if any(np.any(oracle_risk > risk+1e-12) for risk in mse.values()):
        raise ArithmeticError('Exhaustive adaptive optimum failed comparator dominance')
    lam = k*m0/total
    bias = lam*(pilot_weights@(pilots/m0)-p)
    selected_m = candidates[adaptive_index]
    inv = np.sum(pilot_weights[:, :, None]/selected_m, axis=1)
    value_covariance = p*(1-p)*(lam**2/m0+(1-lam)**2*inv)
    return {'mse': mse, 'allocations': candidates, 'pilots': pilots,
            'pilot_probabilities': pilot_weights, 'conditional_risks': risks,
            'unconditional_risks': unconditional_risks, 'selected': selected,
            'oracle_value_bias': bias, 'oracle_value_covariance_diagonal': value_covariance}
