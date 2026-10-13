"""CPU Bernoulli diagnostics. No torch, model calls or GPU dependencies."""
import itertools
import math
import numpy as np

EPS = 1e-6
ALLOCATIONS = ('uniform', 'variance-only', 'JADA', 'oracle')
REUSE = ('main-only', 'naive', 'fixed-0.5', 'budget-precommitted')


def advantage(mu, eps=EPS):
    mu = np.asarray(mu)
    z = mu - mu.mean(axis=-1, keepdims=True)
    return z / np.sqrt(np.mean(z*z, axis=-1, keepdims=True)+eps)


def jacobian(mu, eps=EPS):
    mu = np.asarray(mu)
    k = len(mu)
    z = mu-mu.mean()
    s = np.sqrt(np.mean(z*z)+eps)
    return (np.eye(k)-np.ones((k, k))/k)/s-np.outer(z, z)/(k*s**3)


def column_norms(mu, eps=EPS):
    mu = np.asarray(mu)
    k = mu.shape[-1]
    z = mu-mu.mean(axis=-1, keepdims=True)
    v = np.mean(z*z, axis=-1, keepdims=True)
    if k == 2:
        # The common norm is tiny away from equality; avoid subtracting
        # O(1/v) terms to obtain an O(epsilon^2/v^3) result.
        return np.broadcast_to(.5*eps**2/(v+eps)**3, mu.shape)
    return np.maximum((1-1/k)/(v+eps)-z*z*(v+2*eps)/(k*(v+eps)**3), 0)


def allocate(c, budget):
    c = np.asarray(c, dtype=float)
    k = c.shape[-1]
    if budget < k or int(budget) != budget or not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError('Nonnegative finite coefficients and integer budget >= K required')
    m = np.ones_like(c, dtype=int)
    rows = m.reshape(-1, k)
    coefficients = c.reshape(-1, k)
    for _ in range(budget-k):
        idx = np.argmax(coefficients/(rows*(rows+1)), axis=1)
        rows[np.arange(len(rows)), idx] += 1
    return m


def feasible_cells():
    return [(k, b, m0) for k in [2, 4, 8, 16] for b in [16, 32, 64]
            for m0 in [1, 2, 4] if b >= k*(m0+1)]


def allocations(pilot, p, m0, total):
    pilot = np.asarray(pilot)
    k = len(p)
    budget = total-k*m0
    a = pilot+.5
    bb = m0-pilot+.5
    mean = a/(m0+1)
    var = a*bb/((m0+1)*(m0+2))
    uniform = np.full_like(pilot, budget//k)
    uniform[..., :budget % k] += 1
    oracle = allocate(p*(1-p) if k == 2 else p*(1-p)*column_norms(p), budget)
    variance_counts = allocate(var, budget)
    return {'uniform': uniform, 'variance-only': variance_counts,
            'JADA': variance_counts.copy() if k == 2 else allocate(var*column_norms(mean), budget),
            'oracle': np.broadcast_to(oracle, pilot.shape)}


def values(pilot, main, m, m0, total, reuse):
    if reuse == 'naive':
        return (pilot+main)/(m0+m)
    lam = {'main-only': 0., 'fixed-0.5': .5,
           'budget-precommitted': m.shape[-1]*m0/total}[reuse]
    return lam*pilot/m0+(1-lam)*main/m


def binomial_states(m, p):
    states = np.array(list(itertools.product(*[range(int(n)+1) for n in m])))
    weights = np.ones(len(states))
    for j, (n, q) in enumerate(zip(m, p)):
        weights *= np.array([math.comb(int(n), int(t))*q**t*(1-q)**(n-t) for t in states[:, j]])
    return states, weights


def exact(p, total, m0):
    p = np.asarray(p, dtype=float)
    k = len(p)
    target = advantage(p)
    pilot_states, pilot_weights = binomial_states(np.full(k, m0), p)
    accum = {f'{a}/{r}': {'mse': 0., 'value_mse': 0., 'value_mean': np.zeros(k),
                         'adv_mean': np.zeros(k), 'second': np.zeros((k, k)),
                         'inverse_m': np.zeros(k), 'linear_mse': 0., 'sign_error': 0.}
             for a in ALLOCATIONS for r in REUSE}
    cache = {}
    J = jacobian(p)
    for pilot, pw in zip(pilot_states, pilot_weights):
        for a, m in allocations(pilot, p, m0, total).items():
            key = tuple(m)
            if key not in cache:
                cache[key] = binomial_states(m, p)
            main, weights = cache[key]
            w = pw*weights
            for reuse in REUSE:
                out = accum[f'{a}/{reuse}']
                val = values(pilot, main, m, m0, total, reuse)
                err = val-p
                adv = advantage(val)
                out['mse'] += w@np.mean((adv-target)**2, axis=1)
                out['value_mse'] += w@np.mean(err**2, axis=1)
                out['value_mean'] += w@val
                out['adv_mean'] += w@adv
                out['second'] += np.einsum('n,ni,nj->ij', w, err, err)
                out['inverse_m'] += pw/m
                out['linear_mse'] += w@np.mean((err@J.T)**2, axis=1)
                out['sign_error'] += w@np.mean(np.sign(adv) != np.sign(target), axis=1)
    for name, out in accum.items():
        reuse = name.split('/')[1]
        out['value_bias'] = out.pop('value_mean')-p
        out['advantage_bias'] = out.pop('adv_mean')-target
        out['value_covariance'] = out.pop('second')-np.outer(out['value_bias'], out['value_bias'])
        inv = out.pop('inverse_m')
        if reuse != 'naive':
            lam = {'main-only': 0., 'fixed-0.5': .5, 'budget-precommitted': k*m0/total}[reuse]
            out['covariance_formula'] = p*(1-p)*(lam**2/m0+(1-lam)**2*inv)
            out['surrogate_mse'] = np.sum(column_norms(p)*out['covariance_formula'])/k
        else:
            out['covariance_formula'] = None
            out['surrogate_mse'] = out['linear_mse']
        for field, val in list(out.items()):
            if isinstance(val, np.ndarray):
                out[field] = val.tolist()
    return accum


def monte_carlo(p, total, m0, n, seed):
    """Common pilot and main prefixes pair all sixteen methods trial by trial."""
    p = np.asarray(p)
    k = len(p)
    rng = np.random.default_rng(seed)
    pilot = rng.binomial(m0, p, size=(n, k))
    counts = allocations(pilot, p, m0, total)
    max_main = max(int(m.max()) for m in counts.values())
    stream = (rng.random((n, k, max_main)) < p[None, :, None]).cumsum(axis=2)
    J = jacobian(p)
    result, losses = {}, {}
    for allocation, m in counts.items():
        main = np.take_along_axis(stream, (m-1)[..., None], axis=2)[..., 0]
        for reuse in REUSE:
            key = f'{allocation}/{reuse}'
            val = values(pilot, main, m, m0, total, reuse)
            err = val-p
            adv = advantage(val)
            loss = np.mean((adv-advantage(p))**2, axis=1)
            losses[key] = loss
            out = {'mse': float(loss.mean()), 'mse_se': float(loss.std(ddof=1)/np.sqrt(n)),
                   'value_mse': float(np.mean(err**2)), 'value_bias': err.mean(0).tolist(),
                   'value_bias_se': (err.std(0, ddof=1)/np.sqrt(n)).tolist(),
                   'advantage_bias': (adv-advantage(p)).mean(0).tolist(),
                   'value_covariance': np.cov(err, rowvar=False).tolist(),
                   'linear_mse': float(np.mean((err@J.T)**2)),
                   'sign_error': float(np.mean(np.sign(adv) != np.sign(advantage(p)))),
                   'mean_allocation': m.mean(0).tolist(),
                   'oracle_allocation_L1': float(np.abs(m-counts['oracle']).sum(1).mean())}
            if reuse != 'naive':
                lam = {'main-only': 0., 'fixed-0.5': .5, 'budget-precommitted': k*m0/total}[reuse]
                diag = p*(1-p)*(lam**2/m0+(1-lam)**2*np.mean(1/m, axis=0))
                out['covariance_formula'] = diag.tolist()
                out['surrogate_mse'] = float(column_norms(p)@diag/k)
            else:
                out['covariance_formula'] = None
                out['surrogate_mse'] = out['linear_mse']
            result[key] = out
    comparisons = {}
    for reuse in REUSE:
        for baseline in ['uniform', 'variance-only', 'oracle']:
            x = losses[f'{baseline}/{reuse}']
            y = losses[f'JADA/{reuse}']
            comparisons[f'JADA-vs-{baseline}/{reuse}'] = paired_summary(x, y)
    return result, comparisons


def paired_summary(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    baseline, candidate = x.mean(), y.mean()
    d = x-y
    se = d.std(ddof=1)/np.sqrt(n)
    relative = 1-candidate/baseline if baseline > 0 else 0.
    influence = candidate/baseline**2*(x-baseline)-(y-candidate)/baseline if baseline > 0 else np.zeros(n)
    rse = influence.std(ddof=1)/np.sqrt(n)
    return {'difference': float(d.mean()), 'difference_se': float(se),
            'difference_ci95': [float(d.mean()-1.96*se), float(d.mean()+1.96*se)],
            'relative_reduction': float(relative), 'relative_se': float(rse),
            'relative_ci95': [float(relative-1.96*rse), float(relative+1.96*rse)]}
