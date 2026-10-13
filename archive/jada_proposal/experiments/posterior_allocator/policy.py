"""Pilot-only policies minimizing integrated nonlinear normalized-credit risk."""
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass(frozen=True)
class PilotPolicy:
    total: int
    m0: int
    prior: str
    pilots: np.ndarray
    allocations: np.ndarray
    probabilities: np.ndarray
    posterior_risk: np.ndarray

    def __post_init__(self):
        m = self.allocations
        if m.ndim != 2 or m.shape[1] != 4 or np.any(m < 1) or not np.all(m == np.round(m)) \
                or not np.all(m.sum(1) == self.total-4*self.m0):
            raise ValueError('Policy violates positive integer main-budget counts')
        if set(map(tuple, self.pilots)) != set(itertools.product(range(self.m0+1), repeat=4)):
            raise ValueError('Policy does not cover all pilot histories')
        q = self.probabilities
        if q.shape != (len(self.pilots), len(m)) or not np.all(np.isfinite(q)) or np.any(q < 0) \
                or not np.allclose(q.sum(1), 1, atol=1e-13, rtol=0):
            raise ValueError('Invalid policy randomization probabilities')

    def allocate(self, pilot_counts, rng):
        h = np.asarray(pilot_counts)
        if h.shape != (4,) or np.any(~np.isfinite(h)) or np.any(h < 0) or np.any(h > self.m0) \
                or not np.all(h == np.round(h)):
            raise ValueError('Invalid pilot counts')
        index = np.flatnonzero(np.all(self.pilots == h, axis=1))[0]
        chosen = rng.choice(len(self.allocations), p=self.probabilities[index])
        return self.allocations[chosen].copy()

    def to_dict(self):
        return {'total': self.total, 'm0': self.m0, 'prior': self.prior,
                'pilots': self.pilots.tolist(), 'allocations': self.allocations.tolist(),
                'probabilities': self.probabilities.tolist(), 'posterior_risk': self.posterior_risk.tolist(),
                'inputs': 'pilot counts and independent tie RNG only; no true probabilities'}

    @classmethod
    def from_dict(cls, data):
        return cls(data['total'], data['m0'], data['prior'], np.array(data['pilots'], dtype=int),
                   np.array(data['allocations'], dtype=int), np.array(data['probabilities']), np.array(data['posterior_risk']))


def joint_integrals(grid, weights=None):
    n = len(grid['p'])
    weights = np.full(n, 1/n) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n,) or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('A valid finite prior distribution is required')
    weights = weights/weights.sum()
    likelihood = grid['pilot_probabilities']
    numerator = np.einsum('v,vp,vpc->pc', weights, likelihood, grid['conditional_risks'])
    mass = weights@likelihood
    return numerator, mass


def symmetrize_integrals(pilots, allocations, numerator, mass):
    hmap = {tuple(h): i for i, h in enumerate(pilots)}
    cmap = {tuple(m): i for i, m in enumerate(allocations)}
    averaged = np.zeros_like(numerator)
    averaged_mass = np.zeros_like(mass)
    for perm in itertools.permutations(range(4)):
        hi = [hmap[tuple(h[list(perm)])] for h in pilots]
        ci = [cmap[tuple(m[list(perm)])] for m in allocations]
        averaged += numerator[np.ix_(hi, ci)]/24
        averaged_mass += mass[hi]/24
    return averaged, averaged_mass


def make_policy(pilots, allocations, numerator, mass, total, m0, prior, symmetrize=True):
    if symmetrize:
        numerator, mass = symmetrize_integrals(pilots, allocations, numerator, mass)
    if np.any(mass <= 0) or not np.all(np.isfinite(numerator)):
        raise ValueError('Posterior is undefined for a pilot history')
    risk = numerator/mass[:, None]
    supported = risk <= risk.min(1, keepdims=True)+1e-12
    probabilities = supported/supported.sum(1, keepdims=True)
    return PilotPolicy(total, m0, prior, pilots, allocations, probabilities, risk)


def evaluate(grid, policy):
    if not np.array_equal(grid['pilots'], policy.pilots) or not np.array_equal(grid['allocations'], policy.allocations):
        raise ValueError('Policy and exact risk grid are incompatible')
    conditional = np.sum(grid['conditional_risks']*policy.probabilities[None, :, :], axis=2)
    return np.sum(grid['pilot_probabilities']*conditional, axis=1)
