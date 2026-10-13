"""Offline prior integration. No evaluation examples are accepted by this API."""
from datetime import datetime, timezone
import time
import numpy as np
from scipy.special import betaincinv, expit, ndtri
from scipy.stats import qmc
from experiments.cpu_gate.run import dump
from experiments.nonlinear_oracle.oracle import conditional_risk_grid
from .policy import make_policy, joint_integrals, symmetrize_integrals

PRIORS = ('posterior-Jeffreys', 'posterior-mixture')


def prior_points(prior, replicate, config):
    blocks, labels = [], []
    parameters = [(.5, .5)] if prior == 'posterior-Jeffreys' else [(.5, .5), (2, 2), (8, 8), (.5, 3), (3, .5), None]
    if prior not in PRIORS:
        raise ValueError('Unknown predeclared prior')
    n = config['Jeffreys_points_per_replicate'] if prior == PRIORS[0] else config['mixture_points_per_component_per_replicate']
    power = int(np.log2(n))
    if 2**power != n:
        raise ValueError('Sobol block size must be a power of two')
    for component, ab in enumerate(parameters):
        rng = np.random.default_rng([config['seed'], PRIORS.index(prior), replicate, component])
        u = qmc.Sobol(d=6, scramble=True, seed=rng).random_base2(power)
        u = np.clip(u, 1e-12, 1-1e-12)
        if ab is None:
            location = ndtri(u[:, 4])
            scale = np.exp(np.log(.15)+u[:, 5]*(np.log(3)-np.log(.15)))
            p = expit(location[:, None]+scale[:, None]*ndtri(u[:, :4]))
        else:
            p = betaincinv(*ab, u[:, :4])
        blocks.append(p)
        labels.append(np.full(n, component, dtype=int))
    return np.concatenate(blocks), np.concatenate(labels)


def fit_prior(prior, protocol, out):
    config = protocol['offline_integration']
    tables = {m0: {'numerator': [], 'mass': []} for m0 in protocol['m0']}
    samples, sample_components = [], []
    start = time.perf_counter()
    for rep in range(config['replicates']):
        points, components = prior_points(prior, rep, config)
        samples.append(points)
        sample_components.append(components)
        for m0 in protocol['m0']:
            grid = conditional_risk_grid(points, 16, m0)
            numerator, mass = joint_integrals(grid)
            num, mass = symmetrize_integrals(grid['pilots'], grid['allocations'], numerator, mass)
            tables[m0]['numerator'].append(num)
            tables[m0]['mass'].append(mass)
            tables[m0]['pilots'], tables[m0]['allocations'] = grid['pilots'], grid['allocations']
        print(f'{prior}: scramble {rep+1}/{config["replicates"]}, {len(points)} prior points; {time.perf_counter()-start:.1f}s', flush=True)
    np.savez_compressed(out/f'prior_points_{prior}.npz', points=np.array(samples), components=np.array(sample_components))
    policies = {}
    for m0, table in tables.items():
        nums, masses = np.array(table['numerator']), np.array(table['mass'])
        mean_num, mean_mass = nums.mean(0), masses.mean(0)
        policy = make_policy(table['pilots'], table['allocations'], mean_num, mean_mass, 16, m0, prior)
        policies[m0] = policy
        half = len(nums)//2
        first = make_policy(table['pilots'], table['allocations'], nums[:half].mean(0), masses[:half].mean(0), 16, m0, prior)
        last = make_policy(table['pilots'], table['allocations'], nums[half:].mean(0), masses[half:].mean(0), 16, m0, prior)
        disagreement = .5*np.abs(first.probabilities-last.probabilities).sum(1)
        rep_risk = nums/masses[:, :, None]
        chosen_risk = np.sum(rep_risk*policy.probabilities[None, :, :], axis=2)
        uni = next(i for i, m in enumerate(policy.allocations) if np.all(m == (16-4*m0)//4))
        difference = rep_risk[:, :, uni]-chosen_risk
        diagnostic = {'prior': prior, 'm0': m0, 'points': int(sum(len(x) for x in samples)),
                      'scrambles': len(nums), 'fit_completed_utc': datetime.now(timezone.utc).isoformat(),
                      'pilot_mass_weighted_half_policy_TV': float(mean_mass@disagreement),
                      'pilot_histories_with_half_policy_difference': int(np.sum(disagreement > 1e-12)),
                      'max_selected_posterior_risk_SE_across_scrambles': float(np.max(chosen_risk.std(0, ddof=1)/np.sqrt(len(nums)))),
                      'max_selected_minus_uniform_paired_SE': float(np.max(difference.std(0, ddof=1)/np.sqrt(len(nums)))),
                      'estimated_prior_expected_uniform_mse': float(mean_mass@policy.posterior_risk[:, uni]),
                      'estimated_prior_expected_policy_mse': float(mean_mass@np.sum(policy.posterior_risk*policy.probabilities, axis=1)),
                      'information_access': 'offline declared prior and current pilot counts only; no evaluation vector used'}
        dump(out/f'policy_{prior}_m0_{m0}.json', {**policy.to_dict(), 'construction': diagnostic})
        np.savez_compressed(out/f'integration_{prior}_m0_{m0}.npz', numerators=nums, masses=masses,
                            pilots=table['pilots'], allocations=table['allocations'],
                            first_half_policy=first.probabilities, last_half_policy=last.probabilities)
        dump(out/f'integration_diagnostic_{prior}_m0_{m0}.json', diagnostic)
    return policies
