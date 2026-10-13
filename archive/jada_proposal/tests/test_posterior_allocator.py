import itertools
import unittest
import numpy as np

try:
    from experiments.posterior_allocator import policy as b
except ImportError:
    b = None


class PosteriorAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(b, 'Pilot-only posterior allocator is missing')

    def test_posterior_risk_targets_true_advantage_integrated_over_p(self):
        from experiments.nonlinear_oracle.oracle import conditional_risk_grid
        p = np.array([[.2, .4, .7, .8], [.8, .7, .3, .1]])
        prior = np.array([.25, .75])
        grid = conditional_risk_grid(p, 16, 1)
        numerator, mass = b.joint_integrals(grid, prior)
        pi, ci = 5, 8
        likelihood = grid['pilot_probabilities'][:, pi]
        expected = np.sum(prior*likelihood*grid['conditional_risks'][:, pi, ci])/np.sum(prior*likelihood)
        self.assertAlmostEqual(numerator[pi, ci]/mass[pi], expected, places=13)

    def test_bayes_policy_dominates_uniform_under_finite_prior(self):
        from experiments.nonlinear_oracle.oracle import conditional_risk_grid
        p = np.array([[.2, .4, .7, .8], [.8, .7, .3, .1]])
        grid = conditional_risk_grid(p, 16, 1)
        prior = np.array([.25, .75])
        numerator, mass = b.joint_integrals(grid, prior)
        policy = b.make_policy(grid['pilots'], grid['allocations'], numerator, mass, 16, 1, 'finite-test', symmetrize=False)
        chosen = b.evaluate(grid, policy)
        uniform_index = next(i for i, m in enumerate(grid['allocations']) if np.all(m == 3))
        uniform = np.sum(grid['pilot_probabilities']*grid['conditional_risks'][:, :, uniform_index], axis=1)
        self.assertLessEqual(prior@chosen, prior@uniform+1e-12)

    def test_policy_is_pilot_only_feasible_and_serializable(self):
        from experiments.nonlinear_oracle.oracle import conditional_risk_grid
        grid = conditional_risk_grid(np.array([[.3, .4, .6, .8]]), 16, 1)
        num, mass = b.joint_integrals(grid)
        policy = b.make_policy(grid['pilots'], grid['allocations'], num, mass, 16, 1, 'finite-test', symmetrize=False)
        restored = b.PilotPolicy.from_dict(policy.to_dict())
        np.testing.assert_array_equal(policy.probabilities, restored.probabilities)
        rng = np.random.default_rng(72)
        for pilot in itertools.product(range(2), repeat=4):
            counts = restored.allocate(np.array(pilot), rng)
            self.assertEqual(counts.sum(), 12)
            self.assertTrue(np.all(counts >= 1))
        with self.assertRaises(TypeError):
            restored.allocate(np.array([0, 1, 1, 0]), rng, true_p=np.full(4, .5))
        with self.assertRaises(ValueError):
            restored.allocate(np.array([0, 2, 0, 1]), rng)

    def test_exchangeable_symmetry_preserved_by_randomized_ties(self):
        from experiments.nonlinear_oracle.oracle import conditional_risk_grid
        grid = conditional_risk_grid(np.array([[.2, .3, .7, .9], [.8, .7, .3, .1]]), 16, 1)
        num, mass = b.joint_integrals(grid)
        policy = b.make_policy(grid['pilots'], grid['allocations'], num, mass, 16, 1, 'symmetric-test')
        for permutation in itertools.permutations(range(4)):
            hmap = {tuple(h): i for i, h in enumerate(policy.pilots)}
            cmap = {tuple(m): i for i, m in enumerate(policy.allocations)}
            hi = [hmap[tuple(h[list(permutation)])] for h in policy.pilots]
            ci = [cmap[tuple(m[list(permutation)])] for m in policy.allocations]
            np.testing.assert_allclose(policy.probabilities, policy.probabilities[np.ix_(hi, ci)], atol=1e-14)

    def test_prior_quadrature_draws_match_declared_components(self):
        try:
            from experiments.posterior_allocator.fit import prior_points
        except ImportError:
            self.fail('Predeclared prior integration is missing')
        config = {'seed': 4242422026, 'Jeffreys_points_per_replicate': 1024,
                  'mixture_points_per_component_per_replicate': 256}
        p, components = prior_points('posterior-Jeffreys', 0, config)
        self.assertLess(abs(p.mean()-.5), .001)
        self.assertLess(abs(p.var()-.125), .001)
        p, components = prior_points('posterior-mixture', 0, config)
        self.assertEqual(p.shape, (1536, 4))
        for c in range(6):
            self.assertEqual(np.sum(components == c), 256)
        self.assertTrue(np.all((p > 0) & (p < 1)))


if __name__ == '__main__':
    unittest.main()
