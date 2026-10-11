import itertools
import unittest
import numpy as np

try:
    from experiments.cpu_gate import estimator as e
except ImportError:
    e = None


class EstimatorGateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(e, 'CPU estimator implementation is missing')

    def test_jacobian_and_column_norms(self):
        for p in [np.array([.83, .5, .5, .17]), np.full(4, .5),
                  np.array([.50001, .49999]), np.array([0., 1., .2, .7])]:
            h = 1e-7
            fd = np.column_stack([(e.advantage(p+h*d)-e.advantage(p-h*d))/(2*h)
                                  for d in np.eye(len(p))])
            np.testing.assert_allclose(e.jacobian(p), fd, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(e.column_norms(p), (e.jacobian(p)**2).sum(0), rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(e.jacobian(p) @ np.ones(len(p)), 0, atol=1e-10)

    def test_allocator_is_integer_feasible_and_globally_optimal(self):
        rng = np.random.default_rng(51)
        for k in [2, 3, 4]:
            for budget in range(k, k+7):
                for c in [np.zeros(k), np.ones(k), *rng.uniform(0, 3, (6, k))]:
                    m = e.allocate(c, budget)
                    self.assertEqual(m.sum(), budget)
                    self.assertTrue(np.all(m >= 1))
                    optimum = min(sum(c/np.array(n)) for n in itertools.product(range(1, budget+1), repeat=k)
                                  if sum(n) == budget)
                    self.assertAlmostEqual(sum(c/m), optimum, places=12)
                    np.testing.assert_array_equal(e.allocate(np.stack([c, c]), budget), np.stack([m, m]))
        with self.assertRaises(ValueError):
            e.allocate(np.ones(4), 3)
        with self.assertRaises(ValueError):
            e.allocate(np.array([1., -1.]), 5)

    def test_exact_fixed_reuse_covariance_and_default_regression(self):
        p = np.array([.9, .52, .48, .1])
        result = e.exact(p, 16, 2)
        for allocation in e.ALLOCATIONS:
            for reuse in ['main-only', 'fixed-0.5', 'budget-precommitted']:
                r = result[f'{allocation}/{reuse}']
                np.testing.assert_allclose(r['value_bias'], 0, atol=2e-13)
                np.testing.assert_allclose(r['value_covariance'], np.diag(r['covariance_formula']), atol=2e-13)
        self.assertAlmostEqual(result['uniform/fixed-0.5']['mse'], .280223984, places=8)
        self.assertAlmostEqual(result['JADA/fixed-0.5']['mse'], .288822868, places=8)
        self.assertGreater(abs(result['JADA/naive']['value_bias'][0]), .01)

    def test_budget_weight_matches_uniform_pooling(self):
        pilot = np.array([0, 1, 2, 1])
        m = np.full(4, 6)
        main = np.array([3, 4, 1, 0])
        np.testing.assert_allclose(e.values(pilot, main, m, 2, 32, 'budget-precommitted'), (pilot+main)/8)
        self.assertEqual(len(e.feasible_cells()), 26)


if __name__ == '__main__':
    unittest.main()
