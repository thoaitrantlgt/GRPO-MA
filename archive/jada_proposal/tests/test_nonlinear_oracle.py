import math
import unittest
import numpy as np

try:
    from experiments.nonlinear_oracle import oracle as o
except ImportError:
    o = None


class NonlinearOracleTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(o, 'Exact nonlinear oracle implementation is missing')

    def test_all_allocations_enumerated_without_duplicates(self):
        for main in [4, 8, 12]:
            m = o.compositions(main, 4)
            self.assertEqual(len(m), math.comb(main-1, 3))
            self.assertEqual(len(set(map(tuple, m))), len(m))
            self.assertTrue(np.all(m >= 1))
            np.testing.assert_array_equal(m.sum(1), main)
        with self.assertRaises(ValueError):
            o.compositions(3, 4)

    def test_exact_uniform_anchor_and_oracle_dominance(self):
        p = np.array([[.9, .52, .48, .1], [.5, .5, .5, .5]])
        output = o.solve(p, 16, 2)
        self.assertAlmostEqual(output['mse']['uniform'][0], .280223984, places=8)
        self.assertAlmostEqual(output['mse']['JADA'][0], .288793640, places=8)
        adaptive = output['mse']['adaptive-nonlinear-oracle']
        for method, risk in output['mse'].items():
            self.assertTrue(np.all(adaptive <= risk+1e-13), method)
        self.assertTrue(np.all(output['mse']['fixed-nonlinear-oracle'] <= output['mse']['uniform']+1e-13))
        np.testing.assert_allclose(output['oracle_value_bias'], 0, atol=2e-14)

    def test_vectorized_conditional_risk_matches_direct_enumeration(self):
        from experiments.cpu_gate import estimator as e
        p = np.array([[.9, .8, .7, .1], [.5, .5, .5, .5]])
        result = o.solve(p, 16, 1)
        for v in [0, 1]:
            for pilot_index in [0, 3, 15]:
                pilot = result['pilots'][pilot_index]
                for allocation_index in [0, 7, 164]:
                    m = result['allocations'][allocation_index]
                    main, prob = e.binomial_states(m, p[v])
                    val = e.values(pilot, main, m, 1, 16, 'budget-precommitted')
                    direct = prob@np.mean((e.advantage(val)-e.advantage(p[v]))**2, axis=1)
                    self.assertAlmostEqual(result['conditional_risks'][v, pilot_index, allocation_index], direct, places=12)

    def test_deterministic_rewards_zero_error(self):
        result = o.solve(np.array([[0., 1., 0., 1.], [0., 0., 0., 0.]]), 16, 1)
        for risk in result['mse'].values():
            np.testing.assert_allclose(risk, 0, atol=1e-14)

    def test_saved_table_audit_rejects_nonoptimal_oracle_policy(self):
        try:
            from experiments.nonlinear_oracle.audit import verify_table
        except ImportError:
            self.fail('Independent saved-table audit is missing')
        p = np.array([[.9, .8, .7, .1]])
        result = o.solve(p, 16, 1)
        table = {key: result[key] for key in ['allocations', 'pilots', 'pilot_probabilities',
                                             'conditional_risks', 'unconditional_risks']}
        table['p'] = p
        table.update({f'selected_{method}': value.copy() for method, value in result['selected'].items()})
        verify_table(table, p, 16, 1)
        pilot = int(np.argmax(np.ptp(table['conditional_risks'][0], axis=1)))
        worst = int(np.argmax(table['conditional_risks'][0, pilot]))
        table['selected_adaptive-nonlinear-oracle'][0, pilot] = worst
        with self.assertRaises(ValueError):
            verify_table(table, p, 16, 1)


if __name__ == '__main__':
    unittest.main()
