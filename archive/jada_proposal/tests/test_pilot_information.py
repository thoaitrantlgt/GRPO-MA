import unittest
import numpy as np
from experiments.cpu_gate import estimator as e
from experiments.nonlinear_oracle.oracle import conditional_risk_grid


class PilotInformationTests(unittest.TestCase):
    def test_fixed_lambda_conditional_risk_direct_and_old_default(self):
        p = np.array([[.2,.4,.7,.9]])
        g = conditional_risk_grid(p,12,1,.5)
        for h in [0,3,15]:
            for j in [0,9,34]:
                m = g['allocations'][j]
                t,w = e.binomial_states(m,p[0])
                v = .5*g['pilots'][h]+.5*t/m
                direct = w@np.mean((e.advantage(v)-e.advantage(p[0]))**2,axis=1)
                self.assertAlmostEqual(g['conditional_risks'][0,h,j],direct,places=12)
        a = conditional_risk_grid(p,16,2)
        b = conditional_risk_grid(p,16,2,.5)
        np.testing.assert_array_equal(a['conditional_risks'],b['conditional_risks'])
        with self.assertRaises(ValueError):
            conditional_risk_grid(p,12,1,1.1)

    def test_fixed_lambda_covariance_with_pilot_measurable_counts(self):
        p = np.array([.2,.4,.7,.9])
        result = e.exact(p,12,1)
        for method in ['uniform','variance-only','JADA','oracle']:
            row = result[f'{method}/fixed-0.5']
            np.testing.assert_allclose(row['value_bias'],0,atol=2e-14)
            np.testing.assert_allclose(row['value_covariance'],np.diag(row['covariance_formula']),atol=2e-14)


if __name__ == '__main__':
    unittest.main()
