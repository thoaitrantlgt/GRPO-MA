import unittest
import numpy as np
from experiments.surrogate_diagnostic.core import diagnose
from experiments.cpu_gate.estimator import exact


class SurrogateDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.result = diagnose([[.2,.4,.7,.9],[.5,.5,.5,.5]],12,1)

    def test_moment_and_taylor_identities(self):
        for row in self.rows:
            self.assertAlmostEqual(row['mse'],row['bias_squared']+row['variance'],places=12)
            self.assertAlmostEqual(row['mse'],row['linear']+row['cross']+row['remainder'],places=9)
            self.assertAlmostEqual(row['linear'],row['surrogate'],places=9)
            self.assertLess(row['raw_bias_max'],1e-12)
            self.assertLess(row['linear_mean_max'],1e-10)
            self.assertGreaterEqual(row['variance'],-1e-12)

    def test_independent_existing_exact_engine(self):
        reference = exact(np.array([.2,.4,.7,.9]),12,1)
        for name in ['uniform','JADA','local-surrogate-oracle']:
            row = next(r for r in self.rows if r['index']==0 and r['method']==name)
            old = 'oracle' if name=='local-surrogate-oracle' else name
            self.assertAlmostEqual(row['mse'],reference[old+'/budget-precommitted']['mse'],places=12)

    def test_equal_probability_boundary(self):
        row = next(r for r in self.rows if r['index']==1 and r['method']=='uniform')
        self.assertLess(row['bias_squared'],1e-24)
        self.assertLessEqual(row['mse'],1)
        self.assertGreater(row['surrogate'],1000)


if __name__=='__main__':
    unittest.main()
