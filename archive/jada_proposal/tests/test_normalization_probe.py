import unittest
import numpy as np
from experiments.normalization_probe.core import moments
from experiments.cpu_gate.estimator import advantage


class NormalizationProbeTests(unittest.TestCase):
    def test_direct_risk_and_decomposition(self):
        p=np.array([[.2,.4,.7,.9],[.5,.5,.5,.5],[0,0,1,1]])
        result=moments(p,8)
        a=advantage(result['states']/2)
        for name,metric in result['methods'].items():
            for i,q in enumerate(p):
                risk=result['probabilities'][i]@np.mean((metric['coefficient'][i]*a-advantage(q))**2,axis=1)
                self.assertAlmostEqual(metric['mse'][i],risk,places=12)
                self.assertAlmostEqual(metric['mse'][i],metric['bias_squared'][i]+metric['variance'][i],places=12)
            self.assertTrue(np.all(result['methods']['oracle-scalar']['mse']<=metric['mse']+1e-12))

    def test_equality_and_noiseless_boundaries(self):
        result=moments([[.5]*4,[0,0,1,1]],16)['methods']
        self.assertAlmostEqual(result['fixed-half']['mse'][0],result['plugin']['mse'][0]/4,places=12)
        self.assertLess(result['plugin']['mse'][1],1e-12)
        self.assertGreater(result['fixed-half']['mse'][1],.24)

    def test_half_improvement_condition(self):
        result=moments([[.4,.5,.5,.6],[.01,.2,.8,.99]],16)
        gain=result['methods']['plugin']['mse']-result['methods']['fixed-half']['mse']
        np.testing.assert_allclose(gain,.75*result['energy']-result['alignment'],atol=1e-12)


if __name__=='__main__':
    unittest.main()
