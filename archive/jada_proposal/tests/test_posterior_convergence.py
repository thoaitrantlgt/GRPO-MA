import unittest
import numpy as np
from experiments.nonlinear_oracle.oracle import conditional_risk_grid
from experiments.posterior_allocator.convergence import nested_sums


class ConvergenceTests(unittest.TestCase):
    def test_chunked_component_prefixes_equal_direct_finite_prior_integral(self):
        p=np.random.default_rng(7).uniform(.05,.95,(24,4))
        positions=np.tile(np.arange(4),6)
        grid=conditional_risk_grid(p,12,1,.5)
        for limit in [2,4]:
            totals=[]
            for begin in range(0,24,5):
                end=min(begin+5,24)
                chunk={key: value[begin:end] if key in ['p','pilot_probabilities','conditional_risks'] else value for key,value in grid.items()}
                totals.append(nested_sums(chunk,positions[begin:end],[limit])[0])
            num=sum(x[0] for x in totals); mass=sum(x[1] for x in totals); count=sum(x[2] for x in totals)
            selected=positions<limit
            self.assertEqual(count,6*limit)
            expected=(grid['pilot_probabilities'][selected,:,None]*grid['conditional_risks'][selected]).mean(0)
            np.testing.assert_allclose(num/count,expected,atol=1e-14)
            np.testing.assert_allclose(mass/count,grid['pilot_probabilities'][selected].mean(0),atol=1e-14)
            self.assertAlmostEqual(float((mass/count).sum()),1,places=13)


if __name__=='__main__':
    unittest.main()
