import itertools
import unittest
import numpy as np
from experiments.posterior_credit.core import target,table_from_support,exact_metrics


class PosteriorCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        permutations=list(itertools.permutations(range(4)))
        cls.points=np.array([[.15,.35,.65,.85][i] for i in range(4)])
        cls.support=np.array([cls.points[list(p)] for p in permutations])
        cls.table=table_from_support(cls.support)

    def test_posterior_matches_direct_joint_sum(self):
        from experiments.nonlinear_oracle.oracle import state_probabilities
        likelihood=state_probabilities(self.table['states'],np.full(4,4),self.support)
        direct=likelihood.T@target(self.support)/likelihood.sum(0)[:,None]
        np.testing.assert_allclose(self.table['credit'],direct,atol=1e-14)
        self.assertAlmostEqual(self.table['mass'].sum(),1,places=12)

    def test_finite_prior_bayes_dominance(self):
        t={**self.table,'scalar':.7,'plugin':target(self.table['states']/4)}
        g=np.array([np.eye(4)/4,np.outer([1,-1,1,-1],[1,-1,1,-1])/4])
        result=exact_metrics(self.support,t,g)
        for name,metric in result.items():
            np.testing.assert_allclose(metric['mse'],metric['bias_squared']+metric['variance'],atol=1e-12)
            self.assertTrue(np.all(result['posterior-credit']['mse'].mean(0)<=metric['mse'].mean(0)+1e-12))

    def test_exchangeability_centering_and_equal_counts(self):
        lookup={tuple(s):i for i,s in enumerate(self.table['states'])}
        credit=self.table['credit']
        np.testing.assert_allclose(credit.sum(1),0,atol=1e-12)
        perm=[2,0,3,1]
        for i,state in enumerate(self.table['states']):
            np.testing.assert_allclose(credit[lookup[tuple(state[perm])]],credit[i,perm],atol=1e-12)
        for n in range(5):
            np.testing.assert_allclose(credit[lookup[(n,)*4]],0,atol=1e-12)


if __name__=='__main__':
    unittest.main()
