import unittest
import numpy as np
from experiments.gradient_gate import core as g
from experiments.nonlinear_oracle.oracle import conditional_risk_grid
from experiments.cpu_gate import estimator as e


class GradientGateTests(unittest.TestCase):
    def test_gram_psd_scalar_and_null_boundaries(self):
        matrices=g.grams()
        self.assertTrue(np.all(np.linalg.eigvalsh(matrices)>=-1e-13))
        np.testing.assert_allclose(np.trace(matrices,axis1=1,axis2=2),1)
        p=np.array([[.2,.4,.7,.9]])
        grid=g.risk_grid(p,16,1)
        scalar=conditional_risk_grid(p,16,1)
        np.testing.assert_allclose(grid['conditional_risks'][:,0],scalar['conditional_risks'],atol=1e-12)
        np.testing.assert_array_equal(grid['conditional_risks'][:,1],0)
        np.testing.assert_allclose(grid['conditional_risks'][:,2],.5*grid['conditional_risks'][:,0],atol=1e-12)

    def test_weighted_nonlinear_risk_matches_direct_main_enumeration(self):
        p=np.array([[.1,.3,.6,.9],[.8,.5,.2,.1]])
        grid=g.risk_grid(p,16,2)
        for gram,h,c in [(3,7,3),(4,15,19),(5,80,34),(6,2,1)]:
            m=grid['allocations'][c]; t,w=e.binomial_states(m,p[0])
            value=.5*grid['pilots'][h]/2+.5*t/m
            delta=e.advantage(value)-e.advantage(p[0])
            direct=w@np.einsum('ti,ij,tj->t',delta,grid['matrices'][gram],delta)
            self.assertAlmostEqual(grid['conditional_risks'][0,gram,h,c],direct,places=12)
        result=g.solve(grid,16,2)
        for value in result['mse'].values():
            self.assertTrue(np.all(result['mse']['adaptive-gradient-oracle']<=value+1e-12))
        np.testing.assert_array_equal(result['mse']['gradient-JADA'][:,0],result['mse']['JADA'][:,0])
        np.testing.assert_array_equal(result['selected']['gradient-JADA'][0],result['selected']['gradient-JADA'][1])

    def test_score_covariance_identity_and_local_coefficients(self):
        gram=g.grams()[-1]; vals,vecs=np.linalg.eigh(gram)
        scores=np.sqrt(np.maximum(vals,0))[:,None]*vecs.T
        np.testing.assert_allclose(scores.T@scores,gram,atol=1e-13)
        p=np.array([.15,.4,.65,.9]); t,w=e.binomial_states(np.full(4,3),p)
        delta=e.advantage(t/3)-e.advantage(p)
        mean=w@delta; cov=np.einsum('t,ti,tj->ij',w,delta-mean,delta-mean)
        direct=w@np.sum((delta@scores.T)**2,axis=1)
        self.assertAlmostEqual(direct,np.trace(gram@cov)+mean@gram@mean,places=12)
        coef=g.local_coefficients(p,p*(1-p),g.grams())[:,0]
        j=e.jacobian(p)
        np.testing.assert_allclose(coef[-1],p*(1-p)*np.diag(j.T@gram@j),atol=1e-13)
        np.testing.assert_allclose(coef[0],p*(1-p)*e.column_norms(p)/4,atol=1e-13)


if __name__=='__main__':
    unittest.main()
