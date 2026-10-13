import itertools
import unittest
import numpy as np
from scipy.stats import binom
from experiments.sequential_sampling.core import target, jacobian
from experiments.gradient_headroom.core import allocations, geometries, moments, update_risks


class GradientHeadroomTests(unittest.TestCase):
    def test_exact_moments_match_direct_squared_updates(self):
        p=np.array([[.13,.32,.61,.89],[0,0,1,1],[.5,.5,.5,.5]])
        n=np.array([1,2,1,2]);grams=geometries()[1]
        matrix,mean= moments(n,p)
        direct=np.zeros((len(p),len(grams)))
        for s in itertools.product(*(range(int(v)+1) for v in n)):
            s=np.array(s);e=target(s/n)-target(p)
            w=binom.pmf(s,n,p).prod(axis=1)
            direct+=w[:,None]*np.einsum('vi,gij,vj->vg',e,grams,e)
        self.assertTrue(np.allclose(update_risks(matrix,grams),direct,atol=1e-12))
        covariance=matrix-np.einsum('vi,vj->vij',mean,mean)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(),-1e-12)

    def test_global_oracle_search_contains_uniform_and_minimum_coverage(self):
        ns=allocations(4,16,1)
        self.assertEqual(len(ns),455)
        self.assertEqual(sum(np.min(ns,axis=1)>=2),165)
        self.assertTrue(np.all(ns.sum(axis=1)==16))
        self.assertTrue(any(np.array_equal(n,[4]*4) for n in ns))

    def test_null_and_orthogonal_geometries(self):
        names,grams=geometries();p=np.array([[.1,.4,.6,.8]])
        matrix,_=moments([2,2,2,2],p);risks=update_risks(matrix,grams)
        self.assertAlmostEqual(risks[0,names.index('aligned-null')],0,places=12)
        self.assertAlmostEqual(risks[0,0],np.trace(matrix[0]),places=12)
        self.assertTrue(np.allclose(np.trace(grams,axis1=1,axis2=2),4))

    def test_gradient_jacobian_local_coefficients_in_both_regions(self):
        _,grams=geometries()
        for p,step in [(np.array([.1,.35,.65,.9]),1e-7),(np.array([.5,.500001,.499999,.5]),1e-9)]:
            numerical=np.column_stack([(target(p+step*np.eye(4)[i])-target(p-step*np.eye(4)[i]))/(2*step) for i in range(4)])
            j=jacobian(p)
            self.assertTrue(np.allclose(j,numerical,rtol=1e-6,atol=1e-4))
            coefficients=p*(1-p)*np.einsum('ij,gik,kj->gj',j,grams,j)
            self.assertGreater(coefficients.min(),-1e-8)

if __name__=='__main__':unittest.main()
