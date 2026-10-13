import itertools
import unittest
import numpy as np
from experiments.posterior_credit.core import table_from_support
from experiments.posterior_credit.convergence import symmetrize,nested_integrals


class CreditConvergenceTests(unittest.TestCase):
    def test_symmetrized_integrals_equal_augmented_support(self):
        points=np.array([[.1,.3,.6,.8],[.2,.5,.7,.9],[.4,.6,.8,.95]])
        permutations=np.array(list(itertools.permutations(range(4))))
        augmented=points[:,permutations].reshape(-1,4)
        expected=table_from_support(augmented); actual=symmetrize(table_from_support(points))
        for key in ['mass','numerator','second','credit']:
            np.testing.assert_allclose(actual[key],expected[key],atol=1e-13)

    def test_nested_joint_sums_match_direct_prefixes(self):
        points=np.random.default_rng(7).uniform(.05,.95,size=(9,4))
        nested=nested_integrals(points,[2,5,9])
        for level in [2,5,9]:
            expected=symmetrize(table_from_support(points[:level]))
            for key in ['mass','numerator','second','credit']:
                np.testing.assert_allclose(nested[level][key],expected[key],atol=1e-13)

    def test_reject_invalid_prefix_order(self):
        with self.assertRaises(ValueError):
            nested_integrals(np.full((4,4),.5),[4,2])


if __name__=='__main__':
    unittest.main()
