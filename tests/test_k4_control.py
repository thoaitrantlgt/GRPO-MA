import itertools
import unittest
import numpy as np
from experiments.sequential_sampling.information_control import BayesControl
from experiments.sequential_sampling.symmetry_control import SymmetricControl, ladder_support, build_graph, graph_metrics
from experiments.sequential_sampling.evaluation import exact_sequential
from experiments.sequential_sampling.k4_control import complexity


class K4InformationControlTests(unittest.TestCase):
    def test_symmetry_reduction_matches_full_Bellman_solver(self):
        permutations=np.array(list(itertools.permutations(range(4))))
        points=np.array([[.15,.35,.65,.9],[.2,.45,.55,.8]])[:,permutations].reshape(-1,4)
        full=BayesControl(points,6);reduced=SymmetricControl(points,6)
        for successes in itertools.product([0,1],repeat=4):
            n=(1,)*4
            self.assertAlmostEqual(full.value(n,successes),reduced.value(n,successes),places=12)
        self.assertLess(len(reduced.values),full.value.cache_info().currsize)

    def test_rejects_nonexchangeable_support(self):
        with self.assertRaises(ValueError):SymmetricControl(np.array([[.2,.8,.3,.7]]),6)

    def test_ladder_lowest_support_matches_previous_frozen_points(self):
        with np.load('refine-logs/sequential-feasibility/20261011_run02/prior_support_K4.npz') as data:
            old=data['points']
        np.testing.assert_array_equal(ladder_support(16),old)
        small=ladder_support(16);large=ladder_support(64)
        np.testing.assert_array_equal(small[:384],large[:384])
        np.testing.assert_array_equal(small[384:],large[1536:1920])

    def test_graph_risk_matches_existing_labelled_path_evaluation(self):
        points=ladder_support(2)
        model=SymmetricControl(points,6)
        p=np.array([[.15,.35,.65,.9],[.5,.5,.5,.5]])
        graph=build_graph(model,1)
        result=graph_metrics(model,graph,p)
        expected=exact_sequential(model,p,1,'bayes-control')
        np.testing.assert_allclose(result['risk'],expected['risk'],atol=1e-12)
        np.testing.assert_allclose(result['mean_error'],expected['mean_error'],atol=1e-12)
        np.testing.assert_allclose(result['second'],expected['second'],atol=1e-12)
        self.assertLess(result['mass_error'],1e-12)

    def test_complexity_count_matches_actual_quotient_state_count(self):
        model=SymmetricControl(ladder_support(2),6)
        for s in itertools.product([0,1],repeat=4):model.value((1,)*4,s)
        self.assertEqual(len(model.values),complexity(budget=6)['canonical_states'])


if __name__=='__main__':unittest.main()
