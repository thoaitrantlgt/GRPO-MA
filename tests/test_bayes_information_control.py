import unittest
import numpy as np
from experiments.sequential_sampling.information_control import BayesControl
from experiments.sequential_sampling.evaluation import exact_sequential


class BayesInformationControlTests(unittest.TestCase):
    def test_finite_prior_optimal_control_dominates_balanced_completion(self):
        points=np.array([[.1,.7],[.7,.1],[.3,.8],[.8,.3]])
        model=BayesControl(points,4)
        for n,s in [((1,1),(0,0)),((1,1),(0,1)),((2,1),(1,1))]:
            posterior=model.posterior(n,s)
            self.assertLessEqual(model.value(n,s),posterior@model.uniform_risk(n,s)+1e-12)
        uniform=exact_sequential(model,points,1,'uniform')
        optimal=exact_sequential(model,points,1,'bayes-control')
        self.assertLessEqual(optimal['risk'].mean(),uniform['risk'].mean()+1e-12)
        self.assertLess(optimal['mass_error'],1e-12)


if __name__=='__main__':unittest.main()
