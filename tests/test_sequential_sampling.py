import itertools
import unittest
import numpy as np
from experiments.sequential_sampling.core import RiskModel, target, jacobian, simulate
from experiments.sequential_sampling.evaluation import exact_sequential, outcome_mass, metrics, fixed_counts, allocate, average_permutations


class SequentialSamplingTests(unittest.TestCase):
    def test_one_action_is_blind_before_terminal_quota(self):
        points=np.array([[.2,.8,.6,.4],[.9,.1,.3,.7]])
        model=RiskModel(points,16)
        for n,s in [((1,1,1,1),(0,1,1,0)),((3,3,3,3),(1,2,0,3))]:
            np.testing.assert_allclose(model.scores(n,s)['gain'],0,atol=1e-14)

    def test_randomized_policy_bias_rotates_before_averaging(self):
        permutations=np.array([[0,1],[1,0]])
        mean=np.array([[1.,2.],[3.,4.]])
        second=np.array([np.outer(x,x) for x in mean])
        raw={'risk':np.trace(second,axis1=1,axis2=2),'second':second,'mean_error':mean,
             'bias_squared':np.sum(mean**2,axis=1),'variance':np.zeros(2)}
        averaged=average_permutations(raw,permutations)
        np.testing.assert_allclose(averaged['mean_error'],[[2.5,2.5]])
        np.testing.assert_allclose(averaged['bias_squared'],[12.5])
        np.testing.assert_allclose(averaged['risk'],[15.])
        np.testing.assert_allclose(averaged['variance'],[2.5])

    def test_K2_jacobian_is_exactly_zero_outside_floor(self):
        np.testing.assert_equal(jacobian([.9,.1]),np.zeros((2,2)))
        np.testing.assert_equal(jacobian([1/6,5/6]),np.zeros((2,2)))
        np.testing.assert_equal(allocate(np.zeros(4),12),[3,3,3,3])

    def test_policy_path_mass_and_uniform_terminal_risk(self):
        p=np.array([[.2,.8],[.5,.5]])
        model=RiskModel(np.array([[.2,.8],[.8,.2],[.4,.6],[.6,.4]]),4)
        result=exact_sequential(model,p,1,'uniform')
        states=list(itertools.product(range(3),repeat=2))
        expected=metrics(outcome_mass(states,[2,2],p),target(np.array(states)/2),target(p))
        np.testing.assert_allclose(result['risk'],expected['risk'],atol=1e-14)
        self.assertLess(result['mass_error'],1e-14)
        np.testing.assert_allclose(result['risk'],result['bias_squared']+result['variance'],atol=1e-14)
        for method in ['variance-only','JADA']:
            for s in itertools.product(range(3),repeat=4):
                n=fixed_counts(s,2,16,method)
                self.assertEqual(sum(n),16)
                self.assertTrue(np.all(n>=3))

    def test_target_and_jacobian_in_both_floor_regions(self):
        for p, step in [(np.array([.9,.7,.4,.1]), 1e-7),
                        (np.array([.5,.5,.5,.5]), 1e-9)]:
            fd=np.column_stack([(target(p+step*v)-target(p-step*v))/(2*step)
                                for v in np.eye(4)])
            np.testing.assert_allclose(fd,jacobian(p),rtol=1e-6,atol=1e-5)
        self.assertAlmostEqual(float(target([.9,.1])@target([.9,.1])),1.)
        np.testing.assert_equal(target([.5,.5]),[0,0])

    def test_completion_risk_against_direct_outcome_enumeration(self):
        points=np.array([[.2,.8],[.6,.4],[.5,.5]])
        model=RiskModel(points,4)
        n=(1,1);s=(0,1);actual=model.uniform_risk(n,s)
        expected=np.zeros(3)
        for x,y in itertools.product([0,1],repeat=2):
            prob=(points[:,0] if x else 1-points[:,0])*(points[:,1] if y else 1-points[:,1])
            delta=target(np.array([x,1+y])/2)-target(points)
            expected+=prob*np.sum(delta*delta,axis=1)
        np.testing.assert_allclose(actual,expected,atol=1e-14)

    def test_predictable_history_posterior_and_predictive_identity(self):
        points=np.array([[.2,.8],[.6,.4],[.5,.5]])
        model=RiskModel(points,4)
        w=model.posterior((2,1),(1,1))
        raw=points[:,0]*(1-points[:,0])*points[:,1]
        np.testing.assert_allclose(w,raw/raw.sum())
        parent=model.posterior((1,1),(0,1))
        rho=parent@points[:,0]
        mixture=rho*model.posterior((2,1),(1,1))+(1-rho)*model.posterior((2,1),(0,1))
        np.testing.assert_allclose(mixture,parent,atol=1e-14)

    def test_information_contract_and_exact_budget(self):
        model=RiskModel(np.array([[.2,.8],[.8,.2],[.4,.6],[.6,.4]]),8)
        streams=np.array([[True,False,True,False,True,False,True,False],
                          [False,True,False,True,False,True,False,True]])
        for method in ['uniform','ungated','gated']:
            result=simulate(model,streams,1,method)
            self.assertEqual(sum(result['n']),8)
            self.assertTrue(all(x>=1 for x in result['n']))
            self.assertTrue(all(0<=s<=n for s,n in zip(result['s'],result['n'])))
        scores=model.scores((1,1),(0,1))
        np.testing.assert_equal(scores['gain'][:,0],0)

    def test_relabelled_completion_and_scores(self):
        points=np.array(list(itertools.product([.2,.4,.6,.8],repeat=2)))
        model=RiskModel(points,5)
        other=RiskModel(points[:,::-1],5)
        # Unique minimum count: no unresolved tie in this completion horizon.
        a=model.uniform_risk((1,3),(1,1))
        b=other.uniform_risk((3,1),(1,1))
        np.testing.assert_allclose(a,b,atol=1e-14)


if __name__=='__main__':
    unittest.main()
