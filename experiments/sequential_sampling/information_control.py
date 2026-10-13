"""Exploratory small-case Bayes Bellman control, not the proposed gated policy."""
from functools import lru_cache
import numpy as np
from .core import RiskModel, target, child


class BayesControl(RiskModel):
    def __init__(self,points,budget,deadline_seconds=600):
        super().__init__(points,budget,deadline_seconds=deadline_seconds)
        self.value=lru_cache(maxsize=None)(self._value)
        self.actions={}

    def _value(self,n,s):
        self.check_deadline();weights=self.posterior(n,s)
        if sum(n)==self.budget:
            delta=target(np.asarray(s)/np.asarray(n))-self.truth
            return float(weights@np.sum(delta*delta,axis=1))
        probabilities=weights@self.points
        choices=[probabilities[j]*self.value(*child(n,s,j,1))
                 +(1-probabilities[j])*self.value(*child(n,s,j,0)) for j in range(self.k)]
        j=int(np.argmin(choices));self.actions[(n,s)]=j
        return float(choices[j])

    def choose(self,n,s,method):
        if method!='bayes-control':return super().choose(n,s,method)
        self.value(n,s)
        return self.actions[(n,s)],'bayes-control'
