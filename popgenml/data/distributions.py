# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import rv_continuous
from scipy.stats import randint

class UniformFloatDiscrete:
    """
    Selects uniformly from a specific list of floating-point numbers.
    Acts like a SciPy distribution but securely handles floats.
    """
    def __init__(self, values):
        # Sort and remove duplicates to ensure CDF math works correctly
        self.values = np.sort(np.unique(values))
        self.n = len(self.values)
        self.prob = 1.0 / self.n
        
        # Underlying uniform distribution for the integer indices [0, n-1]
        self._idx_dist = randint(0, self.n)

    def rvs(self, size=None, random_state=None):
        """Random variates."""
        indices = self._idx_dist.rvs(size=size, random_state=random_state)
        return self.values[indices]

    def pmf(self, x):
        """Probability Mass Function: P(X = x)."""
        x = np.asarray(x)
        # Returns 1/N if x is in our list, otherwise 0
        return np.where(np.isin(x, self.values), self.prob, 0.0)

    def logpmf(self, x):
        """Log of the Probability Mass Function."""
        # Using np.log on the PMF. 0s will become -inf safely.
        with np.errstate(divide='ignore'):
            return np.log(self.pmf(x))

    def cdf(self, x):
        """Cumulative Distribution Function: P(X <= x)."""
        x = np.asarray(x)
        # searchsorted finds where x would be inserted to maintain order.
        # This conveniently equals the number of elements <= x.
        counts = np.searchsorted(self.values, x, side='right')
        return counts * self.prob

    def mean(self):
        """Expected value."""
        return np.mean(self.values)
        
    def var(self):
        """Variance."""
        return np.var(self.values)

class _trunc_expon_gen(rv_continuous):
    """Stateless generator for the truncated exponential."""
    def _argcheck(self, lam):
        return lam > 0

    # Note that lam is explicitly passed into all these methods
    def _pdf(self, x, lam):
        denom = np.exp(-lam * self.a) - np.exp(-lam * self.b)
        return (lam * np.exp(-lam * x)) / denom

    def _cdf(self, x, lam):
        denom = np.exp(-lam * self.a) - np.exp(-lam * self.b)
        return (np.exp(-lam * self.a) - np.exp(-lam * x)) / denom

    def _ppf(self, q, lam):
        term = np.exp(-lam * self.a) - q * (np.exp(-lam * self.a) - np.exp(-lam * self.b))
        return -np.log(term) / lam

def TruncatedExponential(a, b, lam):
    """
    Factory function that returns a frozen SciPy distribution.
    This behaves exactly how popgenml expects.
    """
    # Initialize the generator with specific support bounds
    gen = _trunc_expon_gen(a=a, b=b, name='trunc_expon', shapes='lam')
    
    # Return the 'frozen' distribution by calling it with the shape parameter
    return gen(lam=lam)
