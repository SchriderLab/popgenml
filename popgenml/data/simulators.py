# -*- coding: utf-8 -*-
from popgenml.data.io_ import read_slim

import msprime
import numpy as np
import os

import logging
import random
import subprocess

from numpy.polynomial.chebyshev import Chebyshev
from pkg_resources import resource_filename

"""
Base class to define the functionality for different pop size history priors.  These are used to sample population size trajectoies that can be approximated in msprime or another simulator
as piecewise constant
    __init__(N):
        N (float): 'central' estimate of the effective popsize (if N varies back in time then you might choose the harmonic mean of the effective popsize up to E[tmrca]?)

N defines the scale the prior uses.  Other than N, we define a prior only by a strictly positive random curve defined up to some set time which is multiplied by N to produce a history
over time in generations.  This class defaults to a constant pop size = N

Methods:
    sample(return_co = False): 
        returns tuple (list, list) of times and pop sizes (and optionally any coefficients used in the construction of the curve (array-like))

"""
class PiecewisePopSizePrior(object):
    def __init__(self, N):
        self.N = N

    def sample_curve(self):
        t = [0]
        N = np.ones(1)
        
        return t, N, None

    def sample(self, return_co = True):
        t, N, co = self.sample_curve()
        
        N *= self.N
        
        if return_co and (co is not None):
            return t, N, co
        else:
            return t, N

"""
Class to generate population size histories of the form N(t) = (eps * f(t) + 1) * N where f(t) is a 
random sum of Chebyshev polynomials scaled from -1 to 1 and eps is drawn from a uniform distribution from min_eps to max_eps.

    N: The mean pop size in the formula for N(t)
    max_K: The max order of polynomial in the sum
    n_time_points: The number of time points from log(time) = 0 to max_log_time to sample the pop size function for the demography
"""
class ChebyshevHistory(PiecewisePopSizePrior):
    def __init__(self, N = 75000, max_K = 12, n_time_points = 128, min_eps = 0.05, max_eps = 0.9,
                 max_log_time = 11):
        super().__init__(N)
        
        self.max_K = max_K
        self.n_time_points = n_time_points
        self.max_eps = max_eps
        self.min_eps = min_eps
        
        self.max_log_time = max_log_time
        
    def get_N(self, co, eps):
        p = Chebyshev(co)
    
        x = np.linspace(-1., 1., self. n_time_points)
        # get the curve with range (-1, 1)
        y = p(x)
        
        max_p = np.max(y)
        min_p = np.min(y)
        
        y = ((y - min_p) / (max_p - min_p)) * 2 - 1
        
        N = y * eps + 1
        
        t = [0] + list(np.exp(np.linspace(0, self.max_log_time, self.n_time_points - 1)))
        
        return t, N
        
    def sample_curve(self):
        # sample the number of Cheby polynomials to include
        K = self.max_K - 1
        
        # sample ~ N(0, 1) (standard gaussian) coefficients
        co = np.random.normal(0., 1., K + 1)
        
        p = Chebyshev(co)
        
        x = np.linspace(-1., 1., self. n_time_points)
        
        # get the curve with range (-1, 1)
        y = p(x)
        
        max_p = np.max(y)
        min_p = np.min(y)
        
        y = ((y - min_p) / (max_p - min_p)) * 2 - 1
        
        eps = np.random.uniform(self.min_eps, self.max_eps)
        
        N = y * eps + 1
        
        co = np.concatenate([np.array([eps]), np.pad(co, ((0, self.max_K - co.shape[0])))])

        t = [0] + list(np.exp(np.linspace(0, self.max_log_time, self.n_time_points - 1)))

        return t, N, co
    
"""
Base class to define some the attributes common to the supported simulators.  These are:
    L (int): the length of the simulated chromosome in base pairs
    mu (float): the mutation rate
    r: (float): the recombination rate
    n_samples (list of int): the number of samples taken for each population
"""
class BaseSimulator(object):
    def __init__(self, L, mu, r, n_samples):
        self.L = L
        self.mu = mu
        self.r = r        
        self.n_samples = n_samples

"""
Class for simulating with msprime.

By subclassing this class and re-defining make_demography(), you can make custom simulators that draw from specified
priors for parameters such as population growth rates, migration rates, etc.

See StepStoneSimulator for an example.
"""        
class BaseMSPrimeSimulator(BaseSimulator):
    # L is the size of the simulation in base pairs
    # specify mutation rate
    
    # immutable properties of the simulator are defined here:
    # L: the size of the simulation in base pairs
    # mu: mutation rate
    # r: recombination rate
    # whether or not diploid individuals are simulated (vs haploid)
    # the number of samples
    def __init__(self, L = int(1e5), mu = 1.5e-8, r = 1.007e-8, ploidy = 1, 
                 n_samples = [16], N = 75000):
        self.L = L
        self.mu = mu
        self.r = r
        self.n_samples = n_samples
        
        self.ploidy = ploidy
        
        self.sample_size = sum(n_samples)

        self.co = None
        self.demography = None
        
        self.N = N
        
        return
    
    # here's where you should define how to use the prior to create the msprime demography
    # must be over-written in subclass
    def make_demography(self):
        demography = msprime.Demography()
        
        return demography
    
    def simulate(self, *args, verbose = False):
        # get the current logging level
        logger = logging.getLogger(__name__)
        current_level = logger.getEffectiveLevel()
        logging.basicConfig(level = logging.ERROR)
        """
        if not verbose:
            # to quiet down msprime
            logging.basicConfig(level = logging.ERROR, force = True)
        else:
            logging.basicConfig(level = logging.INFO, force = True)
        """
        self.demography = self.make_demography()
        
        # simulate ancestry
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = sum(self.n_samples),
            sequence_length = self.L,
            recombination_rate=self.r,
            ploidy = self.ploidy,
            demography = self.demography,
        )
        
        # set the logging level back to what it was before the call to msprime
        #logging.basicConfig(level = current_level, force = True)
        
        return self.mutate_and_return_(ts)
    
    def mutate_and_return_(self, ts):
        result = dict()
        
        # simulate mutations, binary discrete model
        mutated_ts = msprime.sim_mutations(ts, rate=self.mu, model=msprime.BinaryMutationModel())
        
        X = mutated_ts.genotype_matrix()
        X[X > 1] = 1
        X = X.T

        sites = [u.position for u in list(mutated_ts.sites())]
        sites = np.array(sites) / self.L
        
        result['x'] = X
        result['pos'] = sites
        result['ts'] = mutated_ts
        
        return result

"""
Constant pop size simulator.
"""  
class SimpleCoal(BaseMSPrimeSimulator):
    def __init__(self, N = 75000, **kwargs):
        super().__init__(**kwargs)
        
        self.N = N
        
    def make_demography(self):
        demography = msprime.Demography()
        
        demography.add_population(name="A", initial_size=self.N)
        
        return demography
            
class StepStoneSimulator(BaseMSPrimeSimulator):
    def __init__(self, prior = ChebyshevHistory(), **kwargs):
        super().__init__(**kwargs)
        
        self.prior = prior
        
    def make_demography(self):
        demography = msprime.Demography()
        
        t, N, co = self.prior.sample()
        self.co = co
        
        Nt = list(zip(N, t))
        
        N0, _ = Nt[0]
        demography.add_population(name="A", initial_size=N0)
        
        for N1, T in Nt[1:]:
            demography.add_population_parameters_change(time=T, initial_size=N1)
            
        return demography    
    
"""
Experimental.  Class to simulate with SLiM and get Python natives.  
"""
class SlimSimulator(object):
    """
    script (str): points to a slim script
    args (str): format string for slim command with args if needed
    """
    def __init__(self, script = os.path.join(resource_filename('popgenml', 'slim'), 'introg_bidirectional.slim'),
                 args = "-d sampleSizePerSubpop={} -d donorPop={} -d st={} -d mt={}", 
                 n_samples = 64, L = int(1e4)):
        self.script = script
        self.args = args
        self.n_samples = n_samples
        self.L = L
        
    def simulate(self, *args):
        args = args + (self.script,)
        
        seed = random.randint(0, 2**32-1)
        slim_cmd = "slim -seed {} -d physLen={} ".format(seed, self.L)

        if self.args is not None:
            slim_cmd += self.args.format(*args)
            slim_cmd += " {}".format(self.script)    
        else:
            slim_cmd += "{}".format(self.script)

        procOut = subprocess.Popen(
            slim_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, err = procOut.communicate()
                    
        X, pos, y_ = read_slim(output, self.n_samples, self.L)
        pos = np.array(pos)
        X = np.array(X)
        
        y = np.zeros(X.shape)
        
        for ix, start_end in enumerate(y_):
            if len(start_end) == 0:
                continue
            
            for start,end in start_end:
            
                ii = np.where((pos >= start) & (pos <= end))[0]
                
                y[ix, ii] = 1.
        
        return X, pos, y
            
