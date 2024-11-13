# -*- coding: utf-8 -*-
import numpy as np

from scipy.stats import beta
from scipy.interpolate import interp1d

class MSPrimeFWLoader(object):
    def __init__(self, prior, simulator, batch_size = 32, method = 'relate', cdf = None, n_per = 2):
        # prior is a dictionary specifying a uniform distribution over some or all parameters
        # like {"N" : (100, 1000, -1), "alpha", (-3, -5, 10), "T" : 3.14159}
        # with max min and the log base (-1 for linear scale), in the case of single float the parameter is held constant
        
        # routine for loading the prior from a CSV
        if type(prior) == str:
            x = np.loadtxt(prior)
            
            names = x[:,0]
            max_min_scale = list(map(tuple, x[:,1:]))
        
            prior = dict(zip(names, max_min_scale))
                
        self.prior = prior
        self.method = method
        self.cdf = cdf
        self.n_per = n_per
        self.method = method
        
        # simulator is an object with .simulate(**params) function and subclassing BaseSimulator
        # see /popgenml/data/simulators.py
        self.simulator = simulator
        
        if self.cdf is None:
            self.compute_cdf()
        
    def compute_cdf(self, n_samples = 4096, n_bins = 1024):
        mins = []
        maxs = []
        
        for ix in range(n_samples):
            _, W, _, _ = self.get_replicate_()
            W = np.array(W)
            
            W = np.log(W + 1e-12)
            maxs.append(np.max(W))
            mins.append(np.min(W))
        
        bins = np.linspace(np.min(mins) - 1, np.max(maxs) + 1, n_bins + 1)
        h = np.zeros(len(bins) - 1)
        
        for ix in range(n_samples):
            _, W, _, _ = self.get_replicate_()
            W = np.array(W)
            
            W = np.log(W + 1e-12)
        
            h += np.histogram(W.flatten(), bins, density = True)[0]

        x = bins[:-1] + np.diff(bins) / 2.

        H = h / n_samples
        h = np.cumsum(H)
        h /= np.max(h)

        ii = np.where(h > 0.)

        # cdf of the data
        y = beta.ppf(h, 5, 5)
        x = np.array(x)
        
        _, ii = np.unique(y, return_index = True)
        
        self.cdf = interp1d(x[ii], y[ii])
        
    def get_replicate_(self):
        params = []
        
        for p in self.params.keys():
            if type(self.params[p]) != tuple:
                # assume float or integer
                params.append(self.params[p])
            else:
                mi, ma, log_scale = self.params[p]
                if log_scale == -1:
                    params.append(np.random.uniform(mi, ma))
                else:
                    params.append(log_scale ** np.random.uniform(mi, ma))
                    
        F, W, pop_vectors, coal_times = self.simulator.simulate_fw(*params, method = self.method)
        W = np.array(W)
        F = np.array(F)
        
        W = np.log(W + 1e-12)

        D[D < self.cdf.x[0]] = self.cdf.x[0]
        D[D > self.cdf.x[-1]] = self.cdf.x[-1]

        D = self.cdf(D)
        
        