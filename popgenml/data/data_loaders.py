# -*- coding: utf-8 -*-
import numpy as np

class MSPrimeFWLoader(object):
    def __init__(self, prior, simulator, batch_size = 32, method = 'relate', cdf = None, n_per = 2):
        # prior is a dictionary specifying a uniform distribution over some parameters
        # like {"N" : (100, 1000, -1), "alpha", (-3, -5, 10), "T" : 3.14159}
        # with max min and the log base (-1 for linear scale), in the case of single float the parameter is held constant
        self.prior = prior
        self.method = method
        self.cdf = cdf
        self.n_per = n_per
        self.method = method
        
        # simulator is an object with .simulate(**params) function and subclassing BaseSimulator
        # see /popgenml/data/simulators.py
        self.simulator = simulator
        
    def get_replicate_(self, n_per = 2):
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
        
        
        