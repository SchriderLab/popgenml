# -*- coding: utf-8 -*-

class BaseSimulator(object):
    # L is the size of the simulation in base pairs
    def __init__(self, L = int(1e5)):
        self.L = L
        return
    
    # should return X, sites
    # where X is a binary matrix and sites is an array with positions corresponding to the second axis
    def simulate(self):
        return
    
