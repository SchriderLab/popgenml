# -*- coding: utf-8 -*-
import glob
import os
import tskit
import random

from popgenml.data.functions import tree_to_graph

class MSPrimeTreeDirLoader(object):
    def __init__(self, idir, n = 200, val_prop = 0.05):
        self.idir = idir
        self.n = n
        
        self.ifiles = glob.glob(os.path.join(self.idir, '*.trees'))
        random.shuffle(self.ifiles)
        
        n_val = int(len(self.ifiles) * val_prop)

        self.ifiles_val = self.ifiles[:n_val]
        del self.ifiles[:n_val]
        
        # counters
        self.ix = 0
        self.ix_val = 0

    def get_replicate(self):
        ifile = self.ifiles[self.ix]

        self.ix += 1

        ts = tskit.load(ifile)
        
        X = []
        edge_index = []
        
        tree = ts.first()
        while True:
            x, e = tree_to_graph(tree, self.n)
        
            X.append(x)
            edge_index.append(e)
        
        return X, edge_index
