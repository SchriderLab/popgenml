# -*- coding: utf-8 -*-
import glob
import os
import tskit
import random

from popgenml.data.functions import tree_to_graph

from torch_geometric.data import Data, Batch, DataLoader
import torch

class MSPrimeTreeDirLoader(object):
    def __init__(self, idir, n = 200, val_prop = 0.05, batch_size = 1):
        self.idir = idir
        self.n = n
        self.batch_size = batch_size
        
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
        ret = True
        
        while ret:
            x, e = tree_to_graph(tree, self.n)
        
            X.append(x)
            edge_index.append(e)
        
            ret = tree.next()
        
        return X, edge_index
    
    def get_batch(self):
        X = []
        indices = []
        
        for k in range(self.batch_size):
            x, e = self.get_replicate()
            
            X.extend(x)
            indices.extend(e)
            
        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [
                Data(x=torch.FloatTensor(X[k]), edge_index=indices[k])
                for k in range(len(indices))
            ]
        )
        
        return batch
    
if __name__ == '__main__':
    loader = MSPrimeTreeDirLoader('data/tree_pheno')
    batch = loader.get_batch()
    
    print(batch.x.shape)
    
    