# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np

def msmc2(x, pos, L):
    chrom = np.ones(x.shape[1], dtype = np.uint8)
        
    pos = (np.concatenate([np.zeros((1,)), pos]) * L).astype(np.int32)
    diff = np.diff(pos).astype(np.int32)
    
    x = x.astype(np.object_)
    x[x == 0] = 'A'
    x[x == 1] = 'T'
    
    x = x.T
    
    x = x.sum(axis = -1)    
    x = np.concatenate([chrom.reshape(-1, 1), pos[1:].reshape(-1, 1), diff.reshape(-1, 1), x.reshape(-1, 1)], axis = 1)
    
    ofile = open('test.txt', 'w')
    for l in x:
        l = '\t'.join([str(u) for u in l]) + '\n'
        
        ofile.write(l)
        
    ofile.close()
    
