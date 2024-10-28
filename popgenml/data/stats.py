# -*- coding: utf-8 -*-

import allel
import numpy as np
# functions to take in (n, sites) binary haplotype array and positional array (sites,) and output a statistic

# pos here are integers
def theta_pi(x, pos):
    h = allel.HaplotypeArray(x)
    ac = h.count_alleles()
    
    return allel.sequence_diversity(pos, ac)

def watterson_theta(x, pos):
    g = allel.HaplotypeArray(x).to_genotypes()
    
    ac = g.count_alleles()
    
    return allel.watterson_theta(pos, ac)

def tajimas_d(x, pos):
    g = allel.HaplotypeArray(x).to_genotypes()
    
    ac = g.count_alleles()
    return allel.tajima_d(ac, pos=pos)

def sfs(x):
    sfs, _ = np.histogram(x.sum(0), bins = list(range(1, x.shape[0])))
    sfs = sfs.astype(np.float32)
    sfs /= np.sum(sfs)
    
    return sfs



