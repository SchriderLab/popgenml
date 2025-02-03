# -*- coding: utf-8 -*-

import allel
import numpy as np
# functions to take in (n, sites) binary haplotype array and positional array (sites,) and output a statistic

# pos here are integers
def theta_pi(x, pos):
    h = allel.HaplotypeArray(x.astype(np.uint8))
    ac = h.count_alleles()
    
    return allel.sequence_diversity(pos, ac)

def watterson_theta(x, pos, ploidy = 2):
    g = allel.HaplotypeArray(x.astype(np.uint8)).to_genotypes(ploidy = ploidy)
    
    ac = g.count_alleles()
    
    return allel.watterson_theta(pos, ac)

def tajimas_d(x, pos, ploidy = 2):
    g = allel.HaplotypeArray(x.astype(np.uint8)).to_genotypes(ploidy = ploidy)
    
    ac = g.count_alleles()
    return allel.tajima_d(ac, pos=pos)

# returns the mean and standard deviation of pairwise linkage disequilibrium
def ld_stats(x, ploidy = 2):
    g = allel.HaplotypeArray(x.astype(np.uint8)).to_genotypes(ploidy = ploidy)

    gn = g.to_n_alt(fill=-1)
    r = allel.rogers_huff_r(gn)
    
    return np.array([r.mean(), r.std()])

# returns the mean and std
def het_diversity(x, ploidy = 2):
    g = allel.HaplotypeArray(x.astype(np.uint8)).to_genotypes(ploidy = ploidy)

    r = allel.heterozygosity_observed(g)
    
    return np.array([r.mean(), r.std()])

# x: (ind, sites)
def sfs(x):
    sfs, _ = np.histogram(x.sum(0), bins = list(range(1, x.shape[0])))
    sfs = sfs.astype(np.float32)
    sfs /= np.sum(sfs)
    
    return sfs

if __name__ == '__main__':
    x = np.random.choice([0., 1.], size = (40, 1000))
    pos = sorted(np.random.randint(0, 5000, size = 1000))
    
    print(theta_pi(x, pos))
    print(watterson_theta(x, pos))
    print(tajimas_d(x, pos))
    print(sfs(x))
    print(ld_stats(x))
    print(het_diversity(x))
    