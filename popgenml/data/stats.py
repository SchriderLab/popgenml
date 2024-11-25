# -*- coding: utf-8 -*-

import allel
import numpy as np
# functions to take in (n, sites) binary haplotype array and positional array (sites,) and output a statistic

def to_unique(X):
    site_hist = dict()
    
    ix = 0
    ii = dict()
    
    indices = []
    for k in range(X.shape[1]):
        x = X[:,k]
        h = ''.join(x.astype(str))
        if h in site_hist.keys():
            site_hist[h] += 1
            
        else:
            site_hist[h] = 1
            ii[h] = ix
            
            ix += 1
        
        # this site is the ith unique one found
        indices.append(ii[h])
        
    site_hist = {v: k for k, v in site_hist.items()}
    
    ii = np.argsort(list(site_hist.keys()))[::-1]
    # resort the indices as well
    indices = [indices[u] for u in indices]
    
    v = sorted(list(site_hist.keys()), reverse = True)
    
    _ = []
    for v_ in v:
        x = site_hist[v_]
        x = np.array(list(map(float, [u for u in x])))
        
        _.append(x)
    
    x = np.array(_)
    v = np.array(v, dtype = np.float32).reshape(-1, 1)
    v /= np.sum(v)
    
    x = np.concatenate([x, v], -1)
    
    return x, np.array(indices, dtype = np.int32)

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
    