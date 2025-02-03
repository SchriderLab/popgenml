import numpy as np

from io_ import write_to_ms, load_ms
import sys

from seriate import seriate
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linear_sum_assignment

from scipy.sparse.linalg import eigs
from sklearn.metrics import pairwise_distances

"""
Pads sequences to the same length.
Takes list of 2d arrays [(n_sites, n_samples)] and pads to the max length and returns (b, max_sites, n_samples)
"""
def pad_sequences(sequences, max_length=None, padding_value=0):
    

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        padded_seq = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant', constant_values=padding_value)
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)

"""
x: (ind, sites) genotype matrix
pos: (sites,) array of positions
y: optional (same shape as x) for segmentation tasks
pop_sizes: tuple (n0, n1) or (n, )
out_shape: (n_pops, n_ind, n_sites) intended for the output.  If the genotype matrixs length > n_sites it is randomly cropped, 
    if < it is zero padded to n_sites 
metric: distance metric to use for sorting and/or matching
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
mode: [seriate_match (only for two populations, seriates the first population and matches it to chroms in the second),
       seriate (order individuals via the seriation algorithm and the given distance metric, see https://github.com/src-d/seriate),
       pad (pad the matrix on the site axis to the given size with no sorting. the number of individuals in outshape is ignored)]
"""
def format_matrix(x, pos, pop_sizes, y = None, 
                  out_shape = (2, 32, 128), 
                  metric = 'cosine', mode = 'seriate'):
    if len(pop_sizes) == 1:
        s0 = pop_sizes[0]
        s1 = 0
        
    else:         
        s0, s1 = pop_sizes
    n_pops, n_ind, n_sites = out_shape
            
    pos = np.array(pos)
    
    if x.shape[0] != s0 + s1:
        print('have x with incorrect shape!: {} vs expected {}'.format(x.shape[0], s0 + s1))
        return None, None
    
    if mode == 'seriate_match':
        x0 = x[:s0,:]
        x1 = x[s0:s0 + s1,:]
        
        if y is not None:
            y0 = y[:s0,:]
            y1 = y[s0:s0 + s1,:]
        
        # upsample to the number of individuals
        if s0 != n_ind:
            ii = np.random.choice(range(s0), n_ind)
            x0 = x0[ii,:]
            
            if y is not None:
                y0 = y0[ii,:]

        if s1 != n_ind:
            ii = np.random.choice(range(s1), n_ind)
            x1 = x1[ii,:]
            
            if y is not None:
                y1 = y1[ii,:]
 
        if x0.shape[1] > n_sites:
            ii = np.random.choice(range(x0.shape[1] - n_sites))
            
            x0 = x0[:,ii:ii + n_sites]
            x1 = x1[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
            
            if y is not None:
                y0 = y0[:,ii:ii + n_sites]
                y1 = y1[:,ii:ii + n_sites]
        else:
            to_pad = n_sites - x0.shape[1]
        
            if to_pad % 2 == 0:
                x0 = np.pad(x0, ((0,0), (to_pad // 2, to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2, to_pad // 2)))
                
                if y is not None:
                    y0 = np.pad(y0, ((0,0), (to_pad // 2, to_pad // 2)))
                    y1 = np.pad(y1, ((0,0), (to_pad // 2, to_pad // 2)))
                
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x0 = np.pad(x0, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                
                if y is not None:
                    y0 = np.pad(y0, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                    y1 = np.pad(y1, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
    
        # seriate population 1
        D = squareform(pdist(x0, metric = metric))
        D[np.isnan(D)] = 0.
        
        ii = seriate(D, timeout = 0.)
        
        x0 = x0[ii]
        
        if y is not None:
            y0 = y0[ii]
        
        D = cdist(x0, x1, metric = metric)
        D[np.isnan(D)] = 0.
        
        i, j = linear_sum_assignment(D)
        
        x1 = x1[j]
        
        if y is not None:
            y1 = y1[j]
        
        x = np.concatenate([np.expand_dims(x0, 0), np.expand_dims(x1, 0)], 0)
        if y is not None:
            y = np.concatenate([np.expand_dims(y0, 0), np.expand_dims(y1, 0)], 0)
        
    elif mode == 'pad':
        if x.shape[1] > n_sites:
            ii = np.random.choice(range(x.shape[1] - n_sites))
            
            x = x[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
        else:
            to_pad = n_sites - x.shape[1]

            if to_pad % 2 == 0:
                x = np.pad(x, ((0,0), (to_pad // 2, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x = np.pad(x, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
    
        
    elif mode == 'seriate': # one population
        if x.shape[1] > n_sites:
            ii = np.random.choice(range(x.shape[1] - n_sites))
            
            x = x[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
        else:
            to_pad = n_sites - x.shape[1]

            if to_pad % 2 == 0:
                x = np.pad(x, ((0,0), (to_pad // 2, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x = np.pad(x, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
                
        D = squareform(pdist(x, metric = metric))
        D[np.isnan(D)] = 0.
        
        ii = seriate(D, timeout = 0.)
        
        x = x[ii,:]
        
    return x, pos, y

def to_unique(X):
    site_hist = dict()
    
    ix = 0
    ii = dict()
    
    indices = []
    for k in range(X.shape[1]):
        x = X[:,k]
        #h = hashFor(x)
        h = ''.join(x.astype(str))
        if h in site_hist.keys():
            site_hist[h] += 1
            
        else:
            site_hist[h] = 1
            ii[h] = ix
            
            ix += 1
            
        indices.append(ii[h])
        
    site_hist = {v: k for k, v in site_hist.items()}
    
    ii = np.argsort(list(site_hist.keys()))[::-1]
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
    
    return x

def seriate_spectral(x):    
    C = pairwise_distances(x)
    C[np.where(np.isnan(C))] = 0.

    C = np.diag(C.sum(axis = 1)) - C
    _, v = eigs(C, k = 2, which = 'SM')

    f = v[:,1]
    ix = np.argsort(np.real(f))

    x = x[ix,:]
    
    return x, ix

def get_dist_matrix(x, metric = 'correlation'):
    D = squareform(pdist(x, metric = 'correlation'))
    
    

