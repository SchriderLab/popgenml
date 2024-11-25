# -*- coding: utf-8 -*-

import gzip
import numpy as np

def write_to_ms(ofile, X, sites, params = None):
    ofile = open(ofile, 'w')
    
    if params:
        header = '// ' + ' '.join(['{0:04f}'.format(u) for u in params]) + '\n'
    else:
        header = '//\n'
        
    ofile.write(header)
    n_segsites = X.shape[1]
    ofile.write('segsites: {}\n'.format(n_segsites))
    pos_line = 'positions: ' + ' '.join(['{0:08f}'.format(u) for u in sites]) + '\n'
    ofile.write(pos_line)
    
    for x in X:
        line = ''.join(list(map(str, list(x)))) + '\n'
        ofile.write(line)
        
    ofile.close()
    
def split(word):
    return [char for char in word]
    
######
# generic function for msmodified
# ----------------
# takes a gzipped ms file
# returns a list of genotype matrices, introgressed allele matrices (if *.anc file is provided),
# a list of position vectors, and a list of the parameters listed in the \\ line if any
def load_ms(msFile, ancFile = None, n = None, flip_alleles = True):
    msFile = gzip.open(msFile, 'r')

    # no migration case
    if ancFile is not None:
        ancFile = gzip.open(ancFile, 'r')

    ms_lines = [u.decode('utf-8') for u in msFile.readlines()]
    ms_lines = [u for u in ms_lines if (not '#' in u)]

    idx_list = [idx for idx, value in enumerate(ms_lines) if ('//' in value)] + [len(ms_lines)]
        
            
    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1] += ['\n']

    if ancFile is not None:
        anc_lines = [u.decode('utf-8') for u in ancFile.readlines()]
    else:
        anc_lines = None
        
    X = []
    Y = []
    P = []
    intros = []
    params = []
    
    for chunk in ms_chunks:
        line = chunk[0]
        params_ = list(map(float, line.replace('\n', '').split('\t')[1:]))
        
        if len(params_) == 0:
            params_ = list(map(float, line.replace('\n', '').split()[1:]))
        
    
        if '*' in line:
            intros.append(True)
        else:
            intros.append(False)
        
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)
        _ = [list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]]
        _ = [u for u in _ if len(u) > 0]
        
        x = np.array(_, dtype = np.uint8)
        
        if x.shape[0] == 0:
            X.append(None)
            Y.append(None)
            P.append(None)
            params.append(None)
            continue
        
        if flip_alleles:
            # destroy the perfect information regarding
            # which allele is the ancestral one
            for k in range(x.shape[1]):
                if np.sum(x[:,k]) > x.shape[0] / 2.:
                    x[:,k] = 1 - x[:,k]
                elif np.sum(x[:,k]) == x.shape[0] / 2.:
                    if np.random.choice([0, 1]) == 0:
                        x[:,k] = 1 - x[:,k]
        
        if anc_lines is not None:
            y = np.array([list(map(int, split(u.replace('\n', '')))) for u in anc_lines[:len(pos)]], dtype = np.uint8)
            y = y.T
            
            del anc_lines[:len(pos)]
        else:
            y = None
            
        if len(pos) == x.shape[1] - 1:
            pos = np.array(list(pos) + [1.])
            
        assert len(pos) == x.shape[1]
        
        if n is not None:
            x = x[:n,:]
            y = y[:n,:]
            
        X.append(x)
        Y.append(y)
        P.append(pos)
        params.append(params_)
        
    return X, Y, P, params