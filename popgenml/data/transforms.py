# -*- coding: utf-8 -*-

import pywt
import numpy as np

"""
Base placeholder class for computing a transform.  Defaults to the identity.  
"""
class Transform(object):
    def __init__(self):
        return
    
    def transform(self, x):
        return x

"""
Takes an (n, l) alignment matrix and computes the continuous wavelet transform.
"""
class WaveletTransform(object):
    def __init__(self, N = 256, delta = 2, wavelet = 'mexh', method = 'fft'):
        self.scales = np.arange(1, N + 1, delta)
        self.wavelet = wavelet
        self.method = method
        
    def transform(self, x):
        ret = []
        x = 2 * x - 1
        
        for x_ in x: 
            xw, _ = pywt.cwt(x_, self.scales, self.wavelet, method = self.method)
            xw = np.concatenate([x_.reshape(1, x.shape[0], x.shape[1]), xw])
        
            ret.append(xw)
            
        return np.concatenate(ret)

"""
Computes various summary statistics across windows in a single-population alignment.
"""
class StatTransform(object):
    def __init__(self):
        return
    
    """
    x: (n, l) uint8
    pos: (l,) float32 (0 to 1)
    """
    def transform(self, x, pos):
        return
    
