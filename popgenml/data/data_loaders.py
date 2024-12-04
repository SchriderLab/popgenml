# -*- coding: utf-8 -*-
import numpy as np

from scipy.stats import beta
from scipy.interpolate import interp1d
import torch
from collections import OrderedDict
import random

class H5UDataGenerator(object):
    def __init__(self, ifile, keys = None, 
                 val_prop = 0.05, batch_size = 16, 
                 chunk_size = 4, pred_pop = 1, label_noise = 0.01, label_smooth = True):
        if keys is None:
            self.keys = list(ifile.keys())
            
            n_val = int(len(self.keys) * val_prop)
            random.shuffle(self.keys)
            
            self.val_keys = self.keys[:n_val]
            del self.keys[:n_val]
            
        self.ifile = ifile
        
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.label_noise = label_noise
        self.label_smooth = label_smooth
            
        self.length = len(self.keys) // (batch_size // chunk_size)
        self.val_length = len(self.val_keys) // (batch_size // chunk_size)
        
        self.n_per = batch_size // chunk_size
        
        self.pred_pop = pred_pop
        
        self.ix = 0
        self.ix_val = 0
            
    def define_lengths(self):
        self.length = len(self.keys) // (self.batch_size // self.chunk_size)
        self.val_length = len(self.val_keys) // (self.batch_size // self.chunk_size)
        
    def get_batch(self):
        X = []
        Y = []
        
        for key in self.keys[self.ix : self.ix + self.n_per]:
            x = np.array(self.ifile[key]['x_0'])
            y = np.array(self.ifile[key]['y'])
            
            X.append(x)
            Y.append(y)
            
        Y = np.concatenate(Y)
        
        if self.label_smooth:
            # label smooth
            ey = np.random.uniform(0, self.label_noise, Y.shape)
            
            Y = Y * (1 - ey) + 0.5 * ey
            
        self.ix += self.n_per
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(Y)
    
    def on_epoch_end(self):
        self.ix = 0
        self.ix_val = 0
        
        random.shuffle(self.keys)
        
    def get_val_batch(self):
        X = []
        Y = []
        
        for key in self.val_keys[self.ix_val : self.ix_val + self.n_per]:
            x = np.array(self.ifile[key]['x_0'])
            y = np.array(self.ifile[key]['y'])
            
            X.append(x)
            Y.append(y)
            
        Y = np.concatenate(Y)
            
        self.ix_val += self.n_per
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(Y)

class PopVectorImage(object):
    def __init__(self, size = 128, n_angles = 16, n_b = 65):
        self.n = size + 1
        self.size = size
        
        self.n_angles = n_angles
        self.n_b = n_b
        
        self.angles = np.linspace(0., np.pi, n_angles + 2)[1:-1]
        
        n_required = self.n // n_angles + 1
        
        self.modes = []
        x = np.arange(-size // 2, size // 2, 1)
        X, Y = np.meshgrid(x, x)

        # in pixels
        wavelengths = np.fft.fftfreq(size, d=1)
        wavelengths = (wavelengths[wavelengths > 0] ** -1)[24:]
        
        wavelengths = [wavelengths[u] for u in range(0, len(wavelengths), len(wavelengths) // n_required)]

        self.wavelengths = wavelengths
        self.indices = []
        self.modes_a = []
        
        for wavelength in wavelengths:
            for angle in self.angles:
                grating = np.sin(
                    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength
                )

                ft = np.fft.ifftshift(grating)
                ft = np.fft.fft2(ft)
                ft = np.fft.fftshift(ft)
                
                a = np.abs(ft)
                a = a[:, a.shape[1] // 2:]

                ii = np.unravel_index(np.argmax(a), a.shape)
                if ii in self.indices:
                    continue
                
                self.indices.append(ii)
                self.modes.append(grating)
                self.modes_a.append(a)
                
                if len(self.modes) == self.n:
                    break

            if len(self.modes) == self.n:
                break

        self.indices = np.array(self.indices)
        self.modes = np.array(self.modes)

    def im(self, p):
        ii = np.where(p == 1)[0]
        
        ret = np.sum(self.modes[ii], axis = 0)

        return ret
    
    def inv(self, im):
        
        ft = np.fft.ifftshift(im)
        ft = np.fft.fft2(ft)
        ft = np.fft.fftshift(ft)
        
        a = np.abs(ft)
        a = a[:, a.shape[1] // 2:]

        indices = []
        while len(indices) < self.n_b:
            ii_ = np.unravel_index(np.argmax(a), a.shape)
            
            e = np.linalg.norm(self.indices - np.array(ii_).reshape(1, -1), axis = -1)
            ix = np.argmin(e)

            if not (ix in indices):
                a -= self.modes_a[ix]
                indices.append(ix)
            
        ret = np.zeros((self.n, ))
        ret[indices] = 1

        return ret

class PriorSampler(object):
    def __init__(self, prior):
        x = np.loadtxt(prior)
        
        names = x[:,0]
        max_min_scale = list(map(tuple, x[:,1:]))
    
        self.prior = OrderedDict(zip(names, max_min_scale))
            
    def sample(self):
        params = []
        
        for p in self.params.keys():
            if type(self.params[p]) != tuple:
                # assume float or integer
                params.append(self.params[p])
            else:
                mi, ma, log_scale = self.params[p]
                if log_scale == -1:
                    params.append(np.random.uniform(mi, ma))
                else:
                    params.append(log_scale ** np.random.uniform(mi, ma))
                    
        return params

class MSPrimeFWLoader(object):
    def __init__(self, prior, simulator, size = 128, batch_size = 32, method = 'true', cdf = None, n_per = 2):
        # prior is a dictionary specifying a uniform distribution over some or all parameters
        # like {"N" : (100, 1000, -1), "alpha", (-3, -5, 10), "T" : 3.14159}
        # with max min and the log base (-1 for linear scale), in the case of single float the parameter is held constant
        
        # routine for loading the prior from a CSV
        if type(prior) == str:
            x = [u.strip().split(',') for u in open(prior, 'r').readlines()]
            names = [u[0] for u in x]
            max_min_scale = [tuple(map(float, u[1:])) for u in x]
        
            self.params = OrderedDict(zip(names, max_min_scale))
        elif prior is None:
            self.params = OrderedDict()
            
        self.method = method
        self.cdf = cdf
        self.n_per = n_per
        self.method = method
        self.batch_size = batch_size
        
        self.size = size
        
        # simulator is an object with .simulate(**params) function and subclassing BaseSimulator
        # see /popgenml/data/simulators.py
        self.simulator = simulator
        self.f_size = sum(self.simulator.n_samples) - 1
        
        
        if len(self.simulator.n_samples) > 1:
            s = sum(self.simulator.n_samples) - 1
            self.p_im = PopVectorImage(s, n_b = self.simulator.n_samples[0])
        else:
            self.p_im = None
        
        if self.cdf is None:
            self.compute_cdf()
        
    def get_batch(self):
        X = []
        for k in range(self.batch_size):
            _ = None
            while _ is None:
                _ = self.get_replicate_()
            X.extend(_)
            
        return torch.FloatTensor(np.array(X))
    
    def compute_cdf(self, n_samples = 512, n_bins = 1024):
        print('computing cdf...')
        
        mins = []
        maxs = []
        
        for ix in range(n_samples):
            W = None
            while W is None:
                W = self.get_W_()
                if W is None:
                    continue
                W = np.array(W)
                
                W = np.log(W + 1e-12)
                maxs.append(np.max(W))
                mins.append(np.min(W))
            
        bins = np.linspace(np.min(mins) - 1, np.max(maxs) + 1, n_bins + 1)
        h = np.zeros(len(bins) - 1)
        
        for ix in range(n_samples):
            W = None
            while W is None:
                W = self.get_W_()
                if W is None:
                    continue
                W = np.array(W)
                
                W = np.log(W + 1e-12)
            
                h += np.histogram(W.flatten(), bins, density = True)[0]

        x = bins[:-1] + np.diff(bins) / 2.

        H = h / n_samples
        h = np.cumsum(H)
        h /= np.max(h)

        ii = np.where(h > 0.)

        # cdf of the data
        y = beta.ppf(h, 5, 5)
        x = np.array(x)
        
        print('done!...')
        _, ii = np.unique(y, return_index = True)
        
        self.cdf = interp1d(x[ii], y[ii])
        
    def get_W_(self, n_per = -1):
        params = []
        
        for p in self.params.keys():
            if type(self.params[p]) != tuple:
                # assume float or integer
                params.append(self.params[p])
            else:
                mi, ma, log_scale = self.params[p]
                if log_scale == -1:
                    params.append(np.random.uniform(mi, ma))
                else:
                    params.append(log_scale ** np.random.uniform(mi, ma))
        
        if len(params) > 1:
            ret = self.simulator.simulate_fw_single(*params)
        else:
            ret = self.simulator.simulate_fw_single(None)
        
        if ret is not None:
            F, W, pop_mat, coal_times, X, sites, ts = ret
        else:
            return ret
        
        W = np.array(W)
        
        return W
    
    def get_replicate_(self, return_params = False):
        params = []
        
        for p in self.params.keys():
            if type(self.params[p]) != tuple:
                # assume float or integer
                params.append(self.params[p])
            else:
                mi, ma, log_scale = self.params[p]
                if log_scale == -1:
                    params.append(np.random.uniform(mi, ma))
                else:
                    params.append(log_scale ** np.random.uniform(mi, ma))
                    
        if len(params) > 1:
            ret = self.simulator.simulate_fw_single(*params)
        else:
            ret = self.simulator.simulate_fw_single(None)
            
        if ret is not None:
                    
            F, W, pop_mat, coal_times, Xmat, sites, ts = ret
        else:
            return ret
        
        W = np.array(W)
        F = np.array(F)
        F /= np.max(F)
        
        W = np.log(W + 1e-12)

        if self.cdf is not None:
            W = np.clip(W, self.cdf.x[0], self.cdf.x[-1])
            W = self.cdf(W)
                            
        i, j = np.triu_indices(self.f_size)
        i_, j_ = np.tril_indices(self.f_size)
        
        X = []

        d = W
        f = F
        
        im = np.zeros((self.size, self.size) + (3, ))
        
        if pop_mat is None:
            im[j_, i_,0] = f
            im[i_, j_, 0] = f
            im[j_, i_, 1] = d                
            im[j_, i_, 2] = d ** 0.5
        else:                
            im[j_,i_,0] = f
            im[i_,j_,0] = f
            im[j_, i_, 1] = d                
            im[:, :, 2] = (self.p_im.im(pop_mat) * 1.5 + 100) / 255
        
        X.append(im.transpose(2, 0, 1))
            
        X = np.array(X)
        
        if not return_params:
            return X
        else:
            return X, Xmat, sites, params

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from simulators import TwoPopMigrationSimulator
    
    sim = TwoPopMigrationSimulator(L = int(1e4))
    loader = MSPrimeFWLoader('priors/migration.csv', sim, method = 'relate')
    
    X = loader.get_replicate_()
    plt.imshow(X[0].transpose(1, 2, 0))
    plt.show()        
        
        