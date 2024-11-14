# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import pickle

import torch
import sys
import numpy as np
from collections import deque

import matplotlib
#matplotlib.use('Agg')

import glob
import matplotlib.pyplot as plt
from scipy.special import softmax

sys.path.append('popgenml/data')
sys.path.append('popgenml/models')

from simulators import PopSplitSimulator, BottleNeckSimulator, SecondaryContactSimulator
from layers import TransformerConv
import time
import argparse
import logging

import random

class NPZLoader(object):
    def __init__(self, idir, pkl = None, batch_size = 32, val_prop = 0.05):
        if pkl is None:
            self.ifiles = glob.glob(os.path.join(idir, '*/*/*.npz')) + glob.glob(os.path.join(idir, '*.npz'))
            random.shuffle(self.ifiles)    
            
            
            n_val = int(len(self.ifiles) * val_prop)
            
            self.ifiles_val = self.ifiles[:n_val]
            del self.ifiles[:n_val]
            
        else:
            self.ifiles, self.ifiles_val, self.L = pickle.load(open(pkl, 'rb'))
        
        self.batch_size = batch_size

        
    def get_batch(self, yix = 1):
        ii = np.random.choice(range(len(self.ifiles)), self.batch_size, replace = False)
        
        X = []
        y = []

        for ii_ in ii:
            x = np.load(self.ifiles[ii_])
            x_ = x['x']
            
            # shuffle data
            ii0 = list(range(44))
            ii1 = list(range(44,60))
            ii2 = [60]
            
            random.shuffle(ii0)
            random.shuffle(ii1)
            
            x_ = x_[:, ii0 + ii1 + ii2]
            X.append(x_)

            y_ = x['y']
            y_ = np.array([y_[1], y_[2], y_[4], np.log10(y_[5])])
            y.append(y_)
        return X, y

    def get_val(self, yix = 1):
        X = []
        y = []
        
        for ix in range(len(self.ifiles_val)):
            try:
                x = np.load(self.ifiles_val[ix])
                x_ = x['x']
                
                
            except:
                continue
            
            X.append(x_)

            y_ = x['y']
            y_ = np.array([y_[1], y_[2], y_[4], np.log10(y_[5])])
            y.append(y_)
        
        return X, y
    
from scipy.spatial.distance import pdist, squareform
from seriate import seriate
    
def sort_array(x, metric = 'correlation'):
    D = squareform(pdist(x, metric = metric))

    ii = seriate(D, timeout = 0.)
    return x[ii]

def to_count(x, t, n = 40):    
    x = np.digitize(x, t)
    
    ret = np.zeros((x.shape[0], len(t)))
    
    for k in range(x.shape[0]):
        ret[k,x[k] - 1] += 1
    
    ret = np.cumsum(ret, axis = -1).astype(np.int32)
    ret /= (n - 1)      
          
    return ret

def to_unique(X):
    site_hist = dict()
    for k in range(X.shape[1]):
        x = X[:,k]
        #h = hashFor(x)
        h = ''.join(x.astype(str))
        if h in site_hist.keys():
            site_hist[h] += 1
        else:
            site_hist[h] = 1
        
    site_hist = {v: k for k, v in site_hist.items()}
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
    
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None", help = "npz")
    parser.add_argument("--weights", default = "None", help = "back projection weights")
    parser.add_argument("--ckpt", default = "/overflow/dschridelab/ddray/ghist/bottleneck/rnn_1e6/best.weights")

    parser.add_argument("--cdf", default = "cdf.pkl")
    parser.add_argument("--pkl", default = "None")

    parser.add_argument("--batch", default = 8, type = int)
    parser.add_argument("--latent", default = 41)
    parser.add_argument("--n_steps", default = "10000")
    parser.add_argument("--c_dim", default = 2)
    parser.add_argument("--lr", default = 0.001)
    
    parser.add_argument("--flow", default = "None")
    
    parser.add_argument("--size", default = 64)
    parser.add_argument("--f_size", default = 39)
    
    parser.add_argument("--c_min", default = "13000,")
    parser.add_argument("--c_max", default = ",500")
    parser.add_argument("--c_ix", default = "0,1")
    
    parser.add_argument("--n", default = 1, type = int)
    parser.add_argument("--L", default = 64, type = int)
    
    parser.add_argument("--n_transforms", default = 6, type = int)
    parser.add_argument("--n_components", default = 6, type = int)
    parser.add_argument("--s_dim", default = 42, type = int)
    parser.add_argument("--hidden_dim", default = 512, type = int)
    
    parser.add_argument("--odir", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def chunks(lst, n):
    _ = []
    
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])
        
    return _

import pandas as pd
    
def main():
    # configure MPI
    args = parse_args()
    device = torch.device('cuda')
    
    mean_y = np.array([(25000 + 250000) / 2., (5000 + 50000) / 2., (10 + 1000) / 2., -3.5])
    std_y = np.array([(25000 - 250000), (5000 - 50000), (10 - 1000), 3.]) * (np.sqrt(12) ** -1)
    
    mean_y = torch.FloatTensor(mean_y).unsqueeze(0).to(device)
    std_y = torch.FloatTensor(std_y).unsqueeze(0).to(device)
    
    model = TransformerConv(61, 4).to(device)
    
    if args.weights != "None":
        model.load_state_dict(torch.load(args.weights))
    
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor(np.array([40.])).to(device)) 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = float(args.lr))
    losses = deque(maxlen = 100)
    errors = deque(maxlen = 100)
    
    if args.pkl == "None":
        pkl = None
    else:
        pkl = args.pkl
    
    loader = NPZLoader(args.idir, batch_size = args.batch, pkl = pkl)
    pickle.dump([loader.ifiles, loader.ifiles_val], open(os.path.join(args.odir, 'train_val.pkl'), 'wb'))
    
    
    X_val, y_val = loader.get_val()
    print(len(X_val))
    
    min_loss = np.inf
    
    result = dict()
    result['epoch'] = []
    result['loss'] = []
    result['val_loss'] = []
    
    
    print('training...')
    for ij in range(int(args.n_steps)):
        model.train()
        optimizer.zero_grad()
        
        X, y = loader.get_batch()
        
        optimizer.zero_grad()
        
        X = torch.nn.utils.rnn.pad_sequence([torch.FloatTensor(u).to(device) for u in X], batch_first = True)
        X = X.transpose(1, 2)

        y = (torch.FloatTensor(np.array(y)).to(device) - mean_y) / std_y
        
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        y = (y * std_y + mean_y).detach().cpu().numpy()
        y_pred = (y_pred * std_y + mean_y).detach().cpu().numpy()
        
        errors.append(np.mean(np.sqrt(np.sum(((y - y_pred) / y) ** 2, -1))))
        
        if (ij + 1) % 100 == 0:
            model.eval()
            
            ii = chunks(list(range(len(X_val))), 128)
            
            result['epoch'].append(ij)
            result['loss'].append(np.mean(losses))
            
            Y = []
            Y_pred = []
            
            val_losses = []
            for ii_ in ii:
                y_ = torch.FloatTensor([y_val[u] for u in ii_]).to(device)
                if mean_y is not None:
                    y_ = (y_ - mean_y) / std_y
                
                X_ = torch.nn.utils.rnn.pad_sequence([torch.FloatTensor(X_val[u]).to(device) for u in ii_], batch_first = True)
                X_ = X_.transpose(1, 2)
                
                with torch.no_grad():
                    y_pred_ = model(X_)
                    loss = criterion(y_pred_, y_)
                    
                val_losses.append(loss.item())
                
            
                if mean_y is None:    
                    y = y_.detach().cpu().numpy()
                    y_pred = y_pred_.detach().cpu().numpy()
                else:
                    y = (y_ * std_y + mean_y).detach().cpu().numpy()
                    y_pred = (y_pred_ * std_y + mean_y).detach().cpu().numpy()
                    
                Y.extend(y)
                Y_pred.extend(y_pred)
            
            y = np.array(Y)
            y_pred = np.array(Y_pred)
            
            print(y.shape, y_pred.shape)
            result['val_loss'].append(np.mean(val_losses))
            
            if np.mean(val_losses) < min_loss:
                print('saving weights...')
                print('have validation loss of {}...'.format(np.mean(val_losses)))
                torch.save(model.state_dict(), 
                            os.path.join(args.odir, 'best.weights'))
                
                if mean_y is None:
                    fig, axes = plt.subplots(ncols = 4, nrows = 1, figsize = (32, 8))
                    
                    for k in range(4):
                        axes[k].plot(y[k,:])
                        axes[k].plot(y_pred[k,:])
                else:
                    fig, axes = plt.subplots(ncols = y.shape[1], nrows = 1, figsize = (y.shape[1] * 8, 8))
                
                    for k in range(y.shape[1]):
                        axes[k].scatter(y[:,k], y_pred[:,k])
                        axes[k].plot([np.min(y[:,k]), np.max(y[:,k])], [np.min(y[:,k]), np.max(y[:,k])])
                
                plt.savefig(os.path.join(args.odir, 'plot.png'), dpi = 100)
                plt.close()
                
                min_loss = np.mean(val_losses)
            
            df = pd.DataFrame(result)
            df.to_csv(os.path.join(args.odir, 'loss.csv'), index = False)
        
            if len(result['loss']) > 1:
                plt.plot(result['loss'], label = 'train')
                plt.plot(result['val_loss'], label = 'val')
                plt.legend()
                plt.savefig(os.path.join(args.odir, 'losses.png'), dpi = 100)
                plt.close()
            
        print('step {}: have mean loss of {}, mean error {}...'.format(ij, np.mean(losses), np.mean(errors)))            
        sys.stdout.flush()

if __name__ == '__main__':
    main()
        
        
