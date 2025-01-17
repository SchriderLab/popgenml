# -*- coding: utf-8 -*-
import os
import argparse
import logging

import zuko
import torch

import sys
sys.path.append('popgenml/data')
sys.path.append('popgenml/models')

from io_ import write_to_ms

from torchvision_mod_layers import resnet34
from swagan_gray import Generator, Discriminator

from layers import RNNEstimator
from simulators import TwoPopMigrationSimulator, chebyshev_history, plot_size_history, step_stone_history
from data_loaders import MSPrimeFWLoader
from functions import to_unique, pad_sequences, relate, FWRep

from collections import deque
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from train_stylegan import class_for_name
from torch import nn

from scipy.signal import savgol_filter
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "ckpt_popsize")
    
    parser.add_argument("--latent", default = 64, type = int)
    parser.add_argument("--c_dim", default = 64, type = int)
    
    parser.add_argument("--prior", default = "None")
    parser.add_argument("--cdf", default = "ckpt_popsize/cdf.pkl")
    parser.add_argument("--weights", default = "ckpt_popsize/best.weights")
    
    parser.add_argument("--batch", default = 16, type = int)
    parser.add_argument("--n_steps", default = 10000, type = int)
    parser.add_argument("--sim", default = "StepStoneSimulator")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--n_mlp",
        type=int,
        default=8,
        help="dimensionality of the latent space",
    )
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

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def est_N(coal):
    nc = np.array(range(129, 1, -1))
    nc = (nc - 1) * nc / 2.
    nc = np.expand_dims(nc, 0)
  
    coal_times_ = np.concatenate([np.zeros((coal.shape[0], 1)), coal], 1)
    dt = np.diff(coal_times_, 1)
    dt = dt * nc
    
    n_bins = 9
    bins = np.percentile(coal.flatten(), np.linspace(0, 75, n_bins))
    
    est = np.zeros((n_bins - 1,))
    
    ii = np.digitize(coal, bins) - 1
    for u in range(n_bins - 1):
        
        i, j = np.where(ii == u)
        est[u] = dt[i, j].mean()
        
    Nt_est = list(zip(est, bins[:-1]))
    
    return Nt_est

# well train a flow strapped with some neural network that embeds the site histogram as a conditional vector
# the flow is meant to generate the posterior tree distribution given the site histogram
def main():
    args = parse_args()
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    
    sim = class_for_name("simulators", args.sim)(L = int(5e4))
    
    if args.prior == "None":
        prior = None
    else:
        prior = args.prior
    cdf = pickle.load(open(args.cdf , 'rb'))
    
    loader = MSPrimeFWLoader(prior, sim, batch_size = 2, method = 'true', cdf = cdf['cdf'])
    
    mat_embedding = RNNEstimator(130, args.c_dim).to(device)
    mat_weights = torch.load(os.path.join(args.idir, 'mat.weights'), map_location = device)
    mat_embedding.load_state_dict(mat_weights)
    mat_embedding.eval()
    
    flow = zuko.flows.gaussianization.GF(args.latent, args.c_dim, transforms = 3, components = 8, hidden_features = (256, 256, 256), normalize = False).to(device)
    flow_weights = torch.load(os.path.join(args.idir, 'flow.weights'), map_location = device)
    flow.load_state_dict(flow_weights)
    flow.eval()
    
    generator = Generator(
        128, 64, args.n_mlp, channel_multiplier=args.channel_multiplier,
        n_channels = 3).to(device)
    
    generator_weights = torch.load(os.path.join(args.idir, 'gen.weights'), map_location = device)
    generator.load_state_dict(generator_weights["g_ema"])
    generator.eval()
    
    mean_file = np.load(os.path.join(args.idir, 'mean_std.npz'))
    w_mean = mean_file['mean']
    w_std = mean_file['std']
    
    w_mean = torch.FloatTensor(w_mean).unsqueeze(0).to(device)
    w_std = torch.FloatTensor(w_std).unsqueeze(0).to(device)
    
    plt.plot(cdf['cdf'].y, cdf['cdf'].x)
    plt.show()
    
    mu = 1.26e-8 
    r = 1.007e-8
    L = int(2e5)
    
    for k in range(8):
        Nt = step_stone_history()
        
        
        Xmat, pos, ts = sim.simulate(Nt)
        tree = ts.first()
        
        coal_times = []
        
        while True:
            times = [tree.time(u) for u in tree.nodes()]
            times = [u for u in times if u > 0]
            times = sorted(times)
            coal_times.append(times)
            
            ret = tree.next()
            
            if not ret:
                break
        
        coal_times = np.array(coal_times)
        
        drop = torch.nn.Dropout(0.0)
        
        fw_rep = FWRep(129, args.cdf)
        
        im = []
        
        Xmat = to_unique(Xmat)
        for j in range(32):
            with torch.no_grad():
                X = torch.FloatTensor(Xmat).unsqueeze(0).to(device)
                
                c = mat_embedding(X, [Xmat.shape[0]], drop = drop) 
                
                w = flow(c).sample((4,))
                
                im_ = generator(w, input_is_latent = True)
                im.extend(im_.detach().cpu().numpy())

        im = np.array(im)
            
        coal_times_pred = []
        
        for j in range(im.shape[0]):
            coal_times_, ts_tree, F_pred, W = fw_rep.tree(im[j].transpose(1,2,0))
            coal_times_pred.append(coal_times_)
    
        coal_times_pred = np.array(coal_times_pred)
        coal_times_pred = coal_times_pred[:,::-1]
        
        
        Nt_est = est_N(coal_times)
        Nt_est_ = est_N(coal_times_pred)
        
        max_bin = max([u[1] for u in Nt_est] + [u[1] for u in Nt_est_])
        
        plot_size_history(Nt, max_t = max_bin)
        plot_size_history(Nt_est_, max_t = max_bin, color = 'r')
        plot_size_history(Nt_est, max_t = max_bin, color = 'b')
        plt.show()
    
if __name__ == '__main__':
    main()
    