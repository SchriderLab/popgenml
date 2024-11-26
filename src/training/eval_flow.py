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
from simulators import TwoPopMigrationSimulator
from data_loaders import MSPrimeFWLoader
from functions import to_unique, pad_sequences, relate

from collections import deque
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "matflow_relate")
    
    parser.add_argument("--latent", default = 64, type = int)
    parser.add_argument("--c_dim", default = 64, type = int)
    
    parser.add_argument("--prior", default = "priors/migration.csv")
    parser.add_argument("--cdf", default = "ckpt/cdf.pkl")
    parser.add_argument("--weights", default = "backproj_i1/best.weights")
    
    parser.add_argument("--batch", default = 16, type = int)
    parser.add_argument("--n_steps", default = 10000, type = int)
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

# well train a flow strapped with some neural network that embeds the site histogram as a conditional vector
# the flow is meant to generate the posterior tree distribution given the site histogram
def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    
    sim = TwoPopMigrationSimulator(L = int(1e4))
    cdf = pickle.load(open(args.cdf , 'rb'))
    
    loader = MSPrimeFWLoader(args.prior, sim, batch_size = 2, method = 'true', cdf = cdf['cdf'])
    
    mat_embedding = RNNEstimator(129, args.c_dim).to(device)
    mat_weights = torch.load(os.path.join(args.idir, 'mat.weights'), map_location = device)
    mat_embedding.load_state_dict(mat_weights)
    mat_embedding.eval()
    
    flow = zuko.flows.gaussianization.GF(args.latent, args.c_dim, transforms = 3, components = 8, hidden_features = (128, 128, 128), normalize = False).to(device)
    flow_weights = torch.load(os.path.join(args.idir, 'flow.weights'), map_location = device)
    flow.load_state_dict(flow_weights)
    flow.eval()
    
    generator = Generator(
        128, 64, args.n_mlp, channel_multiplier=args.channel_multiplier,
        n_channels = 3).to(device)
    
    generator_weights = torch.load(os.path.join(args.idir, 'gen.weights'), map_location = device)
    generator.load_state_dict(generator_weights["g_ema"])
    generator.eval()
    
    mu = 1e-6
    r = 1e-8
    N = 500
    L = int(1e4)

    cdf_inv = interp1d(cdf['cdf'].y, cdf['cdf'].x)    

    fig, axes = plt.subplots(ncols = 4)
    axes[0].set_ylabel('log generations')
    
    y_relate = []
    y = []
    
    y_coal = []
    y_coal_relate = []
    
    for k in range(32):
        _ = loader.get_replicate_(-1, True)
        while _ is None:
            _ = loader.get_replicate_(-1, True)
        im, Xmat, sites, _ = _
        im = np.array(im)
        coal_times_true = np.mean(cdf_inv(im[:,1,:,-1]), 0)
        
        im = im * 2 - 1
        i_, j_ = np.tril_indices(128)
    
        F_true = im[:,0,i_,j_]
    
        F_pred_relate, W, _, _, coal_times = relate(Xmat, sites, 129, mu, r, N, L)
        F_pred_relate /= np.max(F_pred_relate)        
                
        coal_times_relate = np.mean(np.log(coal_times[:,:-1]), 0)
        
        with torch.no_grad():
            X = torch.FloatTensor(Xmat.T).unsqueeze(0).to(device)
            
            c = mat_embedding(X)
            w = flow(c).sample((F_pred_relate.shape[0],))
            
            im = generator(w[:,0,:], input_is_latent = True)
            im = im.detach().cpu().numpy()
            
        W = im[:,1,:,-1]
        F_pred = (im[:,0,i_,j_] + 1) / 2.
        
        D = cdist(F_true, F_pred)
        i, j = linear_sum_assignment(D)
        D_ours = D[i, j].sum()
        
        D = cdist(F_true, F_pred_relate)
        i, j = linear_sum_assignment(D)
        D_relate = D[i, j].sum()
        
        y.append(D_ours)
        y_relate.append(D_relate)
        
        coal_times = np.mean(cdf_inv((W + 1) / 2.), 0)
                
        y_coal.append(np.linalg.norm(coal_times - coal_times_true))
        y_coal_relate.append(np.linalg.norm(coal_times_relate - coal_times_true))
        
        if k < 4:
            axes[k].plot(coal_times_true, label = 'true')
            axes[k].plot(coal_times, label = 'ours')
            axes[k].plot(coal_times_relate, label = 'relate')
        
    plt.title('coal time curves')
    plt.legend()
    plt.show()
    
    fig, axes = plt.subplots(ncols = 2)
    
    
    axes[0].scatter(y_relate, y)
    axes[0].set_xlabel('F error (relate)')
    axes[0].set_ylabel('F error (ours)')
    axes[0].plot([np.min(y_relate), np.max(y_relate)], [np.min(y_relate), np.max(y_relate)])
    
    axes[1].scatter(y_coal_relate, y_coal)
    axes[1].set_xlabel('coal error (relate)')
    axes[1].set_ylabel('coal error (ours)')
    axes[1].plot([np.min(y_coal_relate), np.max(y_coal_relate)], [np.min(y_coal_relate), np.max(y_coal_relate)])
    
    plt.show()
        
if __name__ == '__main__':
    main()
    

