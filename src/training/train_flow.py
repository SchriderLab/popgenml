# -*- coding: utf-8 -*-
import os
import argparse
import logging

import zuko
import torch

import sys
sys.path.append('popgenml/data')
sys.path.append('popgenml/models')

from torchvision_mod_layers import resnet34

from layers import RNNEstimator
from simulators import TwoPopMigrationSimulator
from data_loaders import MSPrimeFWLoader
from functions import to_unique, pad_sequences

from collections import deque
import pickle
import numpy as np

import matplotlib.pyplot as plt
from train_stylegan import class_for_name
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--latent", default = 64, type = int)
    parser.add_argument("--c_dim", default = 64, type = int)
    
    parser.add_argument("--prior", default = "priors/migration.csv")
    parser.add_argument("--cdf", default = "ckpt/cdf.pkl")
    parser.add_argument("--weights", default = "backproj_i1/best.weights")
    parser.add_argument("--sim", default = "StepStoneSimulator")
    
    parser.add_argument("--batch", default = 16, type = int)
    parser.add_argument("--n_steps", default = 10000, type = int)
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
    
    mat_embedding = RNNEstimator(129, args.c_dim).to(device)
    mat_embedding.train()
    
    flow = zuko.flows.gaussianization.GF(args.latent, args.c_dim, transforms = 3, components = 8, hidden_features = (128, 128, 128), normalize = False).to(device)
    flow.train()
    sim = class_for_name("simulators", args.sim)()
    
    if args.prior == "None":
        prior = None
    else:
        prio = args.prior
    cdf = pickle.load(open(args.cdf , 'rb'))
    
    loader = MSPrimeFWLoader(args.prior, sim, batch_size = 2, method = 'true', cdf = cdf['cdf'])
    
    optimizer = torch.optim.Adam([{'params' : flow.parameters()}, {'params' : mat_embedding.parameters()}], lr = 1e-3)
    
    model = resnet34(in_channels = 3, num_classes = args.latent).to(device)
    ckpt0 = torch.load(args.weights, map_location = device)
    model.load_state_dict(ckpt0)
    model.eval()
    
    losses = deque(maxlen = 100)
    
    min_loss = np.inf
    for ix in range(args.n_steps):
        X = []
        ims = []

        for ij in range(args.batch):
            _ = loader.get_replicate_(1, True)
            while _ is None:
                _ = loader.get_replicate_(1, True)
            w_, X_, _ = _
        
            X.append(X_.T)
            ims.append((w_ * 2 - 1)[0])
        
        X = pad_sequences(X)
        ims = np.array(ims)
        
        im = torch.FloatTensor(ims).to(device)
        with torch.no_grad():
            wT = model(im) 
            
        optimizer.zero_grad()
        
        X = torch.FloatTensor(X).to(device)    
        w = mat_embedding(X)
        
        loss = -flow(w).log_prob(wT).mean()
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()
 
        print('step {}: have mean loss of {}...'.format(ix, np.mean(losses)))
        sys.stdout.flush()
        
        if ix % 100 == 0:
            if np.mean(losses) < min_loss:
                print('saving weights...')
                min_loss = np.mean(losses)
                torch.save(flow.state_dict(), os.path.join(args.odir, 'flow.weights'))
                torch.save(mat_embedding.state_dict(), os.path.join(args.odir, 'mat.weights'))
        
        
    # ${code_blocks}

if __name__ == '__main__':
    main()


