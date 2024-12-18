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
    
    parser.add_argument("--prior", default = "None")
    parser.add_argument("--cdf", default = "ckpt/cdf.pkl")
    parser.add_argument("--weights", default = "ckpt_popsize/best.weights")
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

from torchvision import utils

# well train a flow strapped with some neural network that embeds the site histogram as a conditional vector
# the flow is meant to generate the posterior tree distribution given the site histogram
def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    
    sim = class_for_name("simulators", args.sim)()
    
    if args.prior == "None":
        prior = None
    else:
        prior = args.prior
    cdf = pickle.load(open(args.cdf , 'rb'))
    
    loader = MSPrimeFWLoader(prior, sim, batch_size = 16, method = 'true', cdf = cdf['cdf'])
    
    print('getting batch...')
    real_img = loader.get_batch()
    real_img = (real_img.to(device).to(torch.float32) * 2 - 1)
    real_img = real_img[:16].detach().cpu()
    
    print('saving sample images...')
    utils.save_image(
        real_img,
        os.path.join("sample.png"),
        nrow=int(4),
        normalize=True,
        value_range=(-1, 1)
    )
    
    X, y = loader.get_median_replicate()
    
    model = resnet34(in_channels = 3, num_classes = args.latent).to(device)
    ckpt0 = torch.load(args.weights, map_location = device)
    model.load_state_dict(ckpt0)
    model.eval()
    
    plt.imshow(y[0].transpose(1,2,0))
    plt.show()
    
    y = torch.FloatTensor(y * 2 - 1).to(device)
    
    with torch.no_grad():
        y_proj = model(y)
    
    print(y_proj.shape)
    
if __name__ == '__main__':
    main()

