# -*- coding: utf-8 -*-
import os
import argparse
import logging

from torchvision_mod_layers import resnet34

import argparse

import torch
import torch.nn.functional as F
import configparser
import torch.nn as nn

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel
from collections import deque

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
import copy
from torchvision import utils

import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    # ${args}
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--latent",
        type=int,
        default=128,
        help="dimensionality of the latent space",
    )
    parser.add_argument(
        "--n_mlp",
        type=int,
        default=8,
        help="dimensionality of the latent space",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--size", type=int, default=128, help="image sizes for the model"
    )
    parser.add_argument("--lr", default = "0.001")
    parser.add_argument("--weight_decay", default = "0.0")
    
    parser.add_argument("--n_steps", default = "200000")
    parser.add_argument("--ckpt", default = "None")
    
    parser.add_argument("--weights", default = "None")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--arch", default = "swagan")
    
    parser.add_argument("--c_dim", default = 2)
    
    parser.add_argument("--json_dir", default = "None")
    
    
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

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    # model = Classifier(config)
    
    from swagan_gray import Generator
    
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        n_channels = 3).to(device)

    ckpt = torch.load(args.ckpt, map_location = device)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()
    
    model = resnet34(in_channels = 3, num_classes = args.latent).to(device)
    if args.weights != "None":
        ckpt0 = torch.load(args.weights, map_location = device)
        model.load_state_dict(ckpt0)
    
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay = float(args.weight_decay))
    
    min_loss = np.inf
    criterion = nn.MSELoss()
    leaky = nn.LeakyReLU()
    
    os.system('mkdir -p {}'.format(os.path.join(args.odir, 'samples')))
    result = dict()
    result['step'] = []
    result['mse_loss'] = []
    result['backproj_error'] = []
    
    losses = deque(maxlen = 500)
    for ix in range(int(args.n_steps)):
        optimizer.zero_grad()
        
        noise_sample = torch.randn(args.batch, args.latent, device=device)
        
        l = generator.get_latent(noise_sample)
        
        imgs = generator(l, input_is_latent = True)
            
        l_pred = model(imgs)
    
        if not args.arch == "swagan":
            loss = criterion(l_pred, l[:,0,:])
        else:
            loss = criterion(l_pred, l)
    
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (ix + 1) % 10 == 0:
            logging.info('step {}, have loss of {}...'.format(ix + 1, np.mean(losses)))
            
        if (ix + 1) % 100 == 0:
            with torch.no_grad():
                imgs = generator(l, input_is_latent = True)
                l_pred = model(imgs)

                imgs_ = generator(l_pred, input_is_latent = True)
                
                utils.save_image(
                    (imgs + 1) / 2.,
                    os.path.join(os.path.join(args.odir, 'samples'), '{0:03d}.png'.format(ix)),
                    nrow=4,
                    normalize=True,
                )
                
                utils.save_image(
                    (imgs_ + 1) / 2.,
                    os.path.join(os.path.join(args.odir, 'samples'), '{0:03d}_backproj.png'.format(ix)),
                    nrow=4,
                    normalize=True,
                )
                
                back_proj_error = criterion(imgs, imgs_)
                result['step'].append(ix + 1)
                result['mse_loss'].append(np.mean(losses))
                result['backproj_error'].append(back_proj_error.item())
                logging.info('step {}, have backproj mse error of {}...'.format(ix + 1, result['backproj_error'][-1]))
            
            df = pd.DataFrame(result)
            df.to_csv(os.path.join(args.odir, 'metrics.csv'))
            
            logging.info('have loss of {}...'.format(np.mean(losses)))
            if np.mean(losses) < min_loss:
                min_loss = copy.copy(np.mean(losses))
                
                torch.save(model.state_dict(), os.path.join(args.odir, 'best.weights'))
                logging.info('saving weights...')
            losses = []
    
    # ${code_blocks}

if __name__ == '__main__':
    main()
