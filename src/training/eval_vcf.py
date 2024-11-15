import os
import argparse
import logging

import allel
import numpy as np

import sys
sys.path.append('popgenml/data')
sys.path.append('popgenml/models')

import torch

from layers import TransformerConv
from stats import to_unique
import matplotlib.pyplot as plt

import copy
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--window_size", default = "0.1")
    parser.add_argument("--weights", default = "None")
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

def main():
    args = parse_args()
    device = torch.device('cuda')
    
    w = float(args.window_size)
    callset = allel.read_vcf(args.ifile)
    gt = allel.GenotypeArray(callset['calldata/GT'])
    pos = callset['variants/POS']
    pos = pos.astype(np.float32)
    pos /= np.max(pos)
    
    mean_y = np.array([(25000 + 250000) / 2., (5000 + 50000) / 2., (10000 + 25000) / 2., (10 + 1000) / 2., -3.5])
    std_y = np.array([(25000 - 250000), (5000 - 50000), (10000 - 25000), (10 - 1000), 3.]) * (np.sqrt(12) ** -1)
    
    mean_y = torch.FloatTensor(mean_y).unsqueeze(0).to(device)
    std_y = torch.FloatTensor(std_y).unsqueeze(0).to(device)
    
    gt = gt.reshape(gt.shape[0], -1)
    gt[gt > 0] = 1
    
    model = TransformerConv(61, 5).to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    
    start = 0
    end = copy.copy(w)
    step = 0.05

    ii = np.where((pos >= start) & (pos < end))[0]
    
    Y = []
    while len(ii) > 0:
        x, _ = to_unique(gt[ii].T)
        x = x.T
        
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        y_pred = model(x) * std_y + mean_y
        
        y_pred = y_pred[0].detach().cpu().numpy()
        y_pred[-1] = 10 ** y_pred[-1]
        
        Y.append(y_pred)
        
        start += step
        end += step
        
        ii = np.where((pos >= start) & (pos < end))[0]
        
        if end > 1.:
            break
    
    Y = np.array(Y)
    
    print(Y)
    print(np.mean(Y, 0))

    # ${code_blocks}

if __name__ == '__main__':
    main()

