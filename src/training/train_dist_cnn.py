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
from simulators import TwoPopMigrationSimulator, chebyshev_history, plot_size_history, step_stone_history

from data_loaders import MSPrimeFWLoader
from functions import to_unique, pad_sequences

from collections import deque
import pickle
import numpy as np

import matplotlib.pyplot as plt
from train_stylegan import class_for_name
from swagan_gray import Generator, Discriminator


from scipy.spatial.distance import pdist, squareform
from seriate import seriate
from functions import to_unique, pad_sequences, relate, FWRep
from mpi4py import MPI

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def get_dist_matrix(x, metric = 'correlation'):
    _ = []
    D = squareform(pdist(x, metric = 'correlation'))
    #ii = seriate(D, timeout = 0.)
    _.append(D)
    
    _.append(squareform(pdist(x, metric = 'dice')))
    _.append(squareform(pdist(x, metric = 'yule')))
    
    return np.array(_)
    
    #
    #return np.expand_dims(D[np.ix_(ii, ii)], 0)

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
    parser.add_argument("--sim", default = "StepStoneSimulator")
    
    parser.add_argument("--batch", default = 12, type = int)
    parser.add_argument("--n_steps", default = 10000, type = int)
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--L", default = "5e5")
    
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
    # configure MPI
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using " + str(device) + " as device")
        
        model = resnet34(in_channels = 3, num_classes = args.latent).to(device)
        ckpt0 = torch.load(os.path.join(args.idir, 'best.weights'), map_location = device)
        model.load_state_dict(ckpt0)
        model.eval()
        
        generator = Generator(
            128, 64, args.n_mlp, channel_multiplier=args.channel_multiplier,
            n_channels = 3).to(device)
        
        generator_weights = torch.load(os.path.join(args.idir, 'gen.weights'), map_location = device)
        generator.load_state_dict(generator_weights["g_ema"])
        generator.eval()
    
        dist_model = resnet34(in_channels = 3, num_classes = args.latent).to(device)
        dist_model.load_state_dict(ckpt0)
        
        losses = deque(maxlen = 100)
        criterion = torch.nn.MSELoss()
        
        optimizer = torch.optim.Adam(dist_model.parameters(), lr = 1e-3)

        min_loss = np.inf
  
    sim = class_for_name("simulators", args.sim)(L = int(float(args.L)))
    
    if args.prior == "None":
        prior = None
    else:
        prior = args.prior
    cdf = pickle.load(open(os.path.join(args.idir, 'cdf.pkl') , 'rb'))
    
    loader = MSPrimeFWLoader(prior, sim, batch_size = 2, method = 'true', cdf = cdf['cdf'])
    fw_rep = FWRep(129, os.path.join(args.idir, 'cdf.pkl'))
    
    comm.Barrier()
    
    n_workers = comm.size - 1
    todo = int(args.batch) // (comm.size - 1)

    for ix in range(args.n_steps):
        
        if comm.rank != 0:
            X = []
            ims = []
    
            for ij in range(todo):
                X_, w_ = loader.get_median_replicate()
                
                X.append(get_dist_matrix(X_))
                ims.append((w_ * 2 - 1)[0])
                
            comm.send([X, ims], dest = 0)
        else:
            X = []
            ims = []
                        
            for k in range(n_workers):
                x, y_ = comm.recv(source = MPI.ANY_SOURCE)
                X.extend(x)
                ims.extend(y_)
                            
            dist_model.train()
            
            im = torch.FloatTensor(np.array(ims)).to(device)
            
                
            X = torch.FloatTensor(np.array(X)).to(device)
            optimizer.zero_grad()
            
            w_pred = dist_model(X)
            im_pred = generator(w_pred, input_is_latent = True)
            
            loss = criterion(torch.triu(im[:,1,:,:]), torch.triu(im_pred[:,1,:,:])) + criterion(torch.triu(im[:,2,:,:]), torch.triu(im_pred[:,2,:,:]))
            loss.backward()
            
            losses.append(loss.item())
            
            optimizer.step()
            print('step {}: have mean loss of {}...'.format(ix, np.mean(losses)))
            sys.stdout.flush()
            
            if ix % 100 == 0:
                if np.mean(losses) < min_loss:
                    print('saving weights...')
                    min_loss = np.mean(losses)
                    torch.save(dist_model.state_dict(), os.path.join(args.odir, 'best.weights'))
                
                    fig, axes = plt.subplots(nrows = 4, sharex = True, figsize = (4, 12))
                    for k in range(4):
                        Nt = step_stone_history()
                        
                        Xmat, pos, ts = sim.simulate(Nt)
                        X = torch.FloatTensor(get_dist_matrix(Xmat)).unsqueeze(0).to(device)
                        
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
                        
                        dist_model.eval()
                        
                        with torch.no_grad():
                            w_pred = dist_model(X)
                        
                            im = generator(w_pred, input_is_latent = True)
                
                        im = im.detach().cpu().numpy()
                        coal_times_pred, ts_tree, F_pred, W = fw_rep.tree(im[0].transpose(1,2,0))
                
                        axes[k].plot(np.log(np.mean(coal_times, 0)), label = 'gt')
                        axes[k].plot(np.log(coal_times_pred[::-1]), label = 'pred')
                    
                    axes[0].legend()
                        
                    plt.savefig(os.path.join(args.odir, 'best.png'), dpi = 100)
                    plt.close()
        
    # ${code_blocks}

if __name__ == '__main__':
    main()


