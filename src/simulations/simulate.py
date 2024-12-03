# -*- coding: utf-8 -*-
import os
import argparse
import logging

import sys
# patch until the package is installed
sys.path.append('popgenml/data')

from simulators import PopSplitSimulator, BottleNeckSimulator, SecondaryContactSimulator
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


import cairosvg
from PIL import Image
from io import BytesIO
import time
import msprime
from stats import to_unique
import networkx as nx
from scipy.interpolate import UnivariateSpline


from mpi4py import MPI

def plot_tree_graph(t, pop, edges):

    pop = np.array(pop)
    pop[pop == 2] = 1

    x = np.array(pop, dtype = np.float32) + np.random.normal(0., 0.1, pop.shape)
    xy = np.concatenate([np.log(t.reshape(-1, 1)), x.reshape(-1, 1)], 1)

    pos = dict(zip(range(t.shape[0]), xy))
    G = nx.from_edgelist(edges)
    
    G = G.subgraph([u for u in range(t.shape[0]) if t[u] != 0.])

    nx.draw(G, pos, node_size = 0.5)
    plt.show()
    
def to_cdf(t, bins):
    t_hist, _ = np.histogram(t, bins)
    
    t_hist = np.array(t_hist, dtype = np.float32)
    t_hist /= np.sum(t_hist)

    t_hist = np.cumsum(t_hist)
    
    return t_hist

def to_count(x, t, n = 40):    
    x = np.digitize(x, t)
    
    ret = np.zeros((x.shape[0], len(t)))
    
    for k in range(x.shape[0]):
        ret[k,x[k] - 1] += 1
    
    ret = np.cumsum(ret, axis = -1).astype(np.float32)
    ret /= (n - 1)

    ret = np.mean(ret, 0)      
          
    return ret
    
import copy
import itertools
from scipy.spatial.distance import pdist
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--model", default = "split")

    parser.add_argument("--n_replicates", default = 20, type = int)
    parser.add_argument("--L", default = "1e8")

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
    # configure MPI
    
    comm = MPI.COMM_WORLD
    args = parse_args()
    
    
    L = int(float(args.L))
    
    time_bins = np.linspace(4, 12, 1025)
    if args.model == "split":
        sim = PopSplitSimulator(L = L)
    
        for ix in range(comm.rank, args.n_replicates, comm.size):
            Nanc = np.random.uniform(50000, 150000)
            N0 = np.random.uniform(15000, 150000)
            N1 = np.random.uniform(15000, 150000)
            T = np.random.uniform(5000, 20000)
            
            print('simulating for: {}'.format([Nanc, N0, N1, T]))
            
            X, sites, ts = sim.simulate(Nanc, N0, N1, T)
            
            x, indices = to_unique(X)
            
            np.savez_compressed(os.path.join(args.odir, '{0:04d}.npz'.format(ix)), x = x.astype(np.uint8), ii = indices.astype(np.uint16), 
                                y = np.array([Nanc, N0, N1, T]))
    
    if args.model == "bottle":
        sim = BottleNeckSimulator(L = L)
        
        N1 = np.random.uniform(50000, 150000)
        N0 = N1 * np.random.uniform(0.03, 0.12)
        T = np.random.uniform(5000, 40000)
        
        X, sites, ts = sim.simulate(N0, N1, T)
        
        times = ts.nodes_time
        times = np.array(times, dtype = np.float32)
        ii = np.where(times != 0)[0]
        times = np.log(times[ii])
        
        times = to_cdf(times, time_bins)
        
    if args.model == "split_pop1":
        sim = BottleNeckSimulator(mu = 5.7e-9, r = 3.386e-9, L = L, n_samples = [22])
        
        for ix in range(comm.rank, args.n_replicates, comm.size):
            N1 = np.random.uniform(50000, 150000)
            N0 = np.random.uniform(15000, 150000)
            T = np.random.uniform(5000, 40000)
            
            X, sites, ts = sim.simulate(N0, N1, T)
            
            times = ts.nodes_time
            times = np.array(times, dtype = np.float32)
            ii = np.where(times != 0)[0]
            times = np.log(times[ii])
            
            times = to_cdf(times, time_bins)
            x, indices = to_unique(X)
            
            np.savez_compressed(os.path.join(args.odir, '{0:04d}.npz'.format(ix)), x = x.astype(np.uint8), ii = indices.astype(np.uint16), y1 = times,
                                y = np.array([N0, N1, T]))
            
    if args.model == "split_pop2":
        sim = BottleNeckSimulator(mu = 5.7e-9, r = 3.386e-9, L = L, n_samples = [18])
        
        for ix in range(comm.rank, args.n_replicates, comm.size):
            N1 = np.random.uniform(50000, 150000)
            N0 = np.random.uniform(15000, 150000)
            T = np.random.uniform(5000, 40000)
            
            X, sites, ts = sim.simulate(N0, N1, T)
            
            times = ts.nodes_time
            times = np.array(times, dtype = np.float32)
            ii = np.where(times != 0)[0]
            times = np.log(times[ii])
            
            times = to_cdf(times, time_bins)
            x, indices = to_unique(X)
            
            np.savez_compressed(os.path.join(args.odir, '{0:04d}.npz'.format(ix)), x = x.astype(np.uint8), ii = indices.astype(np.uint16), y1 = times,
                                y = np.array([N0, N1, T]))

        
    if args.model == "mig":
        sim = SecondaryContactSimulator(L = L)
        
        for ix in range(comm.rank, args.n_replicates, comm.size):
            # Set parameters based on specified ranges
            Nanc = np.random.uniform(12000, 120000)
            N_mainland = np.random.uniform(25000, 250000)
            N_island = np.random.uniform(5000, 50000)
            T_split = np.random.uniform(10000, 25000)
            T_contact = np.random.uniform(10, 1000)
            m = 10 ** np.random.uniform(-5, -2)
            
            X, sites, ts = sim.simulate(Nanc, N_mainland, N_island, T_split, T_contact, m)
            samples = list(range(sim.n_samples[0])) + list(range(sim.n_samples[0], sum(sim.n_samples)))
            samples = list(itertools.combinations(samples, 2))
            
            times = ts.nodes_time
            times = np.array(times, dtype = np.float32)
            ii = np.where(times != 0)[0]
            times = np.log(times[ii])
            
            times = to_cdf(times, time_bins)
            
            x, indices = to_unique(X)
            
            
            
            print(X.shape, x.shape)
            
            plt.imshow(x[:,:-1])
            plt.show()
            
            
            np.savez_compressed(os.path.join(args.odir, '{0:04d}.npz'.format(ix)), x = x.astype(np.uint8), ii = indices.astype(np.uint16), y1 = times,
                                y = np.array([Nanc, N_mainland, N_island, T_split, T_contact, m]))
        
        
        """
        sys.exit()
        
        
        F = np.array(F)
        W = np.array(W)
        
        Wim = np.zeros((79, 79))
        i, j = np.tril_indices(Wim.shape[0])
        
        
        F_ = F * W
        D = squareform(pdist(F_))
        D = D.sum(-1)
        
        ii = np.argmin(D)
        
        Wim[i, j] = np.log(W[ii])
        plt.imshow(Wim)
        plt.show()

        tree = ts.at_index(ii)
        ii = tree.timeasc()
        print(tree.time(ii[-1]))
        
        img_png = cairosvg.svg2png(tree.draw_svg(size = (1600, 800), y_axis=True, y_label=" "))
        img = Image.open(BytesIO(img_png))
        
        plt.imshow(img)
        plt.show()
        
        
        F = F[ii]
        W = W[ii]
        """
        
        
    # ${code_blocks}

if __name__ == '__main__':
    main()

