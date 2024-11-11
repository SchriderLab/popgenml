# -*- coding: utf-8 -*-
import os
import argparse
import logging

import sys
# patch until the package is installed
sys.path.append('popgenml/data')

from simulators import PopSplitSimulator
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
    args = parse_args()
    
    L = int(float(args.L))
    
    if args.model == "split":
        sim = PopSplitSimulator(L = L)
        
        for ix in range(args.n_replicates):
            Nanc = np.random.uniform(50000, 150000)
            N0 = np.random.uniform(60000, 180000)
            N1 = np.random.uniform(9000, 31000)
            T = np.random.uniform(5000, 20000)
            
            X, sites, ts = sim.simulate(Nanc, N0, N1, T)
            
            edges = np.array([ts.edges_parent, ts.edges_child]).T
            times = ts.nodes_time
            pop = ts.nodes_population
            x, indices = to_unique(X)
            
            print(X.shape, indices.shape)
    
            np.savez_compressed(os.path.join(args.odir, '{0:04d}.npz'.format(ix)), x = x, ii = indices, 
                                times = times, edges = edges, pop = pop.astype(np.uint8), y = np.array([Nanc, N0, N1, T]))
        

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

