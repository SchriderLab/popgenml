# -*- coding: utf-8 -*-

import numpy as np
from skbio.tree import TreeNode

import tskit
import pickle
from scipy.interpolate import interp1d
import msprime
from relate import make_FW_rep
from io import BytesIO, StringIO

import copy
import numpy as np

from skbio import read
from skbio.tree import TreeNode

def tree_to_fw(tree, n_samples, diploid = False):
    sample_sizes = n_samples
    if diploid:
        sample_sizes = [2 * u for u in sample_sizes]
    
    f = StringIO(tree.as_newick())  
    root = read(f, format="newick", into=TreeNode)
    root.assign_ids()
            
    populations = [tree.population(u) for u in tree.postorder() if tree.is_leaf(u)]
    
    tips = [u for u in root.postorder() if u.is_tip()]
    for ix, t_ in enumerate(tips):
        t_.pop = populations[ix]
        
    children = root.children
    t = max([tree.time(u) for u in tree.postorder()])
    
    root.age = t
    
    while len(children) > 0:
        _ = []
        for c in children:
            
            c.age = c.parent.age - c.length
            if c.is_tip():
                c.age = 0.
            else:
                c.pop = -1
                
            _.extend(c.children)
            
        children = copy.copy(_)
    
    if len(n_samples) > 1:
        pop_vector = np.array(populations)
    else:
        pop_vector = None
    F, W, _, t_coal = make_FW_rep(root, sample_sizes)
    i, j = np.tril_indices(F.shape[0])
    F = F[i, j]

    return F, W, pop_vector, t_coal

"""
Converts (s, s, 3) redundant FW representation to TSKit tree.  Requires precomputed CDF of log inner coal times.
"""
class FWRep(object):
    def __init__(self, n_nodes, cdf = None, mu = 1e-2, mig = True):
        cdf = pickle.load(open(cdf, 'rb'))['cdf']
        cdf_ = interp1d(cdf.y, cdf.x)
        
        self.cdf = cdf_
        
        self.mig = mig
        
        self.n_nodes = n_nodes
        self.mu = mu
        
    def tree(self, im, pop = False):        
        F = im[:,:,0]
        F = (F + 1) / 2.
        F = F * self.n_nodes
        starts = np.array(range(2, self.n_nodes + 1))
        
        i, j = np.tril_indices(F.shape[0])
        F[j, i] = F[i, j]
        
        F[range(F.shape[0]), range(F.shape[0])] = starts
        F[range(1,F.shape[0]), range(F.shape[0] - 1)] = starts[:-1] - 1
        
        Fo = F.copy()
        
        counts = np.array(range(1, F.shape[0] - 1))
        for k in range(1, F.shape[0] - 1):
            counts = F[k,:k+1]
            
            options = []
            options.append(F[k,:k])
            
            if k > 2:
                df = np.array([F[k,u] - F[k,u - 1] for u in range(1, k)])
                unallowed = list(np.where(df == 0)[0] + 1)
            else:
                unallowed = []

            ii_c = np.where((counts > 0))[0][0]
            ii = [u for u in range(k) if (u >= ii_c)]
            ii = [u for u in ii if not (u in unallowed)]
            
            for ii_ in ii:
                o = F[k + 1,:k].copy()
                o[ii_:k] = F[k,ii_:k] - 1
                o[:ii_] = F[k,:ii_]
                
                options.append(o)
                
            options = np.array(options)
            err = np.mean((options - F[k+1,:k].reshape(1, -1)) ** 2, axis = 1)
            
            F[k+1,:k] = options[np.argmin(err),:]
            
        i, j = np.triu_indices(F.shape[0], 1)
        F[i, j] = F[j, i]
        
        i, j = np.triu_indices(F.shape[0])
        W = np.zeros(F.shape)
        
        W[j,i] = (im[i,j,1] + 1) / 2.
        if not pop:
            W2 = np.zeros(F.shape)
            W2[j,i] = ((im[i,j,2] + 1) / 2.) ** 2
            
            W = (W + W2) / 2.

        # inverse cdf and then back from log space
        # gives the intended inter-arrival time in generations
        W = self.cdf(np.clip(W, self.cdf.x[0], np.nanmax(self.cdf.x)))
        W = np.exp(W)
        
        i, j = np.triu_indices(F.shape[0], 1)
        W[i, j] = 0.
                
        diffs = []
        diffsT = []
        for k in range(W.shape[0]):
            if k < W.shape[0] - 1:
                w = W[k:,k] - W[k:,k + 1]
                
                diffs.append(np.mean(w))
            else:
                diffs.append(W[k, k])
                            
        for k in range(W.shape[0] - 1, -1, -1):
            if k > 0:
                w = W[k,:k] - W[k - 1,:k]
                
                diffsT.append(np.mean(w))
            else:
                diffsT.append(W[k, k])
   
        diffs = np.array(diffs)
        diffsT = np.array(diffsT)
        
        diffs = np.array([diffs, diffsT[::-1]])
        diffs = np.nanmean(diffs, axis = 0)

        diffs = np.abs(diffs)
        
        u1 = np.sum(diffs)
        t_coal = np.cumsum(diffs[::-1])[::-1]

        tables = tskit.TableCollection(sequence_length = 1000)
        node_table = tables.nodes  # set up an alias, for efficiency
        for k in range(self.n_nodes):
            node_table.add_row(flags = 1, population = 0, individual = k, time = 0.)
            
        for k in range(self.n_nodes, self.n_nodes * 2 - 1):
            node_table.add_row(flags = 0, population = 0, individual = -1, time = t_coal[::-1][k - self.n_nodes])
            
        
        edge_table = tables.edges
        individuals = tables.individuals
        pops = tables.populations
        
        for k in range(self.n_nodes):
            individuals.add_row(flags = 0)
        
        pops.add_row(metadata = b"{'description': '', 'name': 'pop_000'}")
        
        ages = np.zeros(2 * self.n_nodes - 1)
        ages[:len(t_coal)] = t_coal
        
        # add the root
        root = TreeNode(name = str(2 * self.n_nodes - 2))
        root.age = ages[0]
        root.id = 2 * self.n_nodes - 2
        
        # add the first child
        root.children = [TreeNode(name = str(2 * self.n_nodes - 3), parent = root)]
        root.children[0].id = 2 * self.n_nodes - 3
        root.children[0].age = ages[1]
        root.children[0].length = ages[0] - ages[1]
        
        edges = []
        edges.append((2 * self.n_nodes - 2, 2 * self.n_nodes - 3))
        
        nodes = [root, root.children[0]]
        for k in range(1, F.shape[0] - 1):
            df = F[k,:k + 1] - F[k + 1,:k + 1]

            ii = np.where((df > 0))[0]
            
            # add a child to the first
            ii = ii[0]
            
            iix = min([u.id for u in nodes]) - 1
            
            new_node = TreeNode(name = str(min([u.id for u in nodes]) - 1), parent = nodes[ii])
            new_node.id = min([u.id for u in nodes]) - 1
            new_node.age = ages[k + 1]
            new_node.length = nodes[ii].age - new_node.age
            nodes[ii].children.append(new_node)
            
            edges.append((nodes[ii].id, iix))
            
            nodes.append(new_node)
            
        counter = self.n_nodes - 1
        for node in sorted(nodes, key = lambda u: u.age):
            if len(node.children) < 2:
                for k in range(2 - len(node.children)):
                    new_node = TreeNode(name = str(counter), parent = node)
                    new_node.id = counter
                    new_node.age = 0.
                    new_node.length = node.age
                    node.children.append(new_node)
                    
                    edges.append((node.id, new_node.id))
                    
                    counter -= 1

        try:
            for e0, e1 in edges:
                edge_table.add_row(left = 0., right = 1000., parent = e0, child = e1)
            

            tables.sort()
            ts_tree = tables.tree_sequence()

            return t_coal, ts_tree, F, W
        except Exception as e:
            # debug?
            return None, None, None, None

