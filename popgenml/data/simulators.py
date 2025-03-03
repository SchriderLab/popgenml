# -*- coding: utf-8 -*-

from popgenml.data.relate import read_anc, RELATE_PATH, relate
from popgenml.data.viz import plot_demography
from popgenml.data.fw import tree_to_fw
from popgenml.data.io_ import read_slim

import msprime
import numpy as np
from io import BytesIO, StringIO

from skbio import read
from skbio.tree import TreeNode
import copy
import tempfile
import os
import glob

import logging
import matplotlib.pyplot as plt
import random
import subprocess

import matplotlib.pyplot as plt

RSCRIPT_PATH = os.path.join(os.getcwd(), 'include/relate/bin/RelateFileFormats')

import sys

from numpy.polynomial.chebyshev import Chebyshev
import tskit
import newick
import scipy
from scipy.stats import poisson, geom
from pkg_resources import resource_filename

def from_newick(
    string, *, min_edge_length=0, span=1, time_units=None, node_name_key=None
) -> tskit.TreeSequence:
    """
    Create a tree sequence representation of the specified newick string.

    The tree sequence will contain a single tree, as specified by the newick. All
    leaf nodes will be marked as samples (``tskit.NODE_IS_SAMPLE``). Newick names and
    comments will be written to the node metadata. This can be accessed using e.g.
    ``ts.node(0).metadata["name"]``.

    :param string string: Newick string
    :param float min_edge_length: Replace any edge length shorter than this value by this
        value. Unlike newick, tskit doesn't support zero or negative edge lengths, so
        setting this argument to a small value is necessary when importing trees with
        zero or negative lengths.
    :param float span: The span of the tree, and therefore the
        :attr:`~TreeSequence.sequence_length` of the returned tree sequence.
    :param str time_units: The value assigned to the :attr:`~TreeSequence.time_units`
        property of the resulting tree sequence. Default: ``None`` resulting in the
        time units taking the default of :attr:`tskit.TIME_UNITS_UNKNOWN`.
    :param str node_name_key: The metadata key used for the node names. If ``None``
        use the string ``"name"``, as in the example of accessing node metadata above.
        Default ``None``.
    :return: A tree sequence consisting of a single tree.
    """
    trees = newick.loads(string)
    if len(trees) > 1:
        raise ValueError("Only one tree can be imported from a newick string")
    if len(trees) == 0:
        raise ValueError("Newick string was empty")
    tree = trees[0]
    tables = tskit.TableCollection(span)
    if time_units is not None:
        tables.time_units = time_units
    if node_name_key is None:
        node_name_key = "name"
    nodes = tables.nodes
    nodes.metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "type": "object",
            "properties": {
                node_name_key: {
                    "type": ["string"],
                    "description": "Name from newick file",
                },
                "comment": {
                    "type": ["string"],
                    "description": "Comment from newick file",
                },
            },
        }
    )

    id_map = {}

    def get_or_add_node(newick_node, time):
        if newick_node not in id_map:
            flags = tskit.NODE_IS_SAMPLE if len(newick_node.descendants) == 0 else 0
            metadata = {}
            if newick_node.name:
                metadata[node_name_key] = newick_node.name
            if newick_node.comment:
                metadata["comment"] = newick_node.comment
            id_map[newick_node] = tables.nodes.add_row(
                flags=flags, time=time, metadata=metadata
            )
        return id_map[newick_node]

    root = next(tree.walk())
    get_or_add_node(root, 0)
    for newick_node in tree.walk():
        node_id = id_map[newick_node]
        for child in newick_node.descendants:
            length = max(child.length, min_edge_length)
            if length <= 0:
                raise ValueError(
                    "tskit tree sequences cannot contain edges with lengths"
                    " <= 0. Set min_edge_length to force lengths to a"
                    " minimum size"
                )
            child_node_id = get_or_add_node(child, nodes[node_id].time - length)
            tables.edges.add_row(0, span, node_id, child_node_id)
    # Rewrite node times to fit the tskit convention of zero at the youngest leaf
    nodes = tables.nodes.copy()
    youngest = min(tables.nodes.time)
    tables.nodes.clear()
    for node in nodes:
        tables.nodes.append(node.replace(time=node.time - youngest + root.length))
    tables.sort()
    return tables.tree_sequence()
        
"""
Base class to define the functionality for different pop size history priors.  These are used to sample population size trajectoies that can be approximated in msprime or another simulator
as piecewise constant
    __init__(N):
        N (float): 'central' estimate of the effective popsize (if N varies back in time then you might choose the harmonic mean of the effective popsize up to E[tmrca]?)

N defines the scale the prior uses.  Other than N, we define a prior only by a strictly positive random curve defined up to some set time which is multiplied by N to produce a history
over time in generations.  This class defaults to a constant pop size = N

Methods:
    sample(return_co = False): 
        returns tuple (list, list) of times and pop sizes (and optionally any coefficients used in the construction of the curve (array-like))

"""
class PiecewisePopSizePrior(object):
    def __init__(self, N):
        self.N = N

    def sample_curve(self):
        t = [0]
        N = np.ones(1)
        
        return t, N, None

    def sample(self, return_co = True):
        t, N, co = self.sample_curve()
        
        N *= self.N
        
        if return_co and (co is not None):
            return t, N, co
        else:
            return t, N

"""
"""
class ChebyshevHistory(PiecewisePopSizePrior):
    def __init__(self, N = 75000, max_K = 12, n_time_points = 128, max_eps = 0.9,
                 min_eps = 0.05, K_dist = 'geom',
                 params = {'mu' : 0.05}):
        super().__init__(N)
        
        self.max_K = max_K
        self.n_time_points = n_time_points
        self.max_eps = max_eps
        self.min_eps = min_eps
        
        if K_dist == 'geom':
            rv = geom(params['mu'])
        elif K_dist == 'poisson':
            rv = poisson(params['mu'])
            
        self.pmf = rv.pmf(np.array(range(1, self.max_K), dtype = np.float32))
        self.pmf /= np.sum(self.pmf)
        
    def sample_curve(self):
        # sample the number of Cheby polynomials to include
        K = self.max_K - 1
        
        # sample ~ N(0, 1) (standard gaussian) coefficients
        co = np.random.normal(0., 1., K + 1)
        
        p = Chebyshev(co)
        
        x = np.linspace(-1., 1., self. n_time_points)
        
        # get the curve with range (-1, 1)
        y = p(x)
        
        max_p = np.max(y)
        min_p = np.min(y)
        
        y = ((y - min_p) / (max_p - min_p)) * 2 - 1
        
        eps = np.random.uniform(self.min_eps, self.max_eps)
        
        N = y * eps + 1
        
        co = np.concatenate([np.array([eps]), np.pad(co, ((0, self.max_K - co.shape[0])))])

        t = [0] + list(np.exp(np.linspace(0, 11, self.n_time_points - 1)))

        return t, N, co
    
"""
Base class to define some the attributes common to the simulators in 'include'.  These are:
    L (int): the length of the simulated chromosome in base pairs
    mu (float): the mutation rate
    r: (float): the recombination rate
    n_samples (list of int): the number of samples taken for each population
"""
class BaseSimulator(object):
    def __init__(self, L, mu, r, n_samples):
        self.L = L
        self.mu = mu
        self.r = r        
        self.n_samples = n_samples

class MSModSimulator(object):
    def __init__(self, prior = None, L = int(1e4), mu = 5.0e-9, n_samples = [64, 64]):
        self.L = L
        self.mu = mu
        
    

"""
"""
class SlimSimulator(object):
    """
    script (str): points to a slim script
    args (str): format string for slim command with args if needed
    """
    def __init__(self, script = os.path.join(resource_filename('popgenml', 'slim'), 'introg_bidirectional.slim'),
                 args = "-d sampleSizePerSubpop={} -d donorPop={} -d st={} -d mt={}", 
                 n_samples = 64, L = int(1e4)):
        self.script = script
        self.args = args
        self.n_samples = n_samples
        self.L = L
        
    def simulate(self, *args):
        args = args + (self.script,)
        
        seed = random.randint(0, 2**32-1)
        slim_cmd = "slim -seed {} -d physLen={} ".format(seed, self.L)

        if self.args is not None:
            slim_cmd += self.args.format(*args)
            slim_cmd += " {}".format(self.script)    
        else:
            slim_cmd += "{}".format(self.script)

        procOut = subprocess.Popen(
            slim_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, err = procOut.communicate()
                    
        X, pos, y_ = read_slim(output, self.n_samples, self.L)
        pos = np.array(pos)
        X = np.array(X)
        
        y = np.zeros(X.shape)
        
        for ix, start_end in enumerate(y_):
            if len(start_end) == 0:
                continue
            
            for start,end in start_end:
            
                ii = np.where((pos >= start) & (pos <= end))[0]
                
                y[ix, ii] = 1.
        
        return X, pos, y

"""
"""        
class BaseMSPrimeSimulator(BaseSimulator):
    # L is the size of the simulation in base pairs
    # specify mutation rate
    
    # immutable properties of the simulator are defined here:
    # L: the size of the simulation in base pairs
    # mu: mutation rate
    # r: recombination rate
    # whether or not diploid individuals are simulated (vs haploid)
    # the number of samples
    def __init__(self, L = int(1e5), mu = 1.26e-8, r = 1.007e-8, ploidy = 1, 
                 n_samples = [129], N = 75000):
        self.L = L
        self.mu = mu
        self.r = r
        self.n_samples = n_samples
        
        self.ploidy = ploidy
        
        self.sample_size = sum(n_samples)

        self.rcmd = 'cd {3} && ' + RSCRIPT_PATH + ' --mode ConvertFromVcf --haps {0} --sample {1} -i {2}'
        self.relate_cmd = 'cd {6} && ' + RELATE_PATH + ' --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'
        
        self.co = None
        self.demography = None
        
        self.N = N
        
        return
    
    # here's where you should define how to use the prior to create the msprime demography
    # must be over-written in subclass
    def make_demography(self):
        demography = msprime.Demography()
        
        return demography
    
    def simulate(self, *args, verbose = False):
        # get the current logging level
        logger = logging.getLogger(__name__)
        current_level = logger.getEffectiveLevel()
        logging.basicConfig(level = logging.ERROR)
        """
        if not verbose:
            # to quiet down msprime
            logging.basicConfig(level = logging.ERROR, force = True)
        else:
            logging.basicConfig(level = logging.INFO, force = True)
        """
        self.demography = self.make_demography()
        
        # simulate ancestry
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = sum(self.n_samples),
            sequence_length = self.L,
            recombination_rate=self.r,
            ploidy = self.ploidy,
            demography = self.demography,
        )
        
        # set the logging level back to what it was before the call to msprime
        #logging.basicConfig(level = current_level, force = True)
        
        return self.mutate_and_return_(ts)
    
    def mutate_and_return_(self, ts):
        result = dict()
        
        # simulate mutations, binary discrete model
        mutated_ts = msprime.sim_mutations(ts, rate=self.mu, model=msprime.BinaryMutationModel())
        
        X = mutated_ts.genotype_matrix()
        X[X > 1] = 1
        X = X.T

        sites = [u.position for u in list(mutated_ts.sites())]
        sites = np.array(sites) / self.L
        
        result['x'] = X
        result['pos'] = sites
        result['ts'] = ts
        
        return result
        
    def simulate_fw_single(self, *args):
        result = self.simulate(*args)
        s = result['ts']
        
        sample_sizes = self.n_samples
        if self.ploidy == 2:
            sample_sizes = [2 * u for u in sample_sizes]
        
        ii = np.random.choice(range(s.num_trees))
        
        tree = s.at(ii)
        
        # convert tree to encoding
        F, W, pop_vector, t_coal = tree_to_fw(tree, self.n_samples, (self.ploidy == 2))
        
        result['F'] = F
        result['W'] = W
        result['t_coal'] = t_coal
        
        return result
    
    def simulate_fw_sequential_pair(self, *args):
        result = self.simulate(*args)
        s = result['ts']
    
        sample_sizes = self.n_samples
        
        if self.ploidy == 2:
            sample_sizes = [2 * u for u in sample_sizes]
        
        ii = np.random.choice(range(s.num_trees - 1))
        
        tree = s.at(ii)
        
        Fs = []
        Ws = []
        t_coals = []
        
        # convert tree to encoding
        F, W, pop_vector, t_coal = tree_to_fw(tree, self.n_samples, (self.ploidy == 2))
        
        Fs.append(F)
        Ws.append(W)
        t_coals.append(t_coal)
        
        tree = s.at(ii + 1)
        
        # convert tree to encoding
        F, W, pop_vector, t_coal = tree_to_fw(tree, self.n_samples, (self.ploidy == 2))
        
        Fs.append(F)
        Ws.append(W)
        t_coals.append(t_coal)
        
        result['F'] = Fs
        result['W'] = Ws
        result['t_coal'] = t_coals
        
        return result
    
    def simulate_relate(self, *args):
        result = self.simulate(*args)
        s = result['ts']
        
        Fs, Ws, _, _, coal_times = relate(result['x'], result['pos'], sum(self.n_samples), self.mu, self.r, self.N, self.L, 
                                          self.ploidy == 2)
        
        result['F'] = Fs
        result['W'] = Ws
        result['t_coal'] = coal_times
        
        return result
    
    # returns FW image(s)
    def simulate_fw(self, *args, method = 'true', sample = False, sample_prob = 0.01):
        result = self.simulate(*args)
        s = result['ts']
        
        Fs = []
        Ws = []
        pop_vectors = []
        coal_times = []
        
        tree = s.first()
        ret = True
        # should be an iteration here but need to be careful in general due to RAM
        while ret:
            if sample:
                if np.random.uniform() > sample_prob:
                    ret = tree.next()
                    continue
        
            F, W, pop_vector, t_coal = tree_to_fw(tree, self.n_samples, self.ploidy == 2)
            
            Fs.append(F)
            Ws.append(W)
            if len(self.n_samples) > 1:
                pop_vectors.append(pop_vector)
            else:
                pop_vectors.append(None)
            coal_times.append(t_coal)
            
            ret = tree.next()
                
        result['F'] = Fs
        result['W'] = Ws
        result['pop'] = pop_vectors
        result['t_coal'] = coal_times
        
        return result
        
class SimpleCoal(BaseMSPrimeSimulator):
    def __init__(self, N = 75000, **kwargs):
        super().__init__(**kwargs)
        
        self.N = N
        
    def make_demography(self):
        demography = msprime.Demography()
        
        demography.add_population(name="A", initial_size=self.N)
        
        return demography
            
class StepStoneSimulator(BaseMSPrimeSimulator):
    def __init__(self, prior = ChebyshevHistory(), **kwargs):
        super().__init__(**kwargs)
        
        self.prior = prior
        
    def make_demography(self):
        demography = msprime.Demography()
        
        t, N, co = self.prior.sample()
        self.co = co
        
        Nt = list(zip(N, t))
        
        N0, _ = Nt[0]
        demography.add_population(name="A", initial_size=N0)
        
        for N1, T in Nt[1:]:
            demography.add_population_parameters_change(time=T, initial_size=N1)
            
        return demography    
            
