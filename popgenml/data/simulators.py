# -*- coding: utf-8 -*-

import msprime
import numpy as np
from functions import make_FW_rep, read_anc, read_slim
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

# to quiet down msprime
logging.basicConfig(level = logging.ERROR)

RELATE_PATH = os.path.join(os.getcwd(), 'include/relate/bin/Relate')
RSCRIPT_PATH = os.path.join(os.getcwd(), 'include/relate/bin/RelateFileFormats')

import sys

from numpy.polynomial.chebyshev import Chebyshev
import tskit
import newick

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


def plot_size_history(Nt, max_t = None, color = 'k'):
    N = [np.log10(u[0]) for u in Nt]
    t = [u[1] for u in Nt]

    for k in range(len(N) - 1):
        plt.plot([t[k], t[k + 1]], [N[k], N[k]], c = color)
        plt.plot([t[k + 1], t[k + 1]], [N[k], N[k + 1]], c = color)

    if max_t:
        plt.plot([t[-1], max_t], [N[-1], N[-1]], c = color)
    
# min and max size in log10 scale
def chebyshev_history(min_size = 3, max_size = 5, max_K = 12, n_time_points = 2048, max_time = 1e6):
    mean_log_size = np.random.uniform(min_size, max_size)

    w = np.random.uniform(0., 1.) # up to a factor of 10 size change

    #k = np.random.choice(range(0, max_K + 1))
    #co = np.random.normal(0., 1., k)
    co = np.random.normal(0., 1., max_K + 1)
    co *= np.random.choice([0., 1.], max_K + 1)
    
    t = np.linspace(0., 1., n_time_points) * (10 ** mean_log_size) * 4
    
    p = Chebyshev(co)
    x = np.linspace(-1., 1., n_time_points)
    
    y = p(x)
    
    y -= np.mean(y)
    if np.var(y) != 0:
        y /= (np.max(y) - np.min(y))
        
    ret = y * w + mean_log_size
    
    ii = [0] + list(np.sort(np.random.choice(range(1, len(ret)), np.random.choice(range(8)), replace = False)))
    
    return t[ii], ret[ii]

def step_stone_history(min_size = 4, max_size = np.log10(1.5e5), max_breaks = 4):
    n_breaks = np.random.choice(range(max_breaks + 1))
    
    initial_size = 10 ** np.random.uniform(min_size, max_size)
    
    # how should we choose the knots for the simulator?
    # we'll say that breaks happen within 10 and 1000 generations and are uniform within that
    N = [initial_size]
    t = [0]
    
    Ns = list(10 ** np.random.uniform(min_size, max_size, n_breaks))
    ts = sorted(list(np.random.uniform(100, 1000, n_breaks)))
    
    N = N + Ns
    t = t + ts
    
    Nt = list(zip(N, t))
    
    return Nt
        
    
    
    
class DiscoalSimulator(object):
    def __init__(self, L = int(1e5), mu = 1.26e-8, r = 1.007e-8, diploid = False, n_samples = [129]):
        self.L = L
        self.mu = mu
        self.r = r
        self.diploid = diploid
        self.n_samples = n_samples
        
    def simulate(self, Nt = None):
        pop_size_str = ''
        if Nt is None:
            t, N = chebyshev_history()
            
            N = 10 ** N
            
            Nt = list(zip(N, t))
                        
        N0 = Nt[0][0]
        for (N, t) in Nt[1:]:
            _ = ' -en {0} 0 {1}'.format(t / (4 * N0), N / N0)            
            pop_size_str += _
            
        N = Nt[0][0]
        
        s = 10 ** np.random.uniform(-4, -2)
        a = 2 * N * s
        
        cmd = 'include/discoal/discoal {0} 1 20000 -t {1} -r {2} -T' + pop_size_str + '-ws 0 -Pf 0.0 0.05 -Pc 0.5 1.0 -Pu 0.0 0.01 -a {} -x {}'.format(a, np.random.uniform(0.05, 0.95))
        theta = 4 * N * self.mu * self.L
        rho = 4 * N * self.r * self.L
        
        cmd_ = cmd.format(self.n_samples[0], theta, rho)
        
        procOut = subprocess.Popen(
            cmd_.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, err = procOut.communicate()
        output = output.decode('utf-8')
        
        output = output.split('\n')
        ns = [u for u in output if (('[' in u) and (']' in u))]
        
        trees = [from_newick(u) for u in ns]
        
        
        
        
class BaseSimulator(object):
    # L is the size of the simulation in base pairs
    # specify mutation rate
    
    # immutable properties of the simulator are defined here:
    # L: the size of the simulation in base pairs
    # mu: mutation rate
    # r: recombination rate
    # whether or not diploid individuals are simulated (vs haploid)
    # the number of samples
    def __init__(self, L = int(1e5), mu = 1.26e-8, r = 1.007e-8, diploid = True, n_samples = [40], N = 500):
        self.L = L
        self.mu = mu
        self.r = r
        self.diploid = diploid
        
        self.N = N
        
        self.n_samples = n_samples
        self.sample_size = sum(n_samples)
        
        self.relate_path = os.path.join(os.getcwd(), 'include/relate/bin/Relate')
        rscript_path = os.path.join(os.getcwd(), 'include/relate/bin/RelateFileFormats')

        self.rcmd = 'cd {3} && ' + rscript_path + ' --mode ConvertFromVcf --haps {0} --sample {1} -i {2}'
        self.relate_cmd = 'cd {6} && ' + self.relate_path + ' --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'
        
        return
    
    # should return X, sites
    # where X is a binary matrix and sites is an array with positions corresponding to the second axis
    def simulate(self, *args):
        return
    
    def mutate_and_return_(self, ts):
        # simulate mutations, binary discrete model
        mutated_ts = msprime.sim_mutations(ts, rate=self.mu, model=msprime.BinaryMutationModel())
        
        X = mutated_ts.genotype_matrix()
        X[X > 1] = 1
        X = X.T

        sites = [u.position for u in list(mutated_ts.sites())]
        sites = np.array(sites) / self.L
        
        return X, sites, mutated_ts
    
    def simulate_fw_single(self, *args):
        X, sites, s = self.simulate(*args)
        
        sample_sizes = self.n_samples
        if self.diploid:
            sample_sizes = [2 * u for u in sample_sizes]
        
        ii = np.random.choice(range(s.num_trees))
        
        tree = s.at(ii)
        
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
        
        if len(self.n_samples) > 1:
            pop_vector = np.array(populations)
        else:
            pop_vector = None
        F, W, _, t_coal = make_FW_rep(root, sample_sizes)
        i, j = np.tril_indices(F.shape[0])
        F = F[i, j]
        
        return F, W, pop_vector, t_coal, X, sites, s
       
    
    # returns FW image(s)
    def simulate_fw(self, *args, method = 'true'):
        print(args)
        X, sites, s = self.simulate(*args)
        
        sample_sizes = self.n_samples
        if self.diploid:
            sample_sizes = [2 * u for u in sample_sizes]
        
        Fs = []
        Ws = []
        pop_vectors = []
        coal_times = []
        
        # return the ground truth tree sequence as a sequence of as a sequence of F and W condensed matrices
        if method == 'true':

            tree = s.first()
            ret = True
            # should be an iteration here but need to be careful in general due to RAM
            while ret:

            
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
        
                pop_vector = np.array(populations)
                F, W, _, t_coal = make_FW_rep(root, sample_sizes)
                i, j = np.tril_indices(F.shape[0])
                F = F[i, j]
                
                Fs.append(F)
                Ws.append(W)
                if len(self.n_samples) > 1:
                    pop_vectors.append(pop_vector)
                else:
                    pop_vectors.append(None)
                coal_times.append(t_coal)
                
                ret = tree.next()
                
        # return the inferred tree sequence from Relate as a sequence of F and W condensed matrices
        elif method == 'relate':
            Fs = []
            Ws = []
            pop_vectors = []
            coal_times = []
            
            n_samples = sum(self.n_samples)
         
            if self.diploid:
                n_samples = n_samples // 2
            
            temp_dir = tempfile.TemporaryDirectory()
            
            odir = os.path.join(temp_dir.name, 'relate')
            os.mkdir(odir)
            
            ms_file = os.path.join(temp_dir.name, 'sim.vcf')
            
            try:
                f = open(os.path.join(temp_dir.name, 'sim.vcf'), 'w')
                s.write_vcf(f)
                f.close()
            except:
                return None
            
            tag = ms_file.split('/')[-1].split('.')[0]
            cmd_ = self.rcmd.format('sim.haps', 'sim.sample', '../sim', odir) + ' >/dev/null 2>&1'
            os.system(cmd_)
            
            map_file = ms_file.replace('.vcf', '.map')
            
            ofile = open(map_file, 'w')
            ofile.write('pos COMBINED_rate Genetic_Map\n')
            ofile.write('0 {} 0\n'.format(self.r * self.L))
            ofile.write('{0} {1} {2}\n'.format(self.L, self.r * self.L, self.r * 10**8))
            ofile.close()
            
            
            haps = list(map(os.path.abspath, sorted(glob.glob(os.path.join(odir, '*.haps')))))
            samples = list(map(os.path.abspath, [u.replace('.haps', '.sample') for u in haps if os.path.exists(u.replace('.haps', '.sample'))]))
            
            # we need to rewrite the haps files (for haploid organisms)
            for sample in samples:
                f = open(sample, 'w')
                f.write('ID_1 ID_2 missing\n')
                f.write('0    0    0\n')
                if self.diploid:
                    for k in range(n_samples):
                        f.write('UNR{} UNR{} 0\n'.format(k + 1, k + 1))
                else:
                    for k in range(n_samples):
                        f.write('UNR{} NA 0\n'.format(k + 1))
                
            f.close()
            
            ofile = haps[0].split('/')[-1].replace('.haps', '') + '_' + map_file.split('/')[-1].replace('.map', '').replace(tag, '').replace('.', '')
            if ofile[-1] == '_':
                ofile = ofile[:-1]

            cmd_ = self.relate_cmd.format(self.mu, 2 * self.N, haps[0], 
                                         samples[0], os.path.abspath(map_file), 
                                         ofile, odir) + ' >/dev/null 2>&1'

            
            os.system(cmd_)
            
            try:
                if len(self.n_samples) == 1:
                    _ = self.n_samples + [0]
                else:
                    _ = self.n_samples
                
                anc_file = os.path.join(odir, '{}.anc'.format(ofile))
                Fs, Ws, snps, pop_vectors, coal_times = read_anc(anc_file, _)
            except:
                return None
            
            temp_dir.cleanup()
            
        return Fs, Ws, pop_vectors, coal_times, X, sites, s
    
class SlimSimulator(object):
    """
    script (str): points to a slim script
    args (str): format string for slim command with args if needed
    """
    def __init__(self, script, args = None, n_samples = 64, L = int(1e4)):
        self.script = script
        self.args = args
        self.n_samples = n_samples
        self.L = L
        
    def simulate(self, *args):
        args = args + (self.script,)
        
        seed = random.randint(0, 2**32-1)
        slim_cmd = "include/SLiM/build/slim -seed {} -d physLen={} ".format(seed, self.L)

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
    
    
class StepStoneSimulator(BaseSimulator):
    def __init__(self, L = int(1e4), mu = 1.26e-8, r = 1.007e-8, diploid = False, n_samples = [129]):
        super().__init__(L, mu, r, diploid, n_samples)
        
    # built in prior for this one
    def simulate(self, Nt = None):
        n_samples = self.sample_size
        
        demography = msprime.Demography()
        if Nt is None:
            Nt = step_stone_history()
     
        N0, _ = Nt[0]
        demography.add_population(name="A", initial_size=N0)
        
        for N1, T in Nt[1:]:
            demography.add_population_parameters_change(time=T, initial_size=N1)
        
        # simulate ancestry
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = sum(self.n_samples),
            sequence_length=self.L,
            recombination_rate=self.r,
            ploidy = 1,
            #mutation_rate=mutation_rate,
            demography=demography,
            #Ne=population_size
        )
        
        return self.mutate_and_return_(ts)
    
class BottleNeckSimulator(BaseSimulator):
    def __init__(self, L = int(1e6), mu = 1.26e-8, r = 1.007e-8, diploid = True, n_samples = [129]):
        super().__init__(L, mu, r, diploid, n_samples)
        
    # population size up to T = N0
    # population size after T = N1
    # time of bottleneck = T
    def simulate(self, N0, N1, T):
        n_samples = self.sample_size
        
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=N0)
        demography.add_population_parameters_change(time=T, initial_size=N1)
        
        # simulate ancestry
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = sum(self.n_samples),
            sequence_length=self.L,
            recombination_rate=self.r,
            ploidy = 1,
            #mutation_rate=mutation_rate,
            demography=demography,
            #Ne=population_size
        )
        
        return self.mutate_and_return_(ts)
    
class SimpleCoal(BaseSimulator):
    def __init__(self, L = int(1e4), mu = 1e-6, r = 0., diploid = False, n_samples = [129]):
        super().__init__(L, mu, r, diploid, n_samples)

    # here we specify recombination rate in the simulation command
    def simulate(self, N = 1000, r = 0.):
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=N)
        
        # simulate ancestry
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = sum(self.n_samples),
            sequence_length=self.L,
            recombination_rate = r,
            
            #mutation_rate=mutation_rate,
            demography=demography,
            ploidy = 1,
            #Ne=population_size
        )
        
        return self.mutate_and_return_(ts)
    
class PopSplitSimulator(BaseSimulator):
    def __init__(self, L = int(1e8), mu = 5.7e-9, r = 3.386e-9, diploid = True, n_samples = [22, 18]):
        super().__init__(L, mu, r, diploid, n_samples)
        
    def simulate(self, Nanc, N0, N1, split_time):
        samples = []
        samples.append(msprime.SampleSet(self.n_samples[0], population = 'A', ploidy = 2))
        samples.append(msprime.SampleSet(self.n_samples[1], population = 'B', ploidy = 2))
        
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size = N0, growth_rate = 0.)
        demography.add_population(name="B", initial_size = N1, growth_rate = 0.)
        demography.add_population(name="C", initial_size = Nanc, growth_rate = 0.)
        demography.add_population_split(time = split_time, derived = ["A", "B"], ancestral = "C")
        
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = samples,
            #additional_nodes=(msprime.NodeType.RECOMBINANT),
            sequence_length=self.L,
            recombination_rate=self.r,
            #mutation_rate=mutation_rate,
            demography=demography,
            #coalescing_segments_only=False
            #Ne=population_size
        )
    
        return self.mutate_and_return_(ts)
    
class TwoPopMigrationSimulator(BaseSimulator):
    def __init__(self, L = int(1e6), mu = 1e-6, r = 1e-8, diploid = False, n_samples = [65, 64]):
        super().__init__(L, mu, r, diploid, n_samples)
        
    # constant size for two popuations and migration coefficient
    def simulate(self, N0, N1, Nanc, m01, split_time):
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size = N0, growth_rate = 0.)
        demography.add_population(name="B", initial_size = N1, growth_rate = 0.)
        demography.add_population(name="C", initial_size = Nanc, growth_rate = 0.)
        demography.add_population_split(time = split_time, derived = ["A", "B"], ancestral = "C")
        
        demography.add_migration_rate_change(0., rate = m01, source = "A", dest = "B")
        demography.sort_events()
        
        samples = []
        samples.append(msprime.SampleSet(self.n_samples[0], population = 'A', ploidy = 1))
        samples.append(msprime.SampleSet(self.n_samples[1], population = 'B', ploidy = 1))
        # simulate ancestry
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = samples,

            sequence_length=self.L,
            recombination_rate=self.r,
            #mutation_rate=mutation_rate,
            demography=demography,
            #Ne=population_size
            ploidy = 1
        )
        
        return self.mutate_and_return_(ts)
    
class SecondaryContactSimulator(BaseSimulator):
    def __init__(self, L = int(1e6), mu = 5.4e-9, r = 3.386e-9, diploid = True, n_samples = [22, 8]):
        super().__init__(L, mu, r, diploid, n_samples)
    
    
    def simulate(self, Nanc, N_mainland, N_island, T_split, T_contact, m):
        """
        Simulates a population split with initial isolation, followed by secondary contact with migration.
    
        Parameters:
            Nanc (float): Ancestral population size.
            N_mainland (float): Mainland population size after split.
            N_island (float): Island population size after split.
            T_split (float): Time of initial population split in generations.
            T_contact (float): Time of secondary contact and migration onset.
            m (float): Migration rate between island and mainland populations.
            mutation_rate (float): Mutation rate per base per generation.
            recombination_rate (float): Recombination rate per base per generation.
            sequence_length (float): Length of the simulated sequence.
    
        Returns:
            mutated_ts (tskit.TreeSequence): Mutated tree sequence with split and secondary contact.
        """
        # Define demography
        # Define demography
        demography = msprime.Demography()
        demography.add_population(name="ancestral", initial_size=Nanc)
        demography.add_population(name="mainland", initial_size=N_mainland)
        demography.add_population(name="island", initial_size=N_island)
    
        # Define the population split at time T_split
        demography.add_population_split(time=T_split, derived=["mainland", "island"], ancestral="ancestral")
        demography.add_migration_rate_change(0., rate = m, source = "mainland", dest = "island")
        demography.add_migration_rate_change(T_contact, rate = 0., source = "mainland", dest = "island")
        demography.sort_events()
    
        samples = []
        samples.append(msprime.SampleSet(self.n_samples[0], population = 'mainland', ploidy = 2))
        samples.append(msprime.SampleSet(self.n_samples[1], population = 'island', ploidy = 2))
        # simulate ancestry
        ts = msprime.sim_ancestry(
            #sample_size=2 * population_size,
            samples = samples,

            sequence_length=self.L,
            recombination_rate=self.r,
            #mutation_rate=mutation_rate,
            demography=demography,
            #Ne=population_size
        )
        
        return self.mutate_and_return_(ts)
    

    
if __name__ == '__main__':
    t, N = chebyshev_history()
    
    N = 10 ** N
    
    Nt = list(zip(N, t))
    plot_size_history(Nt)
    plt.show()
    sys.exit()
    
    sim = DiscoalSimulator()
    
    sim.simulate()
    sys.exit()
    
    from scipy.spatial.distance import pdist, squareform
    
    sim = StepStoneSimulator(L = int(1e4), mu = 5.4e-9, r = 3.386e-9)

    for k in range(16):
        Fs, Ws, pop_vectors, coal_times, X, sites, ts = sim.simulate_fw()
        W = Ws[0]
    
        W = np.log(W + 1e-12)
        print(np.percentile(W, 10), np.percentile(W, 25), np.percentile(W, 50), np.percentile(W, 75), np.percentile(W, 90))        
    
    
    """
    h = []
    
    sim = StepStoneSimulator(int(1e4), r = 1e-8)
    
    for k in range(512):
        F, W, pop_vector, t_coal, X, sites, s = sim.simulate_fw_single()
        print(X.shape)
    """
    