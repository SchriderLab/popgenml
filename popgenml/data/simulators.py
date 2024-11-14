# -*- coding: utf-8 -*-

import msprime
import numpy as np
from functions import make_FW_rep, read_anc
from io import BytesIO, StringIO

from skbio import read
from skbio.tree import TreeNode
import copy
import tempfile
import os
import glob

class BaseSimulator(object):
    # L is the size of the simulation in base pairs
    # specify mutation rate
    
    # immutable properties of the simulator are defined here:
    # L: the size of the simulation in base pairs
    # mu: mutation rate
    # r: recombination rate
    # whether or not diploid individuals are simulated (vs haploid)
    # the number of samples
    def __init__(self, L = int(1e5), mu = 1.26e-8, r = 1.007e-8, diploid = True, n_samples = [40]):
        self.L = L
        self.mu = mu
        self.r = r
        self.diploid = diploid
        
        self.n_samples = n_samples
        self.sample_size = sum(n_samples)
        
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
    
    # returns FW image(s)
    def simulate_fw(self, *args, method = 'true'):
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
            tables = s.dump_tables()
            tables.sort()
    
            # should be an iteration here but need to be careful in general due to RAM
            for t in s.aslist():
            
                f = StringIO(t.as_newick())  
                root = read(f, format="newick", into=TreeNode)
                root.assign_ids()
                        
                populations = list(tables.nodes.population)
                
                tips = [u for u in root.postorder() if u.is_tip()]
                for ix, t in enumerate(tips):
                    t.pop = populations[ix]
                    
                children = root.children
                t = max(tables.nodes.time)
                
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
        
        
                F, W, pop_vector, t_coal = make_FW_rep(root, sample_sizes)
                i, j = np.tril_indices(F.shape[0])
                F = F[i, j]
                
                Fs.append(F)
                Ws.append(W)
                pop_vectors.append(pop_vector)
                coal_times.append(t_coal)
                
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
            
            f = open(os.path.join(temp_dir.name, 'sim.vcf'), 'w')
            s.write_vcf(f)
            f.close()
            
            tag = ms_file.split('/')[-1].split('.')[0]
            cmd_ = self.rcmd.format('sim.haps', 'sim.sample', '../sim', odir)
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
            
            anc_file = os.path.join(odir, '{}.anc'.format(ofile))
            Fs, Ws, snps, coal_times = read_anc(anc_file)
            
            temp_dir.cleanup()
            
        return Fs, Ws, pop_vectors, coal_times, s
        
class BottleNeckSimulator(BaseSimulator):
    def __init__(self, L = int(1e6), mu = 1.26e-8, r = 1.007e-8, diploid = True, n_samples = [20]):
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
            
            #mutation_rate=mutation_rate,
            demography=demography,
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
    def __init__(self, L = int(1e6), mu = 1.26e-8, r = 1.007e-8, diploid = False, n_samples = [65, 64]):
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
            additional_nodes=(
                msprime.NodeType.RECOMBINANT),
            sequence_length=self.L,
            recombination_rate=self.r,
            #mutation_rate=mutation_rate,
            demography=demography,
            #Ne=population_size
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
        print("Starting population split with secondary contact simulation with parameters:")
        print(f"  Nanc: {Nanc}, N_mainland: {N_mainland}, N_island: {N_island}, T_split: {T_split}, T_contact: {T_contact}, m: {m}")

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