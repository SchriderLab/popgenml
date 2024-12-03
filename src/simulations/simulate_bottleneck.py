# -*- coding: utf-8 -*-
import argparse
import msprime
import numpy as np
import tskit
import matplotlib.pyplot as plt

import scipy.stats
import pandas as pd
import random

import os

def write_to_ms(ofile, X, sites, params):
    header = '// ' + ' '.join(['{0:04f}'.format(u) for u in params]) + '\n'
    ofile.write(header)
    
    n_segsites = X.shape[1]
    ofile.write('segsites: {}\n'.format(n_segsites))
    pos_line = 'positions: ' + ' '.join(['{0:04f}'.format(u) for u in sites]) + '\n'
    ofile.write(pos_line)
    
    for x in X:
        line = ''.join(list(map(str, list(x)))) + '\n'
        ofile.write(line)
        
    ofile.write('\n')
    
    

def simulate_bottleneck(initial_population_size, bottleneck_population_size, bottleneck_time, sequence_length, recombination_rate, mutation_rate):
    """
    Simulates n_simulations with a bottleneck event.
    
    Parameters:
        population_size (int): Initial population size.
        bottleneck_fraction (float): Fraction of the population size after bottleneck.
        bottleneck_time (int): Time (in generations) when bottleneck occurs.
        sequence_length (float): Length of the simulated sequence.
        recombination_rate (float): Recombination rate per base per generation.
        mutation_rate (float): Mutation rate per base per generation.
    """
    bottleneck_fraction = bottleneck_population_size / initial_population_size
    print(f"Simulating a bottleneck to this proportion of initial size: {bottleneck_fraction}")
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=bottleneck_population_size)
    demography.add_population_parameters_change(time=bottleneck_time, initial_size=initial_population_size)

    # Plot the demographic model
    #graph = msprime.Demography.to_demes(demography)
    #fig, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
    #demesdraw.tubes(graph, ax=ax, seed=1)
    #plt.savefig("/nas/longleaf/home/adaigle/ghist_2024/demography_plot.png")
    
    # simulate ancestry
    ts = msprime.sim_ancestry(
        #sample_size=2 * population_size,
        samples=20,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        #mutation_rate=mutation_rate,
        demography=demography,
        #Ne=population_size
    )
    
    # simulate mutations, binary discrete model
    mutated_ts = msprime.sim_mutations(ts, rate=mutation_rate, model=msprime.BinaryMutationModel())
    
    X = mutated_ts.genotype_matrix()
    X[X > 1] = 1
    X = X.T
    
    sites = [u.position for u in list(mutated_ts.sites())]
    sites = np.array(sites) / sequence_length
    
    return X, sites
    
def main():
    parser = argparse.ArgumentParser(description="Simulate a population bottleneck.")
    #parser.add_argument("--initial_population_size", type=int, required=True, help="Initial population size before the bottleneck")
    #parser.add_argument("--bottleneck_population_size", type=int, required=True, help="Population size during the bottleneck")
    #parser.add_argument("--bottleneck_time", type=int, required=True, help="Duration of the bottleneck in generations")
    parser.add_argument("--n_replicates", default = 1, type=int, required=False, help="number of replicates")
    parser.add_argument("--L", default = "1e6", help = "sequence length")
    
    parser.add_argument("--n_per", default = 1, type = int, required = False)
    
    parser.add_argument("--ofile", default = "test.msOut", type=str, required=False, help="Path to output file")
    parser.add_argument("--linear", action = "store_true")
    
    args = parser.parse_args()
    
    ofile = open(args.ofile, 'w')
    
    if not args.linear:
        for ix in range(args.n_replicates):
            bottleneck_fraction = np.random.uniform(0.01, 0.15)
            initial_population_size = np.random.uniform(13000, 15000)
            
            bottleneck_population_size = initial_population_size * bottleneck_fraction
            bottleneck_time = np.random.uniform(10, 500)
            
            # Hardcoded values for sequence_length, recombination_rate, and mutation_rate
            sequence_length = int(float(args.L))
            recombination_rate = 1.007e-8
            mutation_rate = 1.26e-8
            
            for ij in range(args.n_per):
        
                X, sites = simulate_bottleneck(initial_population_size, bottleneck_population_size, bottleneck_time, sequence_length, recombination_rate, mutation_rate)
                write_to_ms(ofile, X, sites, [bottleneck_fraction, bottleneck_time])
    
    ofile.close()
    
    os.system('gzip {}'.format(args.ofile))
    
if __name__ == '__main__':
    main()