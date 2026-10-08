# -*- coding: utf-8 -*-
import argparse
from scipy.stats import ks_2samp
import numpy as np
import tempfile
from pathlib import Path

from popgenml.data.simulators import MSPrimeSimulator, DiscoalSimulator
from popgenml.data.functions import tree_to_graph, graph_to_tree, tree_to_distmat, distmat_to_tree
from popgenml.data import PGTreeSequence, relate, singer

def test_statistical_equivalency(sample_config):
    ms_S = []
    disc_S = []
    
    sim_ms = MSPrimeSimulator(sample_config)
    sim_disc = DiscoalSimulator(sample_config)
    
    for i in range(1, 51):
        sim_ms.set_seeds({'Nt' : i})
        sim_disc.set_seeds({'Nt' : i})
        
        # Number of segregating sites = columns in the genotype matrix
        # Note: You may still want to change the seeds arg to a dict if required by your simulators
        ms_S.append(sim_ms.simulate(seeds=(i, i))['x'].shape[1])
        disc_S.append(sim_disc.simulate(seeds=(i, i))['x'].shape[1])
        
    # K-S Test checks if both sets of outputs are drawn from the same continuous distribution
    stat, p_value = ks_2samp(ms_S, disc_S)
    
    # We expect p > 0.05 if the distributions are equivalent
    assert p_value > 0.01, f"Simulators produced statistically divergent site frequencies (p={p_value})"


def run_constant_test():
    print('Running [discoal <=> msprime (constant pop)]...')
    
    config_content = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 100000
ploidy = 2

[samples]
pop1 = {'N0': 10000, 'n': 4}
"""
    test_statistical_equivalency(config_content)
    print('[discoal <=> msprime (constant pop)] PASSED!')


def run_cheby_test():
    print('Running [discoal <=> msprime (cheby)]...')
    
    config_content = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 100000
ploidy = 2

[samples]
# A diploid population with a variable size history defined by a spline
pop1 = {'Nt': 'ChebyshevHistory(target_snps=np.random.uniform(8000, 12000), n_haps=4, volatility = 2.0)', 'n': 4}
"""
    test_statistical_equivalency(config_content)
    print('[discoal <=> msprime (cheby)] PASSED!')


def run_conversion_test():
    print('Running [Tree conversions: graph and distmat]...')
    
    config_content = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 100000
ploidy = 2

[samples]
pop1 = {'N0': 10000, 'n': 4}
"""
    simulator = MSPrimeSimulator(config_content)
    ret = simulator.simulate()

    ts = ret['ts']
    tree = ts.first()

    # ===========
    print('  testing tree to graph and back...')

    times = sorted([tree.time(u) for u in tree.nodes()])
    times = [u for u in times if u > 0]
    times = np.array(times)

    # dynamically set 'n' based on the simulated tree's number of samples
    x, edges = tree_to_graph(tree, n=ret['x'].shape[0])
    ts_tree = graph_to_tree(x, edges)

    times_ = sorted([ts_tree.time(u) for u in ts_tree.nodes()])
    times_ = [u for u in times_ if u > 0]
    times_ = np.array(times_)

    assert np.sum((times - times_) ** 2) == 0
    assert tree.rf_distance(ts_tree) == 0

    print('  success!')
    # ==========

    # ===========
    print('  testing tree to distance matrix and back...')

    D = tree_to_distmat(tree)
    ts_tree, _ = distmat_to_tree(D)

    times_ = sorted([ts_tree.time(u) for u in ts_tree.nodes()])
    times_ = [u for u in times_ if u > 0]
    times_ = np.array(times_)

    assert np.sum((times - times_) ** 2) == 0
    assert tree.rf_distance(ts_tree) == 0
    print('  success!!')
    # ==========
    
    print('[Tree conversions] PASSED!')
    
# infer a tree sequence via Relate
def run_relate_test():
    print('Running [inference: relate]...')
    
    config_content = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 1000000
ploidy = 2

[samples]
pop1 = {'N0': 10000, 'n': 2}
"""
    simulator = MSPrimeSimulator(config_content)
    ret = simulator.simulate()
    
    mu = simulator.mu
    r = simulator.r
    
    N = 10000
    
    ts_est = relate(ret['x'], ret['pos'], ret['x'].shape[0], mu, r, 1e6, N, verbose = True)
    ts = PGTreeSequence.from_tskit(ret['ts'])
    
    print('[inference: relate] SUCCESS!')

    print('Running [ts comparisons: relate]')
    
    err = ts_est.average_kc_distance(ts)
    print('kc err = {}'.format(err))
    
    err = ts_est.average_rf_distance(ts)
    print('rf err = {}'.format(err))
        
    err = ts_est.average_rms_log_coal_time(ts)
    print('rms log coal err = {}'.format(err))
        
    err = ts_est.breakpoint_chamfer_distance(ts)
    print('chamfer err = {}'.format(err))
    
    print('[ts comparisons: relate] SUCCESS!')
    
def run_singer_test():
    print('Running [inference: singer]...')
    
    config_content = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 1000000
ploidy = 2

[samples]
pop1 = {'N0': 10000, 'n': 2}
"""
    simulator = MSPrimeSimulator(config_content)
    ret = simulator.simulate()
    
    mu = simulator.mu
    r = simulator.r
    
    N = 10000
    
    ts_est = next(singer(ret['x'], ret['pos'], 1e6, mu, r, None, n_est=N))
    ts = PGTreeSequence.from_tskit(ret['ts'])
    
    print('[inference: singer] SUCCESS!')

    print('Running [ts comparisons: singer]')
    
    err = ts_est.average_kc_distance(ts)
    print('kc err = {}'.format(err))
    
    err = ts_est.average_rf_distance(ts)
    print('rf err = {}'.format(err))
        
    err = ts_est.average_rms_log_coal_time(ts)
    print('rms log coal err = {}'.format(err))
        
    err = ts_est.breakpoint_chamfer_distance(ts)
    print('chamfer err = {}'.format(err))
    
    print('[ts comparisons: relate] SUCCESS!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run statistical equivalency tests for popgenml simulators.")
    parser.add_argument(
        '-t', '--test', 
        type=str, 
        choices=['constant', 'cheby', 'conversion', 'relate', 'singer', 'all'], 
        default='all',
        help="Specify which test to run. Options: 'constant', 'cheby', 'conversion', or 'all' (default: all)"
    )
    
    args = parser.parse_args()

    if args.test in ['constant', 'all']:
        run_constant_test()
        print() 
        
    if args.test in ['cheby', 'all']:
        run_cheby_test()
        print()

    if args.test in ['conversion', 'all']:
        run_conversion_test()
        print()
        
    if args.test in ['relate', 'all']:
        run_relate_test()
        print()
        
    if args.test in ['singer', 'all']:
        run_singer_test()
        print()