# -*- coding: utf-8 -*-
import argparse
from scipy.stats import ks_2samp
import numpy as np
import tempfile
from pathlib import Path

from popgenml.data.simulators import MSPrimeSimulator, DiscoalSimulator, SLiMSimulator
from popgenml.data.functions import tree_to_graph, graph_to_tree, tree_to_distmat, distmat_to_tree
from popgenml.data import PGTreeSequence, relate, singer

import os

def test_statistical_equivalency(sim_class1, sim_class2, sample_config, n_iters=50):
    """General helper to compare two simulator classes."""
    S1 = []
    S2 = []
    
    sim1 = sim_class1(sample_config)
    sim2 = sim_class2(sample_config)
    
    for i in range(1, n_iters + 1):
        sim1.set_seeds({'Nt' : i})
        sim2.set_seeds({'Nt' : i})
        
        S1.append(sim1.simulate(seeds=(i, i))['x'].shape[1])
        S2.append(sim2.simulate(seeds=(i, i))['x'].shape[1])
        
    stat, p_value = ks_2samp(S1, S2)
    assert p_value > 0.01, f"{sim_class1.__name__} and {sim_class2.__name__} produced divergent site frequencies (p={p_value})"



def run_simulator_tests():
    print('Running [Simulator Equivalency Tests]...')
    
    constant_config = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 100000
ploidy = 2

[samples]
pop1 = {'N0': 100, 'n': 2}
"""

    cheby_config = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 100000
ploidy = 2

[samples]
pop1 = {'Nt': 'ChebyshevHistory(target_snps=100, n_haps=4, volatility = 1.0, N_min = 10, N_max = 1e3)', 'n': 2}
"""

    print('  testing [discoal <=> msprime (constant pop)]...')
    test_statistical_equivalency(MSPrimeSimulator, DiscoalSimulator, constant_config)
    print('  success!')

    print('  testing [discoal <=> msprime (cheby)]...')
    test_statistical_equivalency(MSPrimeSimulator, DiscoalSimulator, cheby_config)
    print('  success!')

    # Lowering iterations for SLiM tests as forward simulations are much slower
    print('  testing [slim <=> msprime (constant pop)]...')
    test_statistical_equivalency(MSPrimeSimulator, SLiMSimulator, constant_config, n_iters=20)
    print('  success!')

    print('  testing [slim <=> msprime (cheby)]...')
    test_statistical_equivalency(MSPrimeSimulator, SLiMSimulator, cheby_config, n_iters=20)
    print('  success!')

    print('[Simulator Equivalency Tests] PASSED!')


# ... (run_conversion_test, run_relate_test, run_singer_test, run_io_test remain unchanged) ...


    
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
    
    print(sum(ts_est.snps), sum(ts.snps))
    
    print('[inference: relate] SUCCESS!')

    print('Running [ts comparisons: relate]')
    
    err = ts_est.kc_distance(ts)
    print('kc err = {}'.format(err))
    
    err = ts_est.rf_distance(ts)
    print('rf err = {}'.format(err))
        
    err = ts_est.rms_log_coal_error(ts)
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
    L = 1e6
    
    N = 10000
    
    ts_est = next(singer(ret['x'], ret['pos'], mu, r, L))
    ts = PGTreeSequence.from_tskit(ret['ts'])
    
    print(sum(ts.snps), sum(ts_est.snps))
    
    print('[inference: singer] SUCCESS!')

    print('Running [ts comparisons: singer]')
    
    err = ts_est.kc_distance(ts)
    print('kc err = {}'.format(err))
    
    err = ts_est.rf_distance(ts)
    print('rf err = {}'.format(err))
        
    err = ts_est.rms_log_coal_error(ts)
    print('rms log coal err = {}'.format(err))
        
    err = ts_est.breakpoint_chamfer_distance(ts)
    print('chamfer err = {}'.format(err))
    
    print('[ts comparisons: singer] SUCCESS!')
    
def run_io_test():
    print('Running [PGTreeSequence IO: save and load]...')
    
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
    
    # Convert to PGTreeSequence
    ts = PGTreeSequence.from_tskit(ret['ts'])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_ts.pgts"
        
        # Test Save
        ts.save(str(file_path))
        
        # Test Load
        ts_loaded = PGTreeSequence.load(str(file_path))
        
        # Verify internal attributes match
        assert len(ts.trees) == len(ts_loaded.trees), "Mismatch in number of trees after load"
        assert ts.snps == ts_loaded.snps, "Mismatch in SNPs after load"
        
        # Verify structural identity
        err = ts.rf_distance(ts_loaded)
        assert err == 0, f"Loaded tree sequence differs structurally from original (RF distance: {err})"
        
    print('  success!')
    print('[PGTreeSequence IO: save and load] PASSED!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run statistical equivalency tests for popgenml simulators.")
    parser.add_argument(
        '-t', '--test', 
        type=str, 
        choices=['simulators', 'conversion', 'relate', 'singer', 'io', 'all'], 
        default='all',
        help="Specify which test to run. Options: 'constant', 'cheby', 'conversion', or 'all' (default: all)"
    )
    
    args = parser.parse_args()

    if args.test in ['simulators', 'all']:
        run_simulator_tests()
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
        
    if args.test in ['io', 'all']:
        run_io_test()
        print()