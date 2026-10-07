# -*- coding: utf-8 -*-
from scipy.stats import ks_2samp

import numpy as np
import tempfile
from pathlib import Path
from popgenml.data.simulators import MSPrimeSimulator, DiscoalSimulator

def test_statistical_equivalency(sample_config):
    ms_S = []
    disc_S = []
    
    sim_ms = MSPrimeSimulator(sample_config)
    sim_disc = DiscoalSimulator(sample_config)
    
    for i in range(1, 51):
        sim_ms.set_seeds({'Nt' : i})
        sim_disc.set_seeds({'Nt' : i})
        
        # Number of segregating sites = columns in the genotype matrix
        ms_S.append(sim_ms.simulate(seeds=(i, i))['x'].shape[1])
        disc_S.append(sim_disc.simulate(seeds=(i, i))['x'].shape[1])
        
    # K-S Test checks if both sets of outputs are drawn from the same continuous distribution
    stat, p_value = ks_2samp(ms_S, disc_S)
    
    # We expect p > 0.05 if the distributions are equivalent
    assert p_value > 0.01, f"Simulators produced statistically divergent site frequencies (p={p_value})"

if __name__ == '__main__':
    print('Running [discoal <=> msprime (contant pop)]...')
    
    config_content = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 100000
ploidy = 2

[samples]
pop1 = {'N0': 10000, 'n': 4}
"""
    test_statistical_equivalency(config_content)
    
    print('[discoal <=> msprime (contant pop)] PASSED!')
    
    config_content = """[base]
mu = 1.5e-8
r = 1.007e-8
L = 100000
ploidy = 2

[samples]
# A diploid population with a variable size history defined by a spline
pop1 = {'Nt': 'ChebyshevHistory(target_snps=np.random.uniform(8000, 12000), n_haps=4, volatility = 2.0)', 'n': 4}
"""

    print('Running [discoal <=> msprime (cheby)]...')
    test_statistical_equivalency(config_content)
    
    print('[discoal <=> msprime (cheby)] PASSED!')

