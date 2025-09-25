# -*- coding: utf-8 -*-
from popgenml.data.io_ import read_slim
from popgenml.data.functions import newick_to_tree

import msprime
import numpy as np
import os

import logging
import random
import subprocess

from numpy.polynomial.chebyshev import Chebyshev
from pkg_resources import resource_filename
import re

import pickle

from scipy import stats
from typing import Dict, Union, Any
from scipy.interpolate import interp1d
import configparser
import math

import ast
# scipy.stats._distn_infrastructure.rv_continuous and rv_discrete are the base classes
# for continuous and discrete distributions, respectively.
# We use this for type hinting to make the code clearer.
Distribution = Union[stats._distn_infrastructure.rv_continuous, stats._distn_infrastructure.rv_discrete]

class ParameterPrior:
    """
    A class to sample from a dictionary of named scipy.stats distributions.
    This version does not contain any internal print statements.
    """

    def __init__(self, distributions: Dict[str, Distribution]):
        """
        Initializes the ParameterPrior.

        Args:
            distributions (Dict[str, Distribution]): A dictionary where keys are
                string variable names and values are scipy.stats continuous
                or discrete distribution objects (e.g., stats.norm(loc=0, scale=1)).
        
        Raises:
            TypeError: If the input is not a dictionary.
            ValueError: If the dictionary is empty.
        """
        if not isinstance(distributions, dict):
            raise TypeError("Input 'distributions' must be a dictionary.")
        if not distributions:
            raise ValueError("Input 'distributions' dictionary cannot be empty.")
            
        self.distributions = distributions

    def sample(self, n_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        Draws a specified number of samples from each distribution.

        Args:
            n_samples (int, optional): The number of samples to draw for each
                variable. Defaults to 1.

        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are the variable
                names and values are NumPy arrays containing the samples.
                If n_samples is 1, the value will be a single-element array.
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("'n_samples' must be a positive integer.")
        
        # Create a dictionary to hold the samples for each variable.
        samples_dict = {}
        
        # Iterate through the distributions provided during initialization.
        for var_name, distribution in self.distributions.items():
            # Use the .rvs() method of the distribution object to generate random variates.
            # The 'size' parameter determines how many samples are drawn.
            samples = distribution.rvs(size=n_samples)
            samples_dict[var_name] = samples
            
        return samples_dict
    
class History:
    """
    Base class for objects that represent a history or trajectory over time.
    Subclasses should implement a method to sample a (time, value) tuple.
    """
    def sample_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a single (time, value) trajectory.
        
        Returns:
            A tuple containing the time points and the corresponding values.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class BottleNeckHistory(History):
    """
    Size history for an instantaneous population size change that takes place some number
    of generations ago.  
    """
    def __init__(self, N0: Distribution, N1: Distribution, T: Distribution):
        """
        Args:
            N0 (Distribution): A scipy.stats distribution object for the initial effective population at time = 0
            N1 (Distribution): scipy.stats distribution object for the population size after time = T
            T (Distribution): scipy.stats distribution object for T = time of the bottleneck / population expansion
        """
        self.N0 = N0
        self.N1 = N1
        self.T = T
        
    def sample_curve(self):
        t = [0]
        t.append(self.T.rvs(size = 1)[0])
        
        N = [self.N0.rvs(size = 1)[0], self.N1.rvs(size = 1)[0]]
        
        self.co = N + t
        
        return t, N

class SplineHistory(History):
    """
    Generates a population size history curve using a spline interpolation.
    The population size at control points is drawn from a given scipy.stats distribution.
    
    The number of control points for each curve is chosen from a uniform discrete distribution with support: range(min_k, max_k + 1)
    The times for the control points is drawn from a uniform distribution from log(t) = 0 to log(t) = max_log_time.
    """
    def __init__(self, N: Distribution, max_k = 33, 
                 min_k = 3, max_log_time = 11, n_time_points = 128, kind = 'linear'):
        """
        Initializes the spline history generator.

        Args:
            N (Distribution): A scipy.stats distribution object for sampling
                population sizes (y-values).
            max_k (int, optional): The maximum number of control points. Defaults to 33.
            min_k (int, optional): The minimum number of control points. Defaults to 3
            max_log_time (int, optional): The maximum time on a log scale. Defaults to 11.
            n_time_points (int, optional): The number of points for the final time grid. Defaults to 128.
        """
        self.N = N
        self.kind = kind
        self.max_k = max_k
        self.max_log_time = max_log_time
        self.n_time_points = n_time_points
        
    def sample_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a parameter curve defined over time in generations.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the fine time grid (t)
            and the corresponding interpolated population sizes (or another parameter) (y).
        """
        n_points = np.random.choice(range(3, self.max_k + 1, 2))
        y_points = self.N.rvs(size=n_points)
        
        mi = np.min(y_points)
        delta = np.max(y_points) - np.min(y_points)
        
        y_points = (y_points - mi) / delta
        
        t_points = [0.] + sorted(list(np.random.uniform(0., self.max_log_time, n_points - 1)))
        max_t = np.max(t_points)
                
        spline = interp1d(t_points, y_points, kind = self.kind)
        t = np.linspace(0, t_points[-1], self.n_time_points)
        y = spline(t)
        
        return np.exp(t), y * delta + mi
    
def _parse_prior_value(value_str: str, safe_globals: dict) -> Any:
    """Helper to parse a string from the config into a float, int, or distribution."""
    try:
        # First try to parse as an integer
        return int(value_str)
    except ValueError:
        try:
            # Then try as a float
            return float(value_str)
        except ValueError:
            # If it fails, assume it's a Python expression for a distribution or class
            try:
                return eval(value_str, safe_globals)
            except Exception as e:
                raise ValueError(f"Could not parse value: '{value_str}'. Error: {e}")

def create_prior_from_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Creates a nested dictionary of priors from a Python config file.

    Args:
        config_path (str): The path to the .ini configuration file.

    Returns:
        A dictionary with 'base' and 'samples' keys, containing the parsed priors.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    # Define a safe context for eval(), allowing access to 'stats' and custom classes.
    safe_globals = {
        'stats': stats,
        'math' : math,
        'SplineHistory': SplineHistory,
        'BottleNeckHistory': BottleNeckHistory,
    }

    priors = {'base': {}, 'samples': {}, 'migration' : {}, 'discoal' : {}, 'demography' : {}}
    
    # Process the [base] section for simple priors
    if 'base' in config:
        for key, value_str in config.items('base'):
            priors['base'][key] = _parse_prior_value(value_str, safe_globals)

    # Process the [samples] section for population-specific priors
    if 'samples' in config:
        for pop_name, value_str in config.items('samples'):
            # Initialize the nested dictionary for this population
            priors['samples'][pop_name] = {}
            
            # Safely evaluate the string representation of the dictionary, e.g., "{'Nt': 'SplineHistory(...)'}"
            pop_config = ast.literal_eval(value_str)
            
            for key in pop_config.keys():
                pop_config[key] = _parse_prior_value(pop_config[key], safe_globals)
            

            # Assign it to the correct nested structure
            priors['samples'][pop_name] = pop_config
    
    if 'migration' in config:
        for key, value_str in config.items('migration'):
            priors['migration'][key] = _parse_prior_value(value_str, safe_globals)
    else:
        priors['migration'] = None
        
    if 'discoal' in config:
        for key, value_str in config.items('discoal'):
            priors['discoal'][key] = _parse_prior_value(value_str, safe_globals)
    else:
        priors['discoal'] = None
        
    if 'demography' in config:
        for key, value_str in config.items('demography'):
            priors['demography'][key] = _parse_prior_value(value_str, safe_globals)
    else:
        priors['demography'] = None
        
    return priors

class BaseSimulator:
    """
    A simulator that holds base parameters and validated sample population priors,
    initialized directly from a configuration file.
    """
    def __init__(self, config_path: str):
        """
        Initializes the BaseSimulator from a configuration file.

        Args:
            config_path (str): The path to the .ini configuration file.

        Raises:
            KeyError: If a required key is missing.
            TypeError: If a value has an incorrect type.
            ValueError: If a value is out of the allowed range (e.g., ploidy).
        """
        # --- Create priors from the config file ---
        priors = create_prior_from_config(config_path)
        base_priors = priors['base']
        sample_priors = priors['samples']
        self.migration_priors = priors['migration']
        self.discoal_priors = priors['discoal']
        self.demography_priors = priors['demography']

        # --- Validate and store base priors ---
        required_base_keys = ['mu', 'r', 'l', 'ploidy']
        for key in required_base_keys:
            if key not in base_priors:
                raise KeyError(f"Required key '{key}' not found in [base] section of config.")
        
        L_val = base_priors['l']
        if not isinstance(L_val, int):
            raise TypeError(f"The value for 'L' must be an integer, but got {type(L_val)}.")

        ploidy_val = base_priors['ploidy']
        if not isinstance(ploidy_val, int):
            raise TypeError(f"The value for 'ploidy' must be an integer, but got {type(ploidy_val)}.")

        if not ploidy_val in [1, 2]:
            raise ValueError(f"The value for 'ploidy' given ({ploidy_val}) is not in [1, 2]...")
    
        self.mu = base_priors['mu']
        self.r = base_priors['r']
        self.L = L_val
        self.ploidy = base_priors['ploidy']
        
        # --- Validate and store sample priors ---
        self.samples = {}
        required_sample_keys = ['n']
        
        for pop_name, pop_priors in sample_priors.items():            
            # Check for required keys
            for key in required_sample_keys:
                if key not in pop_priors:
                    raise KeyError(f"Required key '{key}' not found in priors for sample '{pop_name}'.")
            
            # Check for N0 or Nt
            if 'N0' not in pop_priors and 'Nt' not in pop_priors:
                raise KeyError(f"Either 'N0' or 'Nt' must be specified for sample '{pop_name}'.")

            # Validate types and values
            if not isinstance(pop_priors['n'], int):
                raise TypeError(f"'n' for sample '{pop_name}' must be an integer > 0.")
            
            if pop_priors['n'] < 0:
                raise TypeError(f"'n' for sample '{pop_name}' must be an integer >= 0.")
            
            # If all checks pass, store the priors for this sample
            self.samples[pop_name] = pop_priors
            
class MSPrimeSimulator(BaseSimulator):
    def __init__(self, config_file, mutation_model = msprime.BinaryMutationModel()):
        super().__init__(config_file)
        
        self.mutation_model = mutation_model
        
    def make_demography(self):
        demography = msprime.Demography()
        
        for pop_name in self.samples.keys():
            if 'N0' in self.samples[pop_name].keys() and (not 'Nt' in self.samples[pop_name].keys()):
                # we expect this to be a positive float or integer
                N0 = self.samples[pop_name]['N0']
                if isinstance(N0, (float, int)):
                    demography.add_population(name=pop_name, initial_size = N0)
                else:
                    N0 = N0.rvs(size = 1)[0]
                    
                    demography.add_population(name=pop_name, initial_size = N0)
            
            elif 'Nt' in self.samples[pop_name].keys():
                Nt = self.samples[pop_name]['Nt']
                
                # list of tuples
                if isinstance(Nt, list):
                    demography.add_population(name=pop_name, initial_size = self.samples[pop_name]['Nt'][0])
                    
                    for N1, T in Nt:
                        demography.add_population_parameters_change(time=T, population = pop_name, initial_size=N1)
                else: 
                    t, N = Nt.sample_curve()
                    
                    demography.add_population(name=pop_name, initial_size = N[0])
                    
                    for N1, T in zip(N, t):
                        demography.add_population_parameters_change(time=T, population = pop_name, initial_size=N1)
            else:
                raise ValueError("All simulated populations must have a key 'Nt' or 'N0'")
        
        
        if self.migration_priors:
            for key in self.migration_priors:
                src, dst = key.split(',')
                m = self.migration_priors[key]
                
                if isinstance(m, list):
                    for m_, t_ in m:
                        demography.add_migration_rate_change(time = t_, source = src, dest = dst, rate = m_)
                else:
                    T, M = m.sample_curve()
                    
                    for m_, t_ in zip(M, T):
                        demography.add_migration_rate_change(time = t_, source = src, dest = dst, rate = m_)
        
        if self.demography_priors:
            for key in self.demography_priors:
                c1, c2, p = key.split(',')
                
                T = self.demography_priors[key]
                
                if isinstance(m, float):
                    demography.add_population_split(time = T, derived = [c1, c2], ancestral = p)
                else:
                    T = T.rvs(size = 1)[0]
            
                    demography.add_population_split(time = T, derived = [c1, c2], ancestral = p)
        
        demography.sort_events()
        
        return demography
                    
    def simulate(self, verbose = False):
        self.demography = self.make_demography()
                
        samples = dict()
        for pop in self.samples.keys():
            samples[pop] = self.samples[pop]['n']
        
        if isinstance(self.r, float):
            r = self.r
        else:
            r = self.r.rvs(size = 1)[0]
        
        # simulate ancestry
        ts = msprime.sim_ancestry(
            samples = samples,
            sequence_length = self.L,
            recombination_rate = r,
            ploidy = self.ploidy,
            demography = self.demography,
        )
        
        return self.mutate_and_return_(ts)
    
    def mutate_and_return_(self, ts):
        result = dict()
        
        if isinstance(self.mu, float):
            mu = self.mu
        else:
            mu = self.mu.rvs(size = 1)[0]
        
        # simulate mutations, binary discrete model
        mutated_ts = msprime.sim_mutations(ts, rate=mu, model=msprime.BinaryMutationModel())
        
        X = mutated_ts.genotype_matrix()
        X[X > 1] = 1
        X = X.T

        sites = [u.position for u in list(mutated_ts.sites())]
        sites = np.array(sites) / self.L # scale from 0 to 1
        
        result['x'] = X
        result['pos'] = sites
        result['ts'] = mutated_ts
        
        return result

class DiscoalSimulator(BaseSimulator):
    def __init__(self, config_file):
        super().__init__(config_file)
        
        if self.discoal_priors is not None:
            # selection coefficient
            if 's' in self.discoal_priors.keys():
                self.s = self.discoal_priors['s']
            else:
                self.s = None
        
            # prior on location of selection
            if 'x' in self.discoal_priors.keys():
                self.x = self.discoal_priors['x']
            else:
                self.x = None
                
            # other args passed to discoal
            if 'args' in self.discoal_priors.keys():
                self.args = self.discoal_priors['args']
            else:
                self.args = None
        else:
            self.s = None
            self.x = None
            self.args = None
                    
    def simulate(self, verbose = False):
        pops = []
        
        for ix, pop_name in enumerate(sorted(self.samples.keys())):
            if 'N0' in self.samples[pop_name].keys() and (not 'Nt' in self.samples[pop_name].keys()):
                # we expect this to be a positive float or integer
                N0 = self.samples[pop_name]['N0']
                if not isinstance(N0, (float, int)):
                    N0 = N0.rvs(size = 1)[0]
                    
                Nt = None
            elif 'Nt' in self.samples[pop_name].keys():
                Nt = self.samples[pop_name]['Nt']
                
                # list of tuples
                if not isinstance(Nt, list):
                    t, N = Nt.sample_curve()
                    
                    Nt = list(zip(N, t))
        
                N0 = Nt[0][0]
            
            n = self.samples[pop_name]['n']
            
            if self.ploidy == 2:
                n *= 2
            
            pops.append((N0, Nt, n))
        
        N0 = pops[0][0]
        self.N = N0
        
        if isinstance(self.r, float):
            r = self.r
        else:
            r = self.r.rvs(size = 1)[0]
        
        if isinstance(self.mu, float):
            mu = self.mu
        else:
            mu = self.mu.rvs(size = 1)[0] 
        
        theta = 4 * N0 * self.L * mu
        rho = 4 * N0 * self.L * r
        
        total_n = sum([u[-1] for u in pops])
        
        cmd = 'discoal {} 1 {} -t {} -r {} -T'.format(total_n, self.L, theta, rho)
        
        if len(pops) > 1:
            cmd += ' -p {} '.format(len(pops)) + ' '.join([str(u[-1]) for u in pops])
        
        size_strs = []
        
        for ix, pop in enumerate(pops):
            N0_, Nt, n = pop
            
            if ix > 0:
                pop_size_str = '-en 0.0 {0} {1}'.format(ix, N0_ / N0)
            else:
                pop_size_str = ''
                
            if Nt is not None:
                for (N, t) in Nt[1:]:
                    _ = ' -en {0} {2} {1}'.format(t / (4 * N0), N / N0, ix)            
                    pop_size_str += _
                    
            size_strs.append(pop_size_str)
        
        cmd = ' '.join([cmd] + size_strs)
        
        if self.args is not None:
            cmd = ' '.join((cmd, self.args))
        
        if self.s is not None:
            if isinstance(self.s, float):
                s = self.s
            else:
                s = self.s.rvs(size = 1)[0]
                
            cmd = ' '.join((cmd, '-a {}'.format(4 * N0 * s)))
        
        if self.x is not None:
            if isinstance(self.s, float):
                x = self.x
            else:
                x = self.x.rvs(size = 1)[0]
                
            cmd = ' '.join((cmd, '-x {}'.format(x)))
        
        if verbose:
            print(cmd)
            
        self.co = cmd
        
        return self.run_and_parse_cmd_(cmd)
        
    def run_and_parse_cmd_(self, cmd_):      
        process = subprocess.Popen(cmd_, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        
        lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())  # Remove trailing newline
                
        while True:
            line = lines[0]
            
            if len(line) == 0:
                del lines[0]
                continue
            
            if not line[0] == '[':
                del lines[0]
            else:
                break
                    
        trees = []
        intervals = []
        l = 0
        
        bins = [0]
        while True:
            line = lines[0]
            del lines[0]
            
            if len(line) > 0:
                if line[0] == '[':
                    n_sites = re.findall('\[(\d+)\]', line)[0]
                    n_digits = len(n_sites)
                    n_sites = int(n_sites)
                    
                    intervals.append((l, l + n_sites))
                    l += n_sites
                    bins.append(l)
                    
                    line = line[n_digits + 2:]
    
                    tree = newick_to_tree(line, multiplier = 4 * self.N)
                    trees.append(tree)
                else:
                    break
            else:
                break
                    
        start = 0
        while lines[start] != '//':
            start += 1

        start += 1        
        lines = lines[start:]
        n_segsites = int(lines[0].split()[-1])
        pos = np.array(list(map(float, lines[1].split()[1:])))
        
        trees_ = []
        intervals_ = []
        n_snps = 0
        
        intervals = np.array(intervals)
                
        for ix in range(len(trees)):
            l, r = intervals[ix]

            ii = np.where((pos * self.L >= l) & (pos * self.L < r))[0]
            n_snps += len(ii)
            
            if len(ii) > 0:
                trees_.append(trees[ix])
                intervals_.append(intervals[ix])
                   
        x = []
        for line in lines[2:]:
            x.append(np.fromstring(line,'u1') - ord('0'))            
        
        x = np.array(x, dtype = np.uint8)
        
        result = dict()
        result['x'] = x
        result['pos'] = pos
        result['ts'] = trees_ # just a list of trees here
        result['intervals'] = intervals_
        
        return result
                        
### end of written code
## --------------------            
# --- Example Usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a temporary file path to pass to the function
    config_path = '../PopGenML/configs/mig_n4.ini'

    sim = MSPrimeSimulator(config_path)
    popsize = sim.samples['pop1']['Nt']
    t, N = popsize.sample_curve()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for k in range(12):
        # Sample and plot a history curve from the history object
        print("\nSampling history for 'pop1'...")
        t_curve, y_curve = popsize.sample_curve()
        
        ax.plot(np.log(t_curve), y_curve, label='Sampled Spline History for pop1', color='dodgerblue', linewidth=1)
    
    ax.set_title('Sampled Population Size History from Config', fontsize=16)
    ax.set_xlabel('Time (in generations)', fontsize=12)
    ax.set_ylabel('Effective Population Size (N_e)', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    
    print("\nDisplaying plot of the sampled population history...")
    plt.show()
    
    sys.exit()
    

    """
    # 2. Create the prior object from the config file
    print("--- Parsing config file and creating priors ---")
    priors = create_prior_from_config(config_path)
    print("Priors created successfully.")

    # 3. Demonstrate accessing the new, nested structure
    print("\n--- Accessing created priors ---")
    
    print("\nBase priors:")
    mu_prior = priors['base']['mu']
    r_const = priors['base']['r']
    print(f"  mu is a distribution: {mu_prior}")
    print(f"  r is a constant: {r_const}")

    print("\nSample priors:")
    pop1_history_generator = priors['samples']['pop1']['Nt']
    pop2_n0_prior = priors['samples']['pop2']['N0']
    print(f"  pop1 Nt model is a: {type(pop1_history_generator)}")
    print(f"  pop2 N0 prior is a: {pop2_n0_prior}")

    # 4. Sample from the priors to show they are functional
    print("\n--- Sampling from priors ---")
    print(f"  Sampling mu: {mu_prior.rvs(size=1)[0]:.2e}")
    print(f"  Sampling pop2 N0: {pop2_n0_prior.rvs(size=1)[0]:.2f}")
    
    # 5. Plot the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for k in range(12):
        # Sample and plot a history curve from the history object
        print("\nSampling history for 'pop1'...")
        t_curve, y_curve = pop1_history_generator.sample_curve()
        
        ax.plot(np.log(t_curve), y_curve, label='Sampled Spline History for pop1', color='dodgerblue', linewidth=1)
    
    ax.set_title('Sampled Population Size History from Config', fontsize=16)
    ax.set_xlabel('Time (in generations)', fontsize=12)
    ax.set_ylabel('Effective Population Size (N_e)', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    
    print("\nDisplaying plot of the sampled population history...")
    plt.show()
    """