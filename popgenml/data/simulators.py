# -*- coding: utf-8 -*-
from popgenml.data.functions import newick_to_tree
from popgenml.data.distributions import UniformFloatDiscrete, TruncatedExponential

import msprime
import numpy as np

import subprocess
import re

from scipy import stats
from typing import Dict, Union, Any
from scipy.interpolate import interp1d
import configparser
import math

import ast
import shlex
import sys
# scipy.stats._distn_infrastructure.rv_continuous and rv_discrete are the base classes
# for continuous and discrete distributions, respectively.
# We use this for type hinting to make the code clearer.
Distribution = Union[stats._distn_infrastructure.rv_continuous, stats._distn_infrastructure.rv_discrete]
import tempfile
import os

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
                 min_k = 3, max_log_time = 11, n_time_points = 128):
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
        self.kind = 'linear'
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
        
        t_points = [1] + sorted(list(np.exp(np.random.uniform(0., self.max_log_time, n_points - 1))))

        spline = interp1d(t_points, y_points, kind = self.kind)
        t = np.exp(np.linspace(0, np.log(t_points[-1]), self.n_time_points))
        y = spline(t)
        
        return t, y
    
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp, simpson
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.special import expit
from numpy.polynomial.chebyshev import chebval

def precompute_kingman_lineages(n, tau_max=15.0, num_pts=1000):
    """Precomputes the expected number of surviving lineages over coalescent time tau."""
    tau_grid = np.linspace(0, tau_max, num_pts)
    
    def kingman_odes(tau, P):
        dP = np.zeros_like(P)
        for k in range(2, n + 1):
            rate_out = k * (k - 1) / 2.0
            dP[k] = -rate_out * P[k]
            if k < n:
                rate_in = (k + 1) * k / 2.0
                dP[k] += rate_in * P[k+1]
        return dP
    
    P_init = np.zeros(n + 1)
    P_init[n] = 1.0
    
    sol = solve_ivp(kingman_odes, [0, tau_max], P_init, t_eval=tau_grid, method='BDF')
    states = np.arange(2, n + 1)
    expected_lineages = np.sum(sol.y[2:] * states[:, None], axis=0)
    
    return interp1d(sol.t, expected_lineages, kind='cubic', bounds_error=False, fill_value=0.0)

class TargetedHistory:
    """
    Base class for sampling demographic histories that target a specific expected 
    number of SNPs within strictly enforced population bounds.
    """
    def __init__(self, target_snps=20000, n_haps=16, mu=1.5e-8, seq_len=2.5e6,
                 N_min=5000.0, N_max=100000.0, T_max_sim=200000, n_sim_epochs=100,
                 T_max_math=2_000_000, n_math_pts=1000):
        
        self.target_Ln = target_snps / (mu * seq_len)
        self.N_min = N_min
        self.N_max = N_max
        
        # 1. Math Grid: Massive horizon to ensure the coalescent integral completes
        self.t_math = np.geomspace(1, T_max_math + 1, n_math_pts) - 1 
        self.x_math = 2.0 * (self.t_math / T_max_math) - 1.0
        
        # 2. Simulation Grid: Coarser horizon for efficient msprime stepping
        self.t_sim = np.geomspace(1, T_max_sim + 1, n_sim_epochs) - 1
        
        self.expected_A_func = precompute_kingman_lineages(n_haps)

    def _scale_and_bound(self, raw_log_shape):
        """Finds the optimal scalar, bounds the curve, and interpolates to the sim grid."""
        def expected_Ln_error(log_c):
            squashed = expit(log_c + raw_log_shape)
            N_t = self.N_min + squashed * (self.N_max - self.N_min)
            
            # Using 1.0 / (2.0 * N_t) assumes msprime uses ploidy=2 
            inv_2N = 1.0 / (2.0 * N_t)
            Lambda_t = cumulative_trapezoid(inv_2N, self.t_math, initial=0)
            
            expected_Ln = simpson(y=self.expected_A_func(Lambda_t), x=self.t_math)
            return expected_Ln - self.target_Ln
            
        # Solve for the exact log-shift
        log_c_star = brentq(expected_Ln_error, -20.0, 20.0)
        
        # Generate the mathematically complete curve
        squashed_final = expit(log_c_star + raw_log_shape)
        N_t_math = self.N_min + squashed_final * (self.N_max - self.N_min)
        
        # Interpolate down to the fast simulation grid
        N_t_sim = np.interp(self.t_sim, self.t_math, N_t_math)
        
        return self.t_sim, N_t_sim


class ChebyshevHistory(TargetedHistory):
    """Generates smoothly fluctuating trajectories using Chebyshev polynomials."""
    
    def __init__(self, num_coeffs=13, volatility=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_coeffs = num_coeffs
        self.volatility = volatility

    def sample_curve(self):
        # Decaying variance limits aggressive high-frequency oscillations
        variances = np.array([self.volatility / (j + 1)**0.5 for j in range(self.num_coeffs)])
        coeffs = np.random.randn(self.num_coeffs) * variances
        
        # Evaluate on the massive math grid
        raw_log_shape = chebval(self.x_math, coeffs)
        
        return self._scale_and_bound(raw_log_shape)


class ExponentialHistory(TargetedHistory):
    """Generates trajectories mapping recent exponential growth or decay back into a plateau."""
    
    def __init__(self, abs_r_min=1e-5, abs_r_max=1e-3, **kwargs):
        super().__init__(**kwargs)
        # We define the bounds for the absolute magnitude of the rate
        self.abs_r_min = abs_r_min
        self.abs_r_max = abs_r_max

    def sample_curve(self):
        # 1. Draw a random magnitude for the rate (log-uniform)
        if self.abs_r_min != self.abs_r_max:
            r_mag = np.exp(np.random.uniform(np.log(self.abs_r_min), np.log(self.abs_r_max)))
        else:
            r_mag = 0.
            
        # 2. Randomly assign a positive (growth) or negative (decay) sign
        sign = np.random.choice([-1.0, 1.0])
        r = r_mag * sign
        
        # 3. Looking backwards in time, the log-size scales linearly
        raw_log_shape = -r * self.t_math
        
        return self._scale_and_bound(raw_log_shape)
    
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
        'UniformFloatDiscrete' : UniformFloatDiscrete,
        'np' : np,
        
        'ChebyshevHistory' : ChebyshevHistory,
        'ExponentialHistory' : ExponentialHistory,
        'TruncatedExponential' : TruncatedExponential
    }

    priors = {'base': {}, 'samples': {}, 'migration' : {}, 'discoal' : {}, 'demography' : {}, 'sweep' : {}}
    
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
    
    if 'sweep' in config:
        for key, value_str in config.items('sweep'):
            priors['sweep'][key] = _parse_prior_value(value_str, safe_globals)
    else:
        priors['sweep'] = None
        
    return priors
        
import msprime
import numpy as np
import subprocess
import re

# Note: create_prior_from_config and newick_to_tree are assumed to be defined elsewhere in your module.

class BaseSimulator:
    """
    A base simulator class that loads parameters and validated sample population priors
    directly from a configuration file.

    This class serves as the foundation for specific simulation engines (like msprime 
    or discoal), handling the boilerplate of parsing demographics, sample sizes, and 
    mutation/recombination rates.
    """
    def __init__(self, config_path: str, seed=None):
        """
        Initializes the BaseSimulator from a configuration file.

        Args:
            config_path (str): The path to the .ini configuration file.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Raises:
            KeyError: If a required key (e.g., 'mu', 'r', 'l', 'ploidy') is missing.
            TypeError: If a value has an incorrect type (e.g., L or ploidy not integers).
            ValueError: If a value is out of the allowed range (e.g., ploidy not in [1, 2]).
        """
        self.seed = seed
        self.config_path = config_path
        
        self._instant()
        
    # instantiate prior (need to run each time)
    def _instant(self):
        # --- Create priors from the config file ---
        priors = create_prior_from_config(self.config_path)
        base_priors = priors['base']
        sample_priors = priors['samples']
        self.migration_priors = priors['migration']
        self.discoal_priors = priors['discoal']
        self.demography_priors = priors['demography']
        self.sweep_priors = priors['sweep']

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

        if ploidy_val not in [1, 2]:
            raise ValueError(f"The value for 'ploidy' given ({ploidy_val}) is not in [1, 2]...")
    
        self.mu = base_priors['mu']
        self.r = base_priors['r']
        self.L = L_val
        self.ploidy = ploidy_val
        
        # --- Validate and store sample priors ---
        self.samples = {}
        required_sample_keys = ['n']
        
        for pop_name, pop_priors in sample_priors.items():            
            # Ensure basic required keys exist for each population
            for key in required_sample_keys:
                if key not in pop_priors:
                    raise KeyError(f"Required key '{key}' not found in priors for sample '{pop_name}'.")
            
            # Ensure population size information is provided (either current N0 or trajectory Nt)
            if 'N0' not in pop_priors and 'Nt' not in pop_priors:
                raise KeyError(f"Either 'N0' or 'Nt' must be specified for sample '{pop_name}'.")

            # Validate that sample size 'n' is a valid positive integer
            if not isinstance(pop_priors['n'], int):
                raise TypeError(f"'n' for sample '{pop_name}' must be an integer > 0.")
            if pop_priors['n'] < 0:
                raise TypeError(f"'n' for sample '{pop_name}' must be an integer >= 0.")
            
            # Store validated priors
            self.samples[pop_name] = pop_priors
            
        # to store parameter values from make_demography:
        self.params = {}

    def set_seed(self, seed: int):
        """
        Sets the random seed for the simulator.

        Args:
            seed (int): The seed value to ensure reproducible simulations.
        """
        self.seed = seed
            

class MSPrimeSimulator(BaseSimulator):
    """
    A simulator engine utilizing the `msprime` library for coalescent simulation.
    
    Inherits from BaseSimulator to parse parameters, and implements msprime-specific
    methods to build demography, simulate ancestry, and apply mutations.
    """
    def __init__(self, config_file: str, mutation_model=msprime.BinaryMutationModel()):
        """
        Initializes the MSPrimeSimulator.

        Args:
            config_file (str): Path to the configuration file.
            mutation_model (msprime.MutationModel, optional): The mutation model to apply. 
                Defaults to msprime.BinaryMutationModel().
        """
        super().__init__(config_file)
        self.mutation_model = mutation_model
        
    def make_demography(self) -> msprime.Demography:
        """
        Constructs an msprime.Demography object based on the parsed configuration priors.

        Returns:
            msprime.Demography: The assembled demographic model containing populations,
                size changes, splits, and migration events.
                
        Raises:
            ValueError: If a population is missing both 'Nt' and 'N0' definitions.
        """
        self._instant()
        demography = msprime.Demography()
        
        # 1. Add populations and size changes
        for pop_name in self.samples.keys():
            if 'N0' in self.samples[pop_name].keys() and ('Nt' not in self.samples[pop_name].keys()):
                N0 = self.samples[pop_name]['N0']
                # If N0 is fixed
                if isinstance(N0, (float, int)):
                    demography.add_population(name=pop_name, initial_size=N0)
                # If N0 is a distribution (random variable)
                else:
                    N0 = N0.rvs(size=1)[0]
                    demography.add_population(name=pop_name, initial_size=N0)
                    
                self.params['N0'] = N0
            
            elif 'Nt' in self.samples[pop_name].keys():
                Nt = self.samples[pop_name]['Nt']
                
                # If Nt is a discrete list of (Size, Time) tuples
                if isinstance(Nt, list):
                    demography.add_population(name=pop_name, initial_size=self.samples[pop_name]['Nt'][0])
                    for N1, T in Nt:
                        demography.add_population_parameters_change(time=T, population=pop_name, initial_size=N1)
                # If Nt is a single fixed historical size
                elif isinstance(Nt, (int, float)):
                    demography.add_population(name=pop_name, initial_size=Nt)
                # If Nt is a continuous curve distribution
                else:
                    t, N = Nt.sample_curve()
                    demography.add_population(name=pop_name, initial_size=N[0])
                    for N1, T in zip(N, t):
                        demography.add_population_parameters_change(time=T, population=pop_name, initial_size=N1)
                    
                    self.params['Nt'] = (N, t)
            else:
                raise ValueError("All simulated populations must have a key 'Nt' or 'N0'")
        
        # 2. Add migration events
        if self.migration_priors:
            for key in self.migration_priors:
                src, dst = key.split(',')
                m = self.migration_priors[key]
                
                if isinstance(m, list):
                    for m_, t_ in m:
                        demography.add_migration_rate_change(time=t_, source=src, dest=dst, rate=m_)
                elif isinstance(m, float):
                    demography.add_migration_rate_change(time=0., source=src, dest=dst, rate=m)
                else:
                    T, M = m.sample_curve()
                    for m_, t_ in zip(M, T):
                        demography.add_migration_rate_change(time=t_, source=src, dest=dst, rate=m_)
        
        # 3. Add population splits (demography priors)
        if self.demography_priors:
            for key in self.demography_priors:
                c1, c2, p = key.split(',')
                T = self.demography_priors[key]
                
                # Check if T is a fixed float or a random variable
                if isinstance(T, float): # Fixed original bug: replaced isinstance(m, float) with T
                    demography.add_population_split(time=T, derived=[c1, c2], ancestral=p)
                else:
                    T = T.rvs(size=1)[0]
                    demography.add_population_split(time=T, derived=[c1, c2], ancestral=p)
        
        # Sort events chronologically to satisfy msprime requirements
        demography.sort_events()
        return demography
                    
    def simulate(self, verbose: bool = False, seeds: tuple = (None, None)) -> dict:
        """
        Executes the coalescent ancestry simulation using msprime.

        Args:
            verbose (bool, optional): If True, prints additional logging. Defaults to False.

        Returns:
            dict: A dictionary containing the genotype matrix ('x'), variant positions ('pos'), 
                and the msprime tree sequence ('ts').
        """
        self.params = {}
        
        self.demography = self.make_demography()
                
        # Prepare sample sizes
        samples = {}
        for pop in self.samples.keys():
            samples[pop] = self.samples[pop]['n']
            
        # experimental feature...
        ancestry_model = None
        if self.sweep_priors:
            sweep_kwargs = {}
            for key, val in self.sweep_priors.items():
                # Check if the parameter is a scipy.stats distribution
                if hasattr(val, 'rvs'):
                    drawn_val = val.rvs(size=1)[0]
                    sweep_kwargs[key] = drawn_val
                    self.params[f'sweep_{key}'] = drawn_val
                else:
                    sweep_kwargs[key] = val
                    self.params[f'sweep_{key}'] = val
            
            # Combine the sweep model with the standard coalescent
            sweep_model = msprime.SweepGenicSelection(**sweep_kwargs)
            ancestry_model = [sweep_model, msprime.StandardCoalescent()]
                    
        # Resolve recombination rate (fixed or sampled)
        if isinstance(self.r, float):
            r = self.r
        else:
            r = self.r.rvs(size=1)[0]
            # add to the dictionary if randomly drawn
            self.params['r'] = r
        
        # Simulate ancestry (trees)
        ts = msprime.sim_ancestry(
            samples=samples,
            sequence_length=self.L,
            recombination_rate=r,
            ploidy=self.ploidy,
            demography=self.demography,
            model=ancestry_model,
            random_seed=seeds[0]
        )
        
        ret = self.mutate_and_return_(ts, seed = seeds[1])
        ret['r'] = r
        
        return ret
    
    def mutate_and_return_(self, ts: msprime.TreeSequence, seed = None) -> dict:
        """
        Applies mutations to the generated tree sequence and formats the output.

        Args:
            ts (msprime.TreeSequence): The unmutated tree sequence from sim_ancestry.

        Returns:
            dict: The simulation results containing:
                - 'x' (np.ndarray): The genotype matrix (sites x samples).
                - 'pos' (np.ndarray): Scaled variant positions (0 to 1).
                - 'ts' (msprime.TreeSequence): The fully mutated tree sequence.
        """
        result = {}
        
        # Resolve mutation rate (fixed or sampled)
        if isinstance(self.mu, float):
            mu = self.mu
        else:
            mu = self.mu.rvs(size=1)[0]
            # add to the dictionary if randomly drawn
            self.params['mu'] = mu
            
        # Simulate mutations using a binary discrete model
        mutated_ts = msprime.sim_mutations(ts, rate=mu, model=self.mutation_model, random_seed=None)
        
        # Extract and format genotype matrix
        X = mutated_ts.genotype_matrix()
        X[X > 1] = 1 # Enforce binary constraints for multiple hits
        X = X.T

        # Extract and scale positions relative to sequence length L
        sites = [u.position for u in list(mutated_ts.sites())]
        sites = np.array(sites) / self.L 
        
        result['x'] = X
        result['pos'] = sites
        result['ts'] = mutated_ts
        result['mu'] = mu
        
        return result


class DiscoalSimulator(BaseSimulator):
    """
    A simulator engine utilizing the `discoal` command-line tool, typically used 
    for simulating selective sweeps.

    Inherits from BaseSimulator. Converts demographic and selection priors into 
    a command string, executes it via a subprocess, and parses the custom output.
    """
    def __init__(self, config_file: str):
        """
        Initializes the DiscoalSimulator and parses discoal-specific parameters.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)
        
        if self.discoal_priors is not None:
            # Selection coefficient
            self.s = self.discoal_priors.get('s', None)
            # Prior on location of selection within the sequence
            self.x = self.discoal_priors.get('x', None)
            # Additional raw arguments passed directly to the discoal CLI
            self.args = self.discoal_priors.get('args', None)
        else:
            self.s = None
            self.x = None
            self.args = None
                    
    def simulate(self, verbose: bool = False) -> dict:
        """
        Builds the discoal command string and triggers the simulation.

        Args:
            verbose (bool, optional): If True, prints the raw discoal command. Defaults to False.

        Returns:
            dict: The parsed results from the discoal output.
        """
        pops = []
        
        # 1. Parse sample priors to gather population histories
        for ix, pop_name in enumerate(sorted(self.samples.keys())):
            if 'N0' in self.samples[pop_name].keys() and ('Nt' not in self.samples[pop_name].keys()):
                N0 = self.samples[pop_name]['N0']
                if not isinstance(N0, (float, int)):
                    N0 = N0.rvs(size=1)[0]
                Nt = None
            elif 'Nt' in self.samples[pop_name].keys():
                Nt = self.samples[pop_name]['Nt']
                if not isinstance(Nt, list):
                    t, N = Nt.sample_curve()
                    Nt = list(zip(N, t))
                N0 = Nt[0][0]
            
            n = self.samples[pop_name]['n']
            if self.ploidy == 2:
                n *= 2
            
            pops.append((N0, Nt, n))
        
        # Use the first population's N0 as the reference size for scaling
        N0 = pops[0][0]
        self.N = N0
        
        # Resolve recombination and mutation rates
        r = self.r if isinstance(self.r, float) else self.r.rvs(size=1)[0]
        mu = self.mu if isinstance(self.mu, float) else self.mu.rvs(size=1)[0] 
        
        # Calculate scaled population genetic parameters
        theta = 4 * N0 * self.L * mu
        rho = 4 * N0 * self.L * r
        
        total_n = sum([u[-1] for u in pops])
        
        # 2. Construct the base discoal command
        cmd = f'discoal {total_n} 1 100001 -t {theta} -r {rho} -T'
        
        if len(pops) > 1:
            cmd += f" -p {len(pops)} " + ' '.join([str(u[-1]) for u in pops])
        
        # 3. Add population size changes scaling relative to N0
        size_strs = []
        for ix, pop in enumerate(pops):
            N0_, Nt, n = pop
            
            # Subpopulations (ix > 0) split off from the ancestral population
            if ix > 0:
                pop_size_str = f' -en 0.0 {ix} {N0_ / N0}'
            else:
                pop_size_str = ''
                
            if Nt is not None:
                for (N, t) in Nt[1:]:
                    # discoal times are scaled by 4*N0
                    pop_size_str += f' -en {t / (4 * N0)} {ix} {N / N0}'            
                    
            size_strs.append(pop_size_str)
        
        cmd = ' '.join([cmd] + size_strs)
        
        # 4. Add selection flags and raw arguments
        if self.args is not None:
            cmd = ' '.join((cmd, self.args))
        
        if self.s is not None:
            s = self.s if isinstance(self.s, float) else self.s.rvs(size=1)[0]
            # scale selection coefficient (alpha = 4*N0*s)
            cmd = ' '.join((cmd, f'-a {4 * N0 * s}'))
        
        if self.x is not None:
            x = self.x if isinstance(self.x, float) else self.x.rvs(size=1)[0]
            cmd = ' '.join((cmd, f'-x {x}'))
        
        if verbose:
            print(cmd)
            sys.stdout.flush()
            
        self.co = cmd
        
        # Execute and parse
        return self.run_and_parse_cmd_(cmd)
        
    def run_and_parse_cmd_(self, cmd_: str) -> dict:      
        """
        Executes the discoal command via a subprocess and parses its custom text output.

        Args:
            cmd_ (str): The constructed shell command to run discoal.

        Returns:
            dict: The simulation results containing:
                - 'x' (np.ndarray): Binary genotype matrix.
                - 'pos' (np.ndarray): Scaled variant positions (0 to 1).
                - 'ts' (list): List of phylogenetic trees representing local ancestry.
                - 'intervals' (list): List of positional intervals corresponding to each tree.
        """
        
        fd, out_filename = tempfile.mkstemp(dir='/tmp')
        os.close(fd)  # Close the file descriptor; os.system will handle the writing
        
        # Execute the command and redirect stdout (>) to the temporary file.
        # Note: If you want to discard stderr, append ' 2>/dev/null' to the command.
        # If you want to capture stderr to the same file, use ' > {out_filename} 2>&1'
        os.system(f"{cmd_} > {out_filename}")
        
        lines = []
        try:
            with open(out_filename, 'r') as f:
                # A list comprehension is significantly faster than a while True: readline() loop
                lines = [line.rstrip() for line in f]
        finally:
            # Clean up the temporary file
            if os.path.exists(out_filename):
                os.remove(out_filename)
            
        # delete the unnecessary lines at the top
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
        
        # parse tree sequence intervals and Newick trees
        while True:
            if len(lines) == 0:
                break
            
            line = lines[0]
            del lines[0]
                        
            if len(line) > 0:
                if line[0] == '[':
                    n_sites = re.findall(r'\[(\d+)\]', line)[0]
                    n_digits = len(n_sites)
                    n_sites = int(n_sites)
                    
                    intervals.append((l, l + n_sites))
                    l += n_sites
                    bins.append(l)
                    
                    line = line[n_digits + 2:]
    
                    tree = newick_to_tree(line, multiplier=4 * self.N)
                    trees.append(tree)
                else:
                    break
            else:
                break
                    
        # Fast-forward to segregating sites matrix
        start = 0
        while lines[start] != '//':
            start += 1

        start += 1        
        lines = lines[start:]
        #n_segsites = int(lines[0].split()[-1])
        pos = np.array(list(map(float, lines[1].split()[1:])))
        
        trees_ = []
        intervals_ = []
        n_snps = 0
        
        intervals = np.array(intervals)
                
        # Filter trees and intervals to only those containing actual SNPs
        for ix in range(len(trees)):
            l, r = intervals[ix]

            ii = np.where((pos * 100001 >= l) & (pos * 100001 < r))[0]
            n_snps += len(ii)
            
            if len(ii) > 0:
                trees_.append(trees[ix])
                l = int((self.L / 100001) * l)
                r = int((self.L / 100001) * r)
                
                intervals_.append((l, r))
                    
        # Parse binary genotype sequence
        x = []
        for line in lines[2:]:
            # convert ascii string of 0s and 1s to numpy uint8 array efficiently
            x.append(np.fromstring(line, 'u1') - ord('0'))            
        
        x = np.array(x, dtype=np.uint8)
        
        result = {}
        result['x'] = x
        result['pos'] = pos
        result['ts'] = trees_ # note: outputs a list of trees rather than an msprime.TreeSequence
        result['intervals'] = intervals_
        
        return result
                        
