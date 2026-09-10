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
    
from .histories import TargetedHistory, PiecewiseConstantHistory, ChebyshevHistory, ExponentialHistory

    
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
    r"""
    A simulator engine utilizing the ``msprime`` library for coalescent simulation.

    Inherits from :class:`BaseSimulator` to parse configuration parameters and
    implements msprime-specific routines to assemble demographic models, simulate
    ancestry under neutral or selection models, and overlay mutations.

    Parameters
    ----------
    config_file : str
        Path to the configuration file containing population priors, parameters,
        and simulation settings.
    mutation_model : msprime.MutationModel, default=msprime.BinaryMutationModel()
        The mutation model applied to the simulated ancestral tree sequence.

    Attributes
    ----------
    mutation_model : msprime.MutationModel
        Active mutation model.
    demography : msprime.Demography
        Assembled demographic model built during simulation.
    params : dict
        Tracked parameter draws instantiated for the current simulation run.
    """

    def __init__(self, config_file: str, mutation_model=msprime.BinaryMutationModel()):
        super().__init__(config_file)
        self.mutation_model = mutation_model

    def make_demography(self) -> msprime.Demography:
        r"""
        Construct an msprime demographic model from parsed configuration priors.

        Instantiates demographic parameter values (either static scalars, random
        variable draws, or sampled demographic curves) for population sizes ($N_0$,
        $N(t)$), asymmetric migration rates, and ancestral population splits.

        Returns
        -------
        msprime.Demography
            The assembled demographic model containing populations, historical size
            changes, splits, and migration events sorted in chronological order.

        Raises
        ------
        ValueError
            If any simulated population lacks both 'Nt' and 'N0' definitions.
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
                    demography.add_migration_rate_change(time=0.0, source=src, dest=dst, rate=m)
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
                if isinstance(T, float):
                    demography.add_population_split(time=T, derived=[c1, c2], ancestral=p)
                else:
                    T = T.rvs(size=1)[0]
                    demography.add_population_split(time=T, derived=[c1, c2], ancestral=p)

        # Sort events chronologically to satisfy msprime requirements
        demography.sort_events()
        return demography

    def simulate(self, verbose: bool = False, seeds: tuple = (None, None)) -> dict:
        r"""
        Execute coalescent ancestry and mutation simulation using msprime.

        Draws values from parameter distributions (such as recombination rate $r$
        and sweep parameters), constructs the demography, simulates ancestry via
        :func:`msprime.sim_ancestry`, and overlays mutations.

        Parameters
        ----------
        verbose : bool, default=False
            If True, enables additional runtime logging.
        seeds : tuple of (int or None, int or None), default=(None, None)
            RNG seeds structured as ``(ancestry_seed, mutation_seed)``.

        Returns
        -------
        dict
            Dictionary containing simulation outputs:

            - ``'x'`` (:class:`numpy.ndarray` of shape `(n_samples, n_sites)`):
              Binary haplotype genotype matrix.
            - ``'pos'`` (:class:`numpy.ndarray` of shape `(n_sites,)`):
              Variant physical positions normalized to $[0, 1]$.
            - ``'ts'`` (:class:`msprime.TreeSequence`):
              The simulated mutated tree sequence.
            - ``'r'`` (float):
              Per-base per-generation recombination rate used in this run.
            - ``'mu'`` (float):
              Per-base per-generation mutation rate used in this run.
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
            random_seed=seeds[0],
        )

        ret = self.mutate_and_return_(ts, seed=seeds[1])
        ret['r'] = r

        return ret

    def mutate_and_return_(self, ts: msprime.TreeSequence, seed=None) -> dict:
        r"""
        Apply mutations to an ancestral tree sequence and format matrix outputs.

        Samples or resolves the mutation rate $\mu$, overlays mutations via
        :func:`msprime.sim_mutations`, converts the resulting tree sequence to a
        transposed binary haplotype matrix $(n \times l)$, and scales variant
        coordinates relative to total sequence length $L$.

        Parameters
        ----------
        ts : msprime.TreeSequence
            The unmutated ancestral tree sequence produced by :func:`msprime.sim_ancestry`.
        seed : int, optional
            RNG seed for the mutation generation process.

        Returns
        -------
        dict
            Output dictionary containing:

            - ``'x'`` (:class:`numpy.ndarray`):
              Binary haplotype matrix of shape ``(n_samples, n_sites)``.
            - ``'pos'`` (:class:`numpy.ndarray`):
              Variant coordinates normalized to $[0, 1]$.
            - ``'ts'`` (:class:`msprime.TreeSequence`):
              Mutated tree sequence object.
            - ``'mu'`` (float):
              Mutation rate applied to the sequence.
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
        mutated_ts = msprime.sim_mutations(ts, rate=mu, model=self.mutation_model, random_seed=seed)

        # Extract and format genotype matrix
        X = mutated_ts.genotype_matrix()
        X[X > 1] = 1  # Enforce binary constraints for multiple hits
        X = X.T

        # Extract and scale positions relative to sequence length L
        sites = [u.position for u in list(mutated_ts.sites())]
        sites = np.array(sites) / self.L

        result['x'] = X
        result['pos'] = sites
        result['ts'] = mutated_ts
        result['mu'] = mu

        return result
    
import pyslim

class SLiMSimulator(BaseSimulator):
    """
    A forward-time simulator utilizing SLiM to model explicit background selection.
    Generates tree sequences that are compatible with the msprime mutation pipeline.
    """
    def __init__(self, config_file: str):
        super().__init__(config_file)
        # Store SLiM specific priors (e.g., DFE parameters)
        priors = create_prior_from_config(self.config_path)
        self.slim_priors = priors.get('slim', {})
        
    def write_slim_script(self, out_path: str, seed: int) -> str:
        """
        Dynamically builds a SLiM recipe configured for Background Selection (BGS).
        """
        # Extract BGS parameters with defaults
        sh = self.slim_priors.get('dfe_shape', 0.186)
        mu_s = self.slim_priors.get('dfe_mean', -0.013)
        h = self.slim_priors.get('dominance', 0.5)
        
        # We assume a single population with fixed N0 for this basic BGS template
        pop_name = list(self.samples.keys())[0]
        N0 = self.samples[pop_name].get('N0', 1000)
        
        script = f"""
        initialize() {{
            initializeTreeSeq();
            initializeMutationRate({self.mu});
            initializeRecombinationRate({self.r});
            
            // m1 mutation type: deleterious, gamma DFE for BGS
            initializeMutationType("m1", {h}, "g", {mu_s}, {sh});
            
            // g1 genomic element covers the entire sequence L
            initializeGenomicElementType("g1", m1, 1.0);
            initializeGenomicElement(g1, 0, {self.L - 1});
        }}
        
        1 early() {{
            sim.addSubpop("p1", {N0});
        }}
        
        // Run for 10 * N0 generations to ensure mutation-selection balance
        {int(10 * N0)} late() {{
            sim.treeSeqOutput("{out_path}");
            sim.simulationFinished();
        }}
        """
        return script
        
    def simulate(self, verbose: bool = False, seeds: tuple = (None, None)) -> dict:
        self.params = {}
        
        # 1. Create a temporary file to hold the SLiM output
        with tempfile.TemporaryDirectory() as tmpdir:
            out_trees = os.path.join(tmpdir, "out.trees")
            script_path = os.path.join(tmpdir, "recipe.slim")
            
            # 2. Write the dynamic SLiM script
            script_content = self.write_slim_script(out_trees, seeds[0])
            with open(script_path, "w") as f:
                f.write(script_content)
                
            # 3. Execute SLiM via subprocess
            cmd = ["slim", "-s", str(seeds[0] if seeds[0] else 42), script_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if not verbose else None)
            
            # 4. Load the resulting tree sequence using tskit
            ts = tskit.load(out_trees)
            
        # 5. Overlay neutral mutations using msprime (if desired) and format output
        # Assuming you have a mutate_and_return_ method inherited or defined similar to MSPrimeSimulator
        ret = self.mutate_and_return_(ts, seed=seeds[1])
        ret['r'] = self.r
        return ret

class DiscoalSimulator(BaseSimulator):
    r"""
    A simulator engine utilizing the ``discoal`` command-line tool.

    Inherits from :class:`BaseSimulator` to parse demographic, selection, and
    population genetic priors, convert them into command-line arguments for
    the ``discoal`` coalescent simulator (often used for selective sweeps),
    execute the simulation in a subprocess, and parse the resulting custom
    text stream into matrices and trees.

    Parameters
    ----------
    config_file : str
        Path to the YAML or dictionary configuration file containing population
        priors, parameters, and simulation settings.

    Attributes
    ----------
    s : float, scipy.stats distribution, or None
        Selection coefficient parameter $s$ or its prior distribution.
    x : float, scipy.stats distribution, or None
        Relative chromosomal location of the site under selection in $[0, 1]$.
    args : str or None
        Additional raw CLI flags passed directly to the ``discoal`` executable.
    N : float or None
        Reference diploid effective population size $N_0$ used to scale
        mutation ($\theta = 4N_0 L \mu$), recombination ($\rho = 4N_0 L r$),
        and selection ($\alpha = 4N_0 s$).
    co : str or None
        The most recent shell command executed by :meth:`simulate`.
    """

    def __init__(self, config_file: str):
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
        r"""
        Construct the discoal command string and execute the simulation.

        Resolves parameter distributions (effective sizes, recombination rate $r$,
        mutation rate $\mu$, selection coefficient $s$, and sweep site $x$),
        computes coalescent scaling factors relative to reference size $N_0$,
        constructs the command-line string, and invokes :meth:`run_and_parse_cmd_`.

        Parameters
        ----------
        verbose : bool, default=False
            If True, prints the constructed discoal shell command before execution.

        Returns
        -------
        dict
            Dictionary containing simulation outputs parsed by :meth:`run_and_parse_cmd_`:

            - ``'x'`` (:class:`numpy.ndarray` of shape `(n_samples, n_sites)`):
              Binary haplotype matrix.
            - ``'pos'`` (:class:`numpy.ndarray` of shape `(n_sites,)`):
              Relative positions of segregating sites in $[0, 1]$.
            - ``'ts'`` (list):
              List of phylogenetic tree objects spanning segregating sites.
            - ``'intervals'`` (list of tuple of int):
              Physical genomic intervals in base pairs corresponding to each tree.
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
        r"""
        Execute the discoal command in a subprocess and parse text output.

        Runs the shell command redirected into a temporary file, reads the
        Newick marginal trees and their interval lengths, reconstructs tree
        objects scaled by $4N_0$, filters intervals containing segregating
        sites, and decodes the ASCII segregating sites matrix into a binary
        genotype array.

        Parameters
        ----------
        cmd_ : str
            The complete CLI invocation string to run ``discoal``.

        Returns
        -------
        dict
            Dictionary containing parsed simulation outputs:

            - ``'x'`` (:class:`numpy.ndarray` of shape `(n_samples, n_sites)`):
              Haplotype matrix of binary alleles (0/1) as ``uint8``.
            - ``'pos'`` (:class:`numpy.ndarray` of shape `(n_sites,)`):
              Floating-point variant positions along $[0, 1]$.
            - ``'ts'`` (list):
              List of marginal genealogy tree objects for intervals with SNPs.
            - ``'intervals'`` (list of tuple of int):
              Physical genomic coordinates $[l, r)$ spanning each tree in ``'ts'``.
        """
        fd, out_filename = tempfile.mkstemp(dir='/tmp')
        os.close(fd)  # Close the file descriptor; os.system will handle the writing

        # Execute the command and redirect stdout (>) to the temporary file.
        os.system(f"{cmd_} > {out_filename}")

        lines = []
        try:
            with open(out_filename, 'r') as f:
                lines = [line.rstrip() for line in f]
        finally:
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
            x.append(np.fromstring(line, 'u1') - ord('0'))

        x = np.array(x, dtype=np.uint8)

        result = {}
        result['x'] = x
        result['pos'] = pos
        result['ts'] = trees_
        result['intervals'] = intervals_

        return result
                        
