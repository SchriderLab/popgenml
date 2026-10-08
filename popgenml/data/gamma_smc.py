# -*- coding: utf-8 -*-
# Standard library imports
import os
import sys
import tempfile
import subprocess
import logging
import importlib.util
from typing import Iterator

# Third-party imports
import numpy as np
# pandas and zstandard are used dynamically by the gamma_smc reader.py script,
# so they must be installed in your environment even if not explicitly imported here.

# Local popgenml imports
from popgenml.data.trees import PGTreeSequence
from popgenml.data.functions import distmat_to_tree

def gamma_smc(
    alignment: np.ndarray,
    positions: np.ndarray,
    L: float,
    mutation_rate: float,
    recomb_rate: float,
    gamma_smc_repo_dir: str,
    output_prefix: str = None,
    executable: str = "bin/gamma_smc",
    mode: str = 'mean',
    num_samples: int = 1,
    method: str = 'average'
) -> Iterator['PGTreeSequence']:
    """
    Runs Gamma-SMC inference, parses the pairwise posterior distributions, 
    and converts the expected TMRCAs into a sequence of marginal trees via 
    hierarchical clustering, yielding PGTreeSequence objects.
    
    Args:
        alignment (np.ndarray): 2D integer array (num_haplotypes, num_sites).
        positions (np.ndarray): 1D array of SNP positions.
        L (float): Sequence length.
        mutation_rate (float): Per-base, per-generation mutation rate (unscaled).
        recomb_rate (float): Per-base, per-generation recombination rate (unscaled).
        gamma_smc_repo_dir (str): Path to the cloned gamma_smc GitHub repository.
        output_prefix (str, optional): Where to save intermediate files. Defaults to temp dir.
        executable (str): Relative or absolute path to the gamma_smc binary.
        mode (str): 'mean' to use the expected TMRCA, or 'sample' to draw from the posterior.
        num_samples (int): Number of PGTreeSequences to yield if mode='sample'.
        method (str): Linkage method for hierarchical clustering (default: 'average' / UPGMA).
        
    Yields:
        PGTreeSequence: Reconstructed sequence of marginal trees scaled from 0.0 to 1.0.
    """
    import os
    import sys
    import tempfile
    import subprocess
    import logging
    import importlib.util
    import numpy as np
    # Assumes these are in scope in your actual file:
    # from popgenml.data.functions import distmat_to_tree
    # from popgenml.data.trees import PGTreeSequence

    # 1. Position bounds preparation
    if positions.max() <= 1.0:
        positions = positions * L
    positions = np.round(positions).astype(np.int32)
    positions = np.clip(positions, 1, int(L) - 1)
    
    ratio = recomb_rate / mutation_rate
    
    if executable == "bin/gamma_smc":
        executable = os.path.join(gamma_smc_repo_dir, executable)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_vcf_prefix = os.path.join(temp_dir, "temp_input")
        temp_vcf_path = f"{temp_vcf_prefix}.vcf"
        
        run_prefix = output_prefix if output_prefix is not None else os.path.join(temp_dir, "gamma_out")
        zst_output_path = f"{run_prefix}.zst"
        
        # Write temporary VCF
        write_vcf(alignment, positions, L, temp_vcf_path)
        
        gamma_smc_cmd = [
            executable,
            "-t", str(ratio),
            "-i", temp_vcf_path,
            "-o", zst_output_path
        ]
        
        try:
            # 2. Execute Gamma-SMC
            subprocess.run(gamma_smc_cmd, check=True, capture_output=True)
            
            if not os.path.exists(zst_output_path):
                logging.warning(f"Gamma-SMC succeeded but {zst_output_path} not found.")
                return
                
            # 3. Dynamically load the 'reader.py' from the gamma_smc repository
            reader_path = os.path.join(gamma_smc_repo_dir, "src", "reader.py")
            if not os.path.exists(reader_path):
                raise FileNotFoundError(f"Cannot find reader.py at {reader_path}")
                
            spec = importlib.util.spec_from_file_location("gamma_smc_reader", reader_path)
            reader = importlib.util.module_from_spec(spec)
            sys.modules["gamma_smc_reader"] = reader
            spec.loader.exec_module(reader)
            
            # Read distributions
            alphas, betas, meta = reader.open_posteriors(zst_output_path)
            
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode('utf-8').strip() if e.stderr else str(e)
            logging.error(f"Gamma-SMC failed. Error: {error_output}")
            return

    # 4. Process the Posteriors into Genomic Intervals
    A = alphas.values if hasattr(alphas, 'values') else alphas
    B = betas.values if hasattr(betas, 'values') else betas
    
    n_sites = A.shape[0]
    
    # Calculate midpoints for interval bounds
    midpoints = (positions[:-1] + positions[1:]) / 2.0
    starts = np.concatenate(([0.0], midpoints))
    ends = np.concatenate((midpoints, [L]))
    
    # Scale absolute bounds to 0.0 - 1.0 proportions
    intervals = list(zip(starts / L, ends / L))
    
    iters = num_samples if mode == 'sample' else 1
    
    # 5. Generate and Yield Trees
    for _ in range(iters):
        trees = []
        
        # Derive local TMRCAs
        if mode == 'mean':
            tmrcas = A / B
        elif mode == 'sample':
            tmrcas = np.random.gamma(shape=A, scale=1.0/B)
        else:
            raise ValueError("mode must be 'mean' or 'sample'")
            
        # Build a topological tree for each interval
        for i in range(n_sites):
            # distmat_to_tree divides distance by 2 for the node time.
            # D must be 2 * TMRCA to yield node_time = TMRCA
            D = 2.0 * tmrcas[i]
            
            tree, _ = distmat_to_tree(D, method=method)
            trees.append(tree)
            
        yield PGTreeSequence(trees=trees, intervals=intervals)