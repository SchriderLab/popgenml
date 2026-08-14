# -*- coding: utf-8 -*-
import os
import logging
import tempfile
import subprocess
import numpy as np
from popgenml.data import write_vcf
from popgenml.data.functions import harmonic_number

def singer(
    alignment: np.ndarray,
    positions: np.ndarray,
    L: float,
    mutation_rate: float,
    recomb_rate: float,
    singer_executable: str,
    output_prefix: str,
    n_iters: int = 100,
    thin: int = 20
) -> bool:
    """
    Writes alignment data to a temporary VCF and runs Singer.
    
    Args:
        alignment: 2D numpy array of shape (num_haplotypes, num_sites).
        positions: 1D numpy array of shape (num_sites,).
        L: Sequence length in base pairs.
        mutation_rate: The mutation rate (e.g., 1.5e-8).
        recomb_rate: The recombination rate to calculate the Singer ratio.
        singer_executable: Path to the Singer binary.
        output_prefix: Path and prefix where Singer should save its final outputs.
        n_iters: Number of iterations for Singer (default: 100).
        thin: Thinning interval for Singer (default: 20).
        
    Returns:
        True if Singer runs successfully, False otherwise.
    """
    num_haplotypes, num_sites = alignment.shape
    
    # Calculate Ne using Watterson's estimator 
    # (Matches original: N_est = (sites / (4 * mu * L)) / harmonic_number)
    h_n = harmonic_number(num_haplotypes)
    n_est = (num_sites / (4 * mutation_rate * L)) / h_n
    
    # Singer's -ratio argument is (recombination_rate / mutation_rate)
    ratio = recomb_rate / mutation_rate
    
    # Create a temporary directory that will automatically clean up when finished
    with tempfile.TemporaryDirectory() as temp_dir:
        # Singer expects a prefix without the .vcf extension
        temp_vcf_prefix = os.path.join(temp_dir, "temp_input")
        temp_vcf_path = f"{temp_vcf_prefix}.vcf"
        
        logging.info("Writing temporary VCF...")
        # Calls the write_vcf function we created previously
        write_vcf(alignment, positions, L, temp_vcf_path)
        
        command = [
            singer_executable,
            "-m", str(mutation_rate),
            "-vcf", temp_vcf_prefix,
            "-output", output_prefix,
            "-ratio", str(ratio),
            "-start", "0",
            "-n", str(n_iters),
            "-thin", str(thin),
            "-Ne", str(n_est),
            "-end", str(L)
        ]
        
        logging.info(f"Executing Singer command: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True)
            logging.info(f"Successfully finished running Singer. Outputs saved to {output_prefix}*")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Singer failed. Error: {e}")
            return False
        except FileNotFoundError:
            logging.error(f"Could not find the executable at '{singer_executable}'. Please check the path.")
            return False
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return False

