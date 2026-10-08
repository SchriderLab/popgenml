# -*- coding: utf-8 -*-
import os
import glob
import logging
import tempfile
import subprocess
import numpy as np
import tskit 
from typing import Iterator

from .io_ import write_vcf
from .functions import harmonic_number
import importlib.resources
from .trees import PGTreeSequence


def write_vcf(alignment: np.ndarray, positions: np.ndarray, L: float, vcf_path: str) -> None:
    """
    Writes a binary alignment and positions to a VCF file.
    Vectorized for maximum I/O performance.
    """
    num_haplotypes, num_sites = alignment.shape
    is_diploid = (num_haplotypes % 2 == 0)
    num_samples = num_haplotypes // 2 if is_diploid else num_haplotypes
    
    # Map integers to VCF characters in a single C-level sweep
    align_t = alignment.T
    char_array = np.full(align_t.shape, '.', dtype='U1')
    char_array[align_t == 0] = '0'
    char_array[align_t == 1] = '1'
    
    # Phase genotypes across columns
    if is_diploid:
        char_array = char_array.reshape(num_sites, num_samples, 2)
        gt_strings = np.char.add(char_array[:, :, 0], '|')
        gt_strings = np.char.add(gt_strings, char_array[:, :, 1])
    else:
        gt_strings = char_array

    pos_strings = positions.astype(str)
    
    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID=chr1,length={int(L)}>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        
        header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        sample_names = [f"Sample_{i+1}" for i in range(num_samples)]
        f.write("\t".join(header + sample_names) + "\n")
        
        # Write Data Rows efficiently
        for i in range(num_sites):
            prefix = f"chr1\t{pos_strings[i]}\t.\tA\tT\t.\tPASS\t.\tGT\t"
            f.write(prefix + "\t".join(gt_strings[i]) + "\n")

converter_script = str(importlib.resources.files('popgenml').joinpath('scripts', 'convert_to_tskit.py'))

def singer(
    alignment: np.ndarray,
    positions: np.ndarray,
    L: float,
    mutation_rate: float,
    recomb_rate: float,
    output_prefix: str = None,
    n_est: float = None,
    n_iters: int = 100,
    thin: int = 20
) -> Iterator['PGTreeSequence']:
    """
    Runs the SINGER MCMC algorithm to infer Ancestral Recombination Graphs (ARGs) 
    from genetic alignment data, yielding the sampled topologies as tree sequences.

    This function orchestrates the complete SINGER inference pipeline:
      1. Scales relative SNP positions (if provided) to absolute positions based on L.
      2. Uses the provided effective population size (n_est) or dynamically estimates it 
         using Watterson's estimator.
      3. Writes the haplotype alignment to a temporary VCF file.
      4. Executes the SINGER binary to sample ARGs via MCMC.
      5. Runs SINGER's Python converter script to compile node files into `.trees` files.
      6. Wraps the resulting tskit TreeSequences into PGTreeSequence objects, extracting 
         marginal trees and scaling absolute genomic boundaries back to 0.0 - 1.0 proportions.

    Args:
        alignment (np.ndarray): A 2D integer array of shape (num_haplotypes, num_sites) 
            representing the genotype/haplotype matrix.
        positions (np.ndarray): A 1D array of length `num_sites` containing the SNP 
            positions. Can be relative (bounded 0.0 to 1.0) or absolute coordinates; 
            if relative, they are automatically scaled by `L`.
        L (float): The total sequence length in base pairs.
        mutation_rate (float): The per-base, per-generation mutation rate.
        recomb_rate (float): The per-base, per-generation recombination rate.
        output_prefix (str, optional): The file path prefix for SINGER's intermediate files 
            and `.trees` files. If None, all outputs are written to a temporary directory 
            and deleted when iteration completes. Defaults to None.
        n_est (float, optional): The effective population size (Ne). If None, it is 
            dynamically estimated using Watterson's estimator. Defaults to None.
        n_iters (int, optional): The number of MCMC iterations for SINGER to run. 
            Defaults to 100.
        thin (int, optional): The thinning interval for SINGER's MCMC sampling. 
            Defaults to 20.

    Yields:
        PGTreeSequence: An object containing the extracted marginal trees from a single 
            MCMC sample, with genomic intervals normalized to a 0.0 - 1.0 scale, and 
            sample lists initialized for topological distance calculations.
    """
    import os
    import glob
    import tempfile
    import subprocess
    import logging
    import tskit
    import numpy as np

    # 1. Protect sequence boundaries without shifting internal duplicates
    if positions.max() <= 1.0:
        positions = positions * L
        
    positions = np.round(positions).astype(np.int32)
    
    # Simply clip to bounds to prevent Singer's ws > 0 crash at the ends
    positions = np.clip(positions, 1, int(L) - 1)
            
    # Calculate Watterson's estimate if n_est is not explicitly provided
    if n_est is None:
        num_haplotypes, num_sites = alignment.shape
        h_n = harmonic_number(num_haplotypes)
        n_est = (num_sites / (4 * mutation_rate * L)) / h_n
    
    ratio = recomb_rate / mutation_rate
    singer_executable = "singer_master" # or "singer_master" depending on install
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_vcf_prefix = os.path.join(temp_dir, "temp_input")
        temp_vcf_path = f"{temp_vcf_prefix}.vcf"
        
        # Route SINGER outputs to a temp prefix if none was provided
        run_prefix = output_prefix if output_prefix is not None else os.path.join(temp_dir, "singer_out")
        
        write_vcf(alignment, positions, L, temp_vcf_path)
        
        singer_command = [
            singer_executable,
            "-m", str(mutation_rate),
            "-vcf", temp_vcf_prefix,
            "-output", run_prefix,
            "-ratio", str(ratio),
            "-start", "0",
            "-n", str(n_iters),
            "-thin", str(thin),
            "-Ne", str(n_est),
            "-end", str(L)
        ]
        
        try:
            subprocess.run(singer_command, check=True, capture_output=True)
            
            node_pattern = f"{run_prefix}_nodes_*.txt"
            num_nodes = len(glob.glob(node_pattern))
            
            if num_nodes == 0:
                logging.warning(f"Singer produced no node files.")
                return
            
            converter_command = [
                "python3", converter_script, 
                "-input", run_prefix,
                "-output", run_prefix,
                "-start", "0",
                "-end", str(num_nodes)
            ]
            
            subprocess.run(converter_command, check=True, capture_output=True)
            
            tree_pattern = f"{run_prefix}*.trees"
            infer_files = sorted(glob.glob(tree_pattern))
            
            for infer_file in infer_files:
                ts = tskit.load(infer_file)
                yield PGTreeSequence.from_tskit(ts)
                
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode('utf-8').strip() if e.stderr else str(e)
            logging.error(f"Singer or converter failed. Error: {error_output}")
            return

