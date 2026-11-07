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
import sys
import re


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
    mutation_rate: float,
    recomb_rate: float,
    L: float,
    output_prefix: str = None,
    n_est: float = None,
    n_iters: int = 100,
    thin: int = 20,
    num_samples: int = 1,
) -> Iterator['PGTreeSequence']:
    r"""
    Runs the SINGER MCMC algorithm to infer Ancestral Recombination Graphs (ARGs)
    from genetic alignment data, yielding the requested number of posterior samples
    as tree sequences.

    This function orchestrates the complete SINGER inference pipeline:
      1. Scales relative SNP positions (if provided) to absolute positions based on L.
      2. Uses the provided effective population size (n_est) or dynamically estimates it
         using Watterson's estimator.
      3. Writes the haplotype alignment to a temporary VCF file.
      4. Executes the SINGER binary to sample ARGs via MCMC across `n_iters` thinned steps.
      5. Runs SINGER's Python converter script on the final `num_samples` post-burn-in
         node files to compile them into `.trees` files.
      6. Wraps the resulting tskit TreeSequences into PGTreeSequence objects, extracting
         marginal trees and scaling absolute genomic boundaries back to 0.0 - 1.0 proportions.

    Args:
        alignment (np.ndarray): A 2D integer array of shape (num_haplotypes, num_sites)
            representing the genotype/haplotype matrix.
        positions (np.ndarray): A 1D array of length `num_sites` containing the SNP
            positions. Can be relative (bounded 0.0 to 1.0) or absolute coordinates;
            if relative, they are automatically scaled by `L`.
        mutation_rate (float): The per-base, per-generation mutation rate.
        recomb_rate (float): The per-base, per-generation recombination rate.
        L (float): The total sequence length in base pairs.
        output_prefix (str, optional): The file path prefix for SINGER's intermediate files
            and `.trees` files. If None, all outputs are written to a temporary directory
            and deleted when iteration completes. Defaults to None.
        n_est (float, optional): The effective population size (Ne). If None, it is
            dynamically estimated using Watterson's estimator. Defaults to None.
        n_iters (int, optional): The total number of thinned MCMC steps (`-n`) for SINGER
            to run (including burn-in). Defaults to 100.
        thin (int, optional): The thinning interval (`-thin`, MCMC iterations between
            adjacent saved steps) for SINGER's MCMC sampling. Defaults to 20.
        num_samples (int, optional): The number of posterior ARG samples to convert and
            yield from the end of the MCMC chain (after burn-in). Defaults to 1.

    Yields:
        PGTreeSequence: An object containing the extracted marginal trees from a single
            posterior MCMC sample, with genomic intervals normalized to a 0.0 - 1.0 scale,
            and sample lists initialized for topological distance calculations.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    if n_iters < num_samples:
        logging.warning(
            f"n_iters ({n_iters}) is less than num_samples ({num_samples}); "
            f"increasing n_iters to {num_samples}."
        )
        n_iters = num_samples

    if positions.max() <= 1.0:
        positions = positions * L

    positions = np.round(positions).astype(np.int32)

    if n_est is None:
        num_haplotypes, num_sites = alignment.shape
        h_n = harmonic_number(num_haplotypes)
        n_est = (num_sites / (4 * mutation_rate * L)) / h_n

    ratio = recomb_rate / mutation_rate
    singer_executable = "singer_master"  # or "singer_master" depending on install

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_vcf_prefix = os.path.join(temp_dir, "temp_input")
        temp_vcf_path = f"{temp_vcf_prefix}.vcf"

        run_prefix = (
            output_prefix
            if output_prefix is not None
            else os.path.join(temp_dir, "singer_out")
        )

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
            "-end", str(L),
        ]

        try:
            subprocess.run(singer_command, check=True, capture_output=True)

            node_pattern = f"{run_prefix}_nodes_*.txt"
            num_nodes = len(glob.glob(node_pattern))

            if num_nodes == 0:
                logging.warning("Singer produced no node files.")
                return

            start_idx = max(0, num_nodes - num_samples)

            converter_command = [
                sys.executable,
                converter_script,
                "-input", run_prefix,
                "-output", run_prefix,
                "-start", str(start_idx),
                "-end", str(num_nodes),
            ]

            subprocess.run(converter_command, check=True, capture_output=True)

            tree_pattern = f"{run_prefix}*.trees"
            infer_files = sorted(
                glob.glob(tree_pattern),
                key=lambda p: int(re.search(r"(\d+)\.trees$", p).group(1))
                if re.search(r"(\d+)\.trees$", p)
                else p,
            )

            for infer_file in infer_files[-num_samples:]:
                ts = tskit.load(infer_file)
                yield PGTreeSequence.from_tskit(ts, positions=positions)

        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode("utf-8").strip() if e.stderr else str(e)
            logging.error(f"Singer or converter failed. Error: {error_output}")
            return

