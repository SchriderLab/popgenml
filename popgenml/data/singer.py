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


def singer(
    alignment: np.ndarray,
    positions: np.ndarray,
    L: float,
    mutation_rate: float,
    recomb_rate: float,
    singer_dir: str,
    output_prefix: str,
    n_iters: int = 100,
    thin: int = 20
) -> Iterator[tskit.TreeSequence]:
    """
    Writes alignment data to a temporary VCF, runs Singer, converts the ARGs 
    to tskit format, and yields the inferred tree sequences.
    """
    # 1. Protect sequence boundaries without shifting internal duplicates
    if positions.max() <= 1.0:
        positions = positions * L
        
    positions = np.round(positions).astype(np.int32)
    
    # Simply clip to bounds to prevent Singer's ws > 0 crash at the ends
    positions = np.clip(positions, 1, int(L) - 1)
            
    num_haplotypes, num_sites = alignment.shape
    h_n = harmonic_number(num_haplotypes)
    n_est = (num_sites / (4 * mutation_rate * L)) / h_n
    
    ratio = recomb_rate / mutation_rate
    singer_executable = os.path.join(singer_dir, "singer_master")
    converter_script = os.path.join(os.getcwd(), 'src/data/convert_to_tskit.py')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_vcf_prefix = os.path.join(temp_dir, "temp_input")
        temp_vcf_path = f"{temp_vcf_prefix}.vcf"
        
        write_vcf(alignment, positions, L, temp_vcf_path)
        
        singer_command = [
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
        
        try:
            subprocess.run(singer_command, check=True, capture_output=True)
            
            node_pattern = f"{output_prefix}_nodes_*.txt"
            num_nodes = len(glob.glob(node_pattern))
            
            if num_nodes == 0:
                logging.warning(f"Singer produced no node files.")
                return
            
            converter_command = [
                "python3", converter_script,
                "-input", output_prefix,
                "-output", output_prefix,
                "-start", "0",
                "-end", str(num_nodes)
            ]
            
            subprocess.run(converter_command, check=True, capture_output=True)
            
            tree_pattern = f"{output_prefix}*.trees"
            infer_files = sorted(glob.glob(tree_pattern))
            
            for infer_file in infer_files:
                yield tskit.load(infer_file)
                
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode('utf-8').strip() if e.stderr else str(e)
            logging.error(f"Singer or converter failed. Error: {error_output}")
            return

