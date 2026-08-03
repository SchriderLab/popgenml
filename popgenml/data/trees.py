# -*- coding: utf-8 -*-
import itertools
import numpy as np
import tskit
from dataclasses import dataclass
from typing import List, Iterable

@dataclass
class PGTreeSequence:
    """
    A representation of a sequence of marginal trees mapped to segregating sites.

    This class wraps a list of `tskit.Tree` objects and aligns them with a 
    corresponding list of segregating site counts. It provides methods to 
    unroll these compressed marginal trees back into a site-by-site format, 
    and allows for tree topology and breakpoint comparisons against other 
    `PGTreeSequence` instances.

    Attributes:
        trees (List[tskit.Tree]): A list of sequential marginal trees.
        segregating_sites (List[int]): The number of segregating sites associated 
            with each tree. The length of this list must exactly match the length 
            of `trees`.
    """
    trees: List[tskit.Tree]
    segregating_sites: List[int]
    
    def __post_init__(self):
        if len(self.trees) != len(self.segregating_sites):
            raise ValueError("Length of 'trees' and 'segregating_sites' must be identical.")

    def iter_site_trees(self) -> Iterable[tskit.Tree]:
        """
        Unrolls the compressed marginal trees into a site-by-site generator.

        Yields:
            tskit.Tree: The corresponding marginal tree for every individual 
            segregating site.
        """
        for tree, num_sites in zip(self.trees, self.segregating_sites):
            for _ in range(num_sites):
                yield tree

    def _validate_comparison(self, other: 'PGTreeSequence'):
        """Ensures that two tree sequences are comparable."""
        if sum(self.segregating_sites) != sum(other.segregating_sites):
            raise ValueError("Both PGTreeSequence objects must have the same total number of segregating sites.")

    def site_by_site_kc_distance(self, other: 'PGTreeSequence') -> List[float]:
        """
        Computes the site-by-site Kendall-Colijn (KC) distance.

        Unrolls both tree sequences to a site-by-site level and computes the 
        KC distance between corresponding trees.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.

        Returns:
            List[float]: A list of KC distances for each segregating site.
        """
        self._validate_comparison(other)
        return [
            t1.kc_distance(t2) 
            for t1, t2 in zip(self.iter_site_trees(), other.iter_site_trees())
        ]

    def site_by_site_rf_distance(self, other: 'PGTreeSequence') -> List[float]:
        """
        Computes the site-by-site unweighted Robinson-Foulds (RF) distance.

        Unrolls both tree sequences to a site-by-site level and computes the 
        RF distance between corresponding trees using tskit's native method.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.

        Returns:
            List[float]: A list of unweighted RF distances for each segregating site.
        """
        self._validate_comparison(other)
        return [
            t1.rf_distance(t2) 
            for t1, t2 in zip(self.iter_site_trees(), other.iter_site_trees())
        ]

    def site_by_site_rms_log_coal_time(self, other: 'PGTreeSequence', epsilon: float = 1e-8) -> List[float]:
        """
        Computes the site-by-site root-mean-square (RMS) difference of log 
        coalescent times.

        For every corresponding site, calculates the RMS of the log differences 
        in the Time to Most Recent Common Ancestor (TMRCA) across all pairs of 
        shared samples between the two trees.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.
            epsilon (float, optional): A pseudo-count to prevent log(0) domain 
                errors for zero-length branches. Defaults to 1e-8.

        Returns:
            List[float]: A list of RMS log TMRCA differences for each segregating site.
        """
        self._validate_comparison(other)
        return [
            self._calculate_rms_log_tmrca(t1, t2, epsilon) 
            for t1, t2 in zip(self.iter_site_trees(), other.iter_site_trees())
        ]

    def breakpoint_chamfer_distance(self, other: 'PGTreeSequence') -> float:
        """
        Computes the symmetric mean Chamfer distance between sequence breakpoints.

        Breakpoints are defined as the cumulative sum of segregating sites 
        (excluding the final sequence boundary). This metric calculates the 
        average distance from each breakpoint in the first sequence to the 
        nearest breakpoint in the second, plus the reverse.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.

        Returns:
            float: The computed Chamfer distance. Returns 0.0 if neither 
            sequence has breakpoints, or NaN if only one sequence lacks breakpoints.
        """
        self._validate_comparison(other)
        
        # Breakpoints are the cumulative sum of sites (excluding the final sequence boundary)
        bp1 = np.cumsum(self.segregating_sites)[:-1]
        bp2 = np.cumsum(other.segregating_sites)[:-1]
        
        # Handle cases where one or both tree sequences contain no breakpoints (only 1 tree)
        if len(bp1) == 0 and len(bp2) == 0:
            return 0.0
        if len(bp1) == 0 or len(bp2) == 0:
            return float('nan') # Distance is mathematically undefined if one set is empty
            
        def nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Finds the distance from each point in 'a' to the nearest point in 'b'."""
            idx = np.searchsorted(b, a)
            idx_left = np.clip(idx - 1, 0, len(b) - 1)
            idx_right = np.clip(idx, 0, len(b) - 1)
            
            return np.minimum(np.abs(a - b[idx_left]), np.abs(a - b[idx_right]))
            
        dist_1_to_2 = nearest_distances(bp1, bp2).mean()
        dist_2_to_1 = nearest_distances(bp2, bp1).mean()
        
        return float(dist_1_to_2 + dist_2_to_1)

    @staticmethod
    def _calculate_rms_log_tmrca(t1: tskit.Tree, t2: tskit.Tree, epsilon: float) -> float:
        """Helper method to compute the RMS log TMRCA for two individual trees."""
        common_samples = list(set(t1.samples()).intersection(t2.samples()))
        
        # If there are fewer than 2 common samples, we cannot compute pairwise TMRCA
        if len(common_samples) < 2:
            return 0.0
            
        sq_diff_sum = 0.0
        count = 0
        
        for u, v in itertools.combinations(common_samples, 2):
            mrca1 = t1.mrca(u, v)
            mrca2 = t2.mrca(u, v)
            
            # Extract times, using max() to prevent log(0) domain errors 
            time1 = max(t1.time(mrca1), epsilon) if mrca1 != tskit.NULL else epsilon
            time2 = max(t2.time(mrca2), epsilon) if mrca2 != tskit.NULL else epsilon
            
            diff = np.log(time1) - np.log(time2)
            sq_diff_sum += diff ** 2
            count += 1
            
        return np.sqrt(sq_diff_sum / count) if count > 0 else 0.0
    
    def coalescent_time_histogram(self, bins=50, time_range: tuple = None) -> tuple:
        """
        Computes the histogram of coalescent times across the tree sequence.

        This extracts the times of all internal nodes (coalescent events) 
        across all marginal trees. The contribution of each tree's coalescent 
        times is weighted by its number of segregating sites.

        Args:
            bins (int or sequence of scalars, optional): The number of bins or 
                an array of bin edges. Defaults to 50.
            time_range (tuple, optional): The lower and upper range of the bins 
                (min_time, max_time). If not provided, it defaults to the 
                (min, max) of the extracted times.

        Returns:
            tuple: A tuple (counts, bin_edges) identical to numpy.histogram.
                - counts (np.ndarray): The weighted frequency of coalescent events.
                - bin_edges (np.ndarray): The edges of the bins.
        """
        times = []
        weights = []
        
        for tree, num_sites in zip(self.trees, self.segregating_sites):
            # Skip trees that don't cover any segregating sites
            if num_sites == 0:
                continue
                
            # Extract times for all internal nodes (coalescent events)
            for u in tree.nodes():
                if tree.is_internal(u):
                    times.append(tree.time(u))
                    weights.append(num_sites)
                    
        if not times:
            # Return empty histogram structure if no internal nodes exist
            return np.histogram([], bins=bins, range=time_range)
            
        return np.histogram(times, bins=bins, range=time_range, weights=weights)
    
    @classmethod
    def from_tskit(cls, ts: tskit.TreeSequence) -> 'PGTreeSequence':
        """
        Creates a PGTreeSequence directly from a standard tskit.TreeSequence.

        Extracts the marginal trees and the number of segregating sites 
        (tskit sites) associated with each tree's genomic interval.

        Args:
            ts (tskit.TreeSequence): The input tskit tree sequence.

        Returns:
            PGTreeSequence: A new instance populated with the marginal trees 
            and their corresponding segregating site counts.
        """
        # Safely extract all marginal trees as independent objects
        trees = ts.aslist()
        
        # Count the number of sites falling within each tree's genomic span
        segregating_sites = [tree.num_sites for tree in trees]
        
        return cls(trees=trees, segregating_sites=segregating_sites)
    
    def simulate_sfs(self, mutation_rate: float, return_expected: bool = False) -> np.ndarray:
        """
        Simulates an unfolded Site Frequency Spectrum (SFS) for the tree sequence.

        This analytically calculates the expected number of mutations for each 
        derived allele frequency based on branch lengths and genomic span, and 
        then draws the simulated counts from a Poisson distribution.

        Args:
            mutation_rate (float): The mutation rate per base pair per generation.
            return_expected (bool, optional): If True, returns the continuous 
                expected SFS without applying stochastic Poisson sampling. 
                Defaults to False.

        Returns:
            np.ndarray: A 1D array of size (n_samples + 1) where the index `k` 
            represents the count of sites with exactly `k` derived alleles. 
            Indices 0 and n_samples will be 0.
        """
        if not self.trees:
            return np.array([])
            
        # Total number of samples in the trees
        n_samples = self.trees[0].num_samples
        
        # Initialize an array to hold the expected SFS (size n+1 so index matches frequency)
        expected_sfs = np.zeros(n_samples + 1)
        
        for tree in self.trees:
            span = tree.span
            if span == 0:
                continue
                
            for u in tree.nodes():
                parent = tree.parent(u)
                
                # Exclude the root (it has no parent branch)
                if parent != tskit.NULL:
                    branch_length = max(tree.time(parent) - tree.time(u), 0.0)
                    
                    # Number of sample leaves subtended by this branch
                    # len(tree.samples(u)) is very fast in tskit (O(1) slice)
                    k = len(tree.samples(u))
                    
                    # Expected mutations: rate * branch_length * genomic_span
                    expected_sfs[k] += mutation_rate * branch_length * span
                    
        if return_expected:
            return expected_sfs
        else:
            # Draw actual simulated mutation counts from a Poisson distribution
            return np.random.poisson(expected_sfs)
