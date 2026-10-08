# -*- coding: utf-8 -*-
import itertools
import numpy as np
import tskit
from dataclasses import dataclass
from typing import List, Iterable, Tuple

@dataclass
class PGTreeSequence:
    """
    A representation of a sequence of marginal trees mapped to genomic intervals.

    This class wraps a list of `tskit.Tree` objects and aligns them with a 
    corresponding list of scaled genomic intervals (0.0 to 1.0). It provides methods 
    to compare tree topologies, breakpoints, and coalescent times against other 
    `PGTreeSequence` instances by computing interval-weighted averages.

    Attributes:
        trees (List[tskit.Tree]): A list of sequential marginal trees.
        intervals (List[Tuple[float, float]]): The [left, right] bounding positions 
            on the chromosome for each tree, scaled from 0.0 to 1.0. The length 
            of this list must exactly match the length of `trees`.
    """
    trees: List[tskit.Tree]
    intervals: List[Tuple[float, float]]
    
    def __post_init__(self):
        if len(self.trees) != len(self.intervals):
            raise ValueError("Length of 'trees' and 'intervals' must be identical.")

    def iter_overlapping_intervals(self, other: 'PGTreeSequence') -> Iterable[Tuple[tskit.Tree, tskit.Tree, float]]:
        """
        Yields overlapping tree pairs and the length of their overlap.

        Since two tree sequences may have different recombination breakpoints, this 
        generator computes the intersections of their intervals, returning the trees 
        that share genomic space and the weight (size) of that shared interval.

        Yields:
            Tuple[tskit.Tree, tskit.Tree, float]: Tree from self, Tree from other, 
            and the scaled length of their overlapping interval.
        """
        i, j = 0, 0
        while i < len(self.trees) and j < len(other.trees):
            t1, int1 = self.trees[i], self.intervals[i]
            t2, int2 = other.trees[j], other.intervals[j]
            
            # Find the overlapping continuous region
            left = max(int1[0], int2[0])
            right = min(int1[1], int2[1])
            
            if left < right:
                yield t1, t2, float(right - left)
            
            # Advance the pointer of the interval that ends first
            if int1[1] < int2[1]:
                i += 1
            else:
                j += 1

    def average_kc_distance(self, other: 'PGTreeSequence') -> float:
        """
        Computes the interval-weighted average Kendall-Colijn (KC) distance.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.

        Returns:
            float: The weighted average KC distance across the entire [0, 1] scaled chromosome.
        """
        return sum(
            t1.kc_distance(t2) * weight 
            for t1, t2, weight in self.iter_overlapping_intervals(other)
        )

    def average_rf_distance(self, other: 'PGTreeSequence') -> float:
        """
        Computes the interval-weighted average unweighted Robinson-Foulds (RF) distance.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.

        Returns:
            float: The weighted average RF distance across the [0, 1] scaled chromosome.
        """
        return sum(
            t1.rf_distance(t2) * weight 
            for t1, t2, weight in self.iter_overlapping_intervals(other)
        )

    def average_rms_log_coal_time(self, other: 'PGTreeSequence', epsilon: float = 1e-8) -> float:
        """
        Computes the interval-weighted root-mean-square (RMS) difference of log 
        coalescent times.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.
            epsilon (float, optional): A pseudo-count to prevent log(0) domain 
                errors for zero-length branches. Defaults to 1e-8.

        Returns:
            float: The weighted average RMS log TMRCA difference across the sequence.
        """
        return sum(
            self._calculate_rms_log_tmrca(t1, t2, epsilon) * weight 
            for t1, t2, weight in self.iter_overlapping_intervals(other)
        )

    def breakpoint_chamfer_distance(self, other: 'PGTreeSequence') -> float:
        """
        Computes the symmetric mean Chamfer distance between sequence breakpoints.

        Breakpoints are defined as the right-side boundary of each interval 
        (excluding the final sequence boundary at 1.0).

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.

        Returns:
            float: The computed Chamfer distance. Returns 0.0 if neither 
            sequence has breakpoints, or NaN if only one sequence lacks breakpoints.
        """
        # Breakpoints are the right boundaries of all intervals except the very last one
        bp1 = np.array([interval[1] for interval in self.intervals[:-1]])
        bp2 = np.array([interval[1] for interval in other.intervals[:-1]])
        
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
        across all marginal trees, weighted by the scaled genomic interval 
        span (0-1) of that tree.

        Args:
            bins (int or sequence of scalars, optional): The number of bins or 
                an array of bin edges. Defaults to 50.
            time_range (tuple, optional): The lower and upper range of the bins 
                (min_time, max_time).

        Returns:
            tuple: A tuple (counts, bin_edges) identical to numpy.histogram.
        """
        times = []
        weights = []
        
        for tree, interval in zip(self.trees, self.intervals):
            span = interval[1] - interval[0]
            if span <= 0:
                continue
                
            # Extract times for all internal nodes (coalescent events)
            for u in tree.nodes():
                if tree.is_internal(u):
                    times.append(tree.time(u))
                    weights.append(span)
                    
        if not times:
            return np.histogram([], bins=bins, range=time_range)
            
        return np.histogram(times, bins=bins, range=time_range, weights=weights)
    
    @classmethod
    def from_tskit(cls, ts: tskit.TreeSequence) -> 'PGTreeSequence':
        """
        Creates a PGTreeSequence directly from a standard tskit.TreeSequence.

        Extracts the marginal trees and scales their absolute genomic intervals 
        into relative 0.0 to 1.0 bounding proportions.

        Args:
            ts (tskit.TreeSequence): The input tskit tree sequence.

        Returns:
            PGTreeSequence: A new instance populated with the marginal trees 
            and their scaled intervals.
        """
        import tskit
        
        trees = []
        L = ts.sequence_length
        intervals = []
        
        for tree in ts.trees():
            # Build a standalone TableCollection for the marginal tree
            tables = tskit.TableCollection(sequence_length=L)
            
            # Isolate samples and internal nodes
            samples = list(tree.samples())
            internal_nodes = [u for u in tree.nodes() if u not in samples]
            
            # Rank internal nodes by their time (age)
            internal_nodes.sort(key=lambda u: tree.time(u))
            
            node_map = {}
            
            # 1. Add samples first so they are strictly numbered 0 to n-1
            for u in samples:
                node_map[u] = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=tree.time(u))
                
            # 2. Add internal nodes sequentially by rank
            for u in internal_nodes:
                node_map[u] = tables.nodes.add_row(flags=0, time=tree.time(u))
                
            # 3. Reconstruct edges using the new ranked IDs
            for u in tree.nodes():
                parent = tree.parent(u)
                if parent != tskit.NULL:
                    tables.edges.add_row(
                        left=0, right=L,
                        parent=node_map[parent], child=node_map[u]
                    )
                    
            tables.sort()
            
            # Extract the independent, properly ranked tree
            new_tree = tables.tree_sequence().first(sample_lists = True)
            trees.append(new_tree)
            
            # Scale each tree's absolute boundaries to a 0-1 proportion
            intervals.append((tree.interval[0] / L, tree.interval[1] / L))
            
        return cls(trees=trees, intervals=intervals)
    
    def simulate_sfs(self, mutation_rate: float, return_expected: bool = False) -> np.ndarray:
        """
        Simulates an unfolded Site Frequency Spectrum (SFS) for the tree sequence.

        This analytically calculates the expected number of mutations for each 
        derived allele frequency based on absolute branch lengths and genomic span, 
        and draws the simulated counts from a Poisson distribution.

        Args:
            mutation_rate (float): The mutation rate per base pair per generation.
            return_expected (bool, optional): If True, returns the continuous 
                expected SFS without applying stochastic Poisson sampling.

        Returns:
            np.ndarray: A 1D array of size (n_samples + 1)
        """
        if not self.trees:
            return np.array([])
            
        n_samples = self.trees[0].num_samples
        expected_sfs = np.zeros(n_samples + 1)
        
        for tree in self.trees:
            # We use tskit's native tree.span here as the SFS calculation 
            # requires absolute genome lengths to pair with the absolute mutation rate.
            span = tree.span
            if span == 0:
                continue
                
            for u in tree.nodes():
                parent = tree.parent(u)
                
                if parent != tskit.NULL:
                    branch_length = max(tree.time(parent) - tree.time(u), 0.0)
                    k = len(tree.samples(u))
                    
                    # Expected mutations: rate * branch_length * absolute genomic span
                    expected_sfs[k] += mutation_rate * branch_length * span
                    
        if return_expected:
            return expected_sfs
        else:
            return np.random.poisson(expected_sfs)