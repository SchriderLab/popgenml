from dataclasses import dataclass
from typing import List
import numpy as np
import tskit

import tskit
from typing import Tuple, List, Optional

def extract_standardized_trees(
    ts: tskit.TreeSequence, 
    samples_to_keep: Optional[List[int]] = None
) -> Tuple[List[tskit.Tree], List[int]]:
    r"""
    Converts a tskit TreeSequence into a list of standardized trees and a list of SNP counts.
    
    For each marginal tree, this rebuilds the tree such that:
      1. Leaves (samples) retain IDs 0 to n-1.
      2. Internal nodes are assigned IDs n to 2n-2, sorted strictly ascending by coalescent time.
      
    Args:
        ts (tskit.TreeSequence): The input tree sequence (e.g., from msprime or SLiM).
        samples_to_keep (List[int], optional): A list of sample node IDs to retain. 
            If provided, the tree sequence is downsampled to only these leaves 
            using tskit.simplify() before extraction.
            
    Returns:
        Tuple[List[tskit.Tree], List[int]]: A tuple containing the list of standardized 
        tskit.Tree objects and a list of the number of sites (SNPs) for each tree.
    """
    # Downsample the tree sequence if a subset of samples is requested
    if samples_to_keep is not None:
        # This correctly prunes the topology and re-indexes the remaining samples 0 to k-1.
        # By default, it also filters out sites that are no longer polymorphic in the subset.
        ts = ts.simplify(samples=samples_to_keep)
        
    trees = []
    snps = []
    
    for tree in ts.trees():
        # Record the number of SNPs (sites) on this specific marginal tree
        snps.append(tree.num_sites)
        
        # Initialize a new table collection for this single tree's span
        tables = tskit.TableCollection(sequence_length=tree.span)
        old_to_new = {}
        
        # 1. Add samples (leaves) first so they get IDs 0 to n-1
        samples = list(tree.samples())
        for u in samples:
            old_to_new[u] = tables.nodes.add_row(time=tree.time(u), flags=tskit.NODE_IS_SAMPLE)
            
        # 2. Get all internal (non-sample) nodes and sort them by coalescent time
        internal_nodes = [u for u in tree.nodes() if not tree.is_sample(u)]
        internal_nodes.sort(key=lambda u: tree.time(u))
        
        # Add internal nodes to tables, which automatically assigns them ascending IDs (n, n+1, ...)
        for u in internal_nodes:
            old_to_new[u] = tables.nodes.add_row(time=tree.time(u))
            
        # 3. Reconstruct the edges using the new node IDs
        for u in tree.nodes():
            for c in tree.children(u):
                tables.edges.add_row(
                    left=0, 
                    right=tree.span, 
                    parent=old_to_new[u], 
                    child=old_to_new[c]
                )
                
        # 4. Sort tables and generate the new standardized tree
        tables.sort()
        new_ts = tables.tree_sequence()
        trees.append(new_ts.first())
        
    return trees, snps

@dataclass
class PGTreeSequence:
    r"""
    A representation of a sequence of marginal trees mapped to SNP counts.

    This class provides tools to compare two different tree sequences (e.g., 
    predicted vs. true simulated trees) site-by-site using topological 
    metrics and coalescent time errors, as well as structural breakpoint distances.

    Attributes:
        trees (List[tskit.Tree]): A list of marginal trees.
        snps (List[int]): A list of the number of SNPs corresponding to each tree.
    """
    trees: List[tskit.Tree]
    snps: List[int]
    
    def __post_init__(self):
        r"""
        Validates the initialization parameters and computes the total sequence length.
        
        Raises:
            ValueError: If the number of trees does not match the number of SNP intervals.
        """
        if len(self.trees) != len(self.snps):
            raise ValueError("Length of 'trees' and 'snps' must be identical.")
        self.total_snps = sum(self.snps)
        
    @classmethod
    def from_tskit(
        cls, 
        ts: tskit.TreeSequence, 
        samples_to_keep: Optional[List[int]] = None,
        positions: Optional[np.ndarray] = None 
    ) -> "PGTreeSequence":
        
        if samples_to_keep is not None:
            ts = ts.simplify(samples=samples_to_keep)
            
        # ---------------------------------------------------------
        # Use provided positions, or fallback to the tree sequence's sites
        # ---------------------------------------------------------
        if positions is not None:
            # Assume the user provided the already-filtered valid positions
            valid_positions = np.array(positions)
        elif ts.num_sites > 0:
            G = ts.genotype_matrix()
            is_ref = (G == 0)
            is_alt = (G == 1)
            valid_sites_mask = is_ref.any(axis=1) & is_alt.any(axis=1)
            valid_positions = ts.sites_position[valid_sites_mask]
        else:
            valid_positions = np.array([])
            
        trees = []
        snps = []
        
        for tree in ts.trees():
            # Count the exact number of original SNPs falling in this physical interval
            left, right = tree.interval
            num_valid = int(np.sum((valid_positions >= left) & (valid_positions < right)))
            snps.append(num_valid)
            
            tables = tskit.TableCollection(sequence_length=tree.span)
            old_to_new = {}
            
            samples = sorted(list(tree.samples())) # Prevent leaf-shuffling bug
            for u in samples:
                old_to_new[u] = tables.nodes.add_row(time=tree.time(u), flags=tskit.NODE_IS_SAMPLE)
                
            internal_nodes = [u for u in tree.nodes() if not tree.is_sample(u)]
            internal_nodes.sort(key=lambda u: tree.time(u))
            
            for u in internal_nodes:
                old_to_new[u] = tables.nodes.add_row(time=tree.time(u))
                
            for u in tree.nodes():
                for c in tree.children(u):
                    tables.edges.add_row(
                        left=0, right=tree.span, 
                        parent=old_to_new[u], child=old_to_new[c]
                    )
                    
            tables.sort()
            new_ts = tables.tree_sequence()
            trees.append(new_ts.first(sample_lists = True))
            
        return cls(trees=trees, snps=snps)
    
    def save(self, filepath: str) -> None:
        r"""
        Saves the PGTreeSequence to a concise, compressed disk format (.npz).
        
        Serializes the marginal trees into high-precision Newick strings to preserve 
        coalescent times and topologies, and stores the SNP counts as a native array.

        Args:
            filepath (str): The destination path (e.g., 'model_output.npz').
        """
        # precision=14 prevents floating-point truncation of coalescent times
        newicks = np.array([tree.newick(precision=14) for tree in self.trees], dtype=str)
        snps_array = np.array(self.snps, dtype=np.int32)
        
        np.savez_compressed(filepath, newicks=newicks, snps=snps_array)

    @classmethod
    def load(cls, filepath: str) -> "PGTreeSequence":
        r"""
        Loads a PGTreeSequence from a compressed .npz archive.
        
        Reads the high-precision Newick strings, converts them back to tskit.Tree 
        objects, and strictly re-standardizes the node IDs (leaves 0 to n-1, 
        internal nodes ascending by time) to guarantee metric compatibility.

        Args:
            filepath (str): The path to the saved .npz file.

        Returns:
            PGTreeSequence: The restored tree sequence object.
        """
        data = np.load(filepath)
        snps = data['snps'].tolist()
        newicks = data['newicks']
        
        trees = []
        
        for nw_str in newicks:
            # Parse Newick back into a temporary tskit tree
            raw_tree = tskit.load_newick(nw_str).first()
            
            # Re-standardize the parsed tree to strictly match the PGTreeSequence layout
            tables = tskit.TableCollection(sequence_length=1.0)
            old_to_new = {}
            
            samples = sorted(list(raw_tree.samples()))
            for u in samples:
                old_to_new[u] = tables.nodes.add_row(time=raw_tree.time(u), flags=tskit.NODE_IS_SAMPLE)
                
            internal_nodes = [u for u in raw_tree.nodes() if not raw_tree.is_sample(u)]
            internal_nodes.sort(key=lambda u: raw_tree.time(u))
            
            for u in internal_nodes:
                old_to_new[u] = tables.nodes.add_row(time=raw_tree.time(u))
                
            for u in raw_tree.nodes():
                for child in raw_tree.children(u):
                    tables.edges.add_row(
                        left=0.0, right=1.0, 
                        parent=old_to_new[u], child=old_to_new[child]
                    )
                    
            tables.sort()
            clean_ts = tables.tree_sequence()
            trees.append(clean_ts.first(sample_lists=True))
            
        return cls(trees=trees, snps=snps)

    def _iter_site_trees(self):
        r"""
        Generates the corresponding marginal tree for each individual SNP site.

        This flattens the chunked tree representations into a continuous 
        site-by-site generator, naturally handling comparisons between 
        tree sequences with mismatched internal breakpoints.

        Yields:
            tskit.Tree: The marginal tree at the current SNP site.
        """
        for tree, num_snps in zip(self.trees, self.snps):
            for _ in range(num_snps):
                yield tree

    def _check_compatibility(self, other: "PGTreeSequence"):
        r"""
        Ensures both sequences span the same total number of SNPs.

        Args:
            other (PGTreeSequence): The other tree sequence to compare against.

        Raises:
            ValueError: If the total number of SNPs differs between the two sequences.
        """
        if self.total_snps != other.total_snps:
            raise ValueError("Both tree sequences must have the same total number of SNPs to compare.")

    def rf_distance(self, other: "PGTreeSequence") -> np.ndarray:
        r"""
        Computes the Robinson-Foulds topological distance site-by-site.

        Args:
            other (PGTreeSequence): The tree sequence to compare against.

        Returns:
            np.ndarray: A 1D float32 array of length `total_snps` containing 
            the RF distance at each site.
        """
        self._check_compatibility(other)
        distances = np.zeros(self.total_snps, dtype=np.float32)
        
        for i, (t1, t2) in enumerate(zip(self._iter_site_trees(), other._iter_site_trees())):
            distances[i] = t1.rf_distance(t2)
            
        return distances.mean()

    def kc_distance(self, other: "PGTreeSequence") -> np.ndarray:
        r"""
        Computes the Kendall-Colijn topological distance site-by-site.

        Args:
            other (PGTreeSequence): The tree sequence to compare against.

        Returns:
            np.ndarray: A 1D float32 array of length `total_snps` containing 
            the KC distance at each site.
        """
        self._check_compatibility(other)
        distances = np.zeros(self.total_snps, dtype=np.float32)
        
        for i, (t1, t2) in enumerate(zip(self._iter_site_trees(), other._iter_site_trees())):
            distances[i] = t1.kc_distance(t2)
            
        return distances.mean()

    def rms_log_coal_error(self, other: "PGTreeSequence") -> np.ndarray:
        r"""
        Computes the Root Mean Square (RMS) error of log coalescent times site-by-site.

        For each site, this extracts the strictly positive node coalescent times 
        from both trees, sorts them, aligns them up to the minimum length, and 
        computes the RMS error of their log differences.

        Args:
            other (PGTreeSequence): The tree sequence to compare against.

        Returns:
            np.ndarray: A 1D float32 array of length `total_snps` containing 
            the RMS log coalescent time error at each site.
        """
        self._check_compatibility(other)
        errors = np.zeros(self.total_snps, dtype=np.float32)
        
        for i, (t1, t2) in enumerate(zip(self._iter_site_trees(), other._iter_site_trees())):
            c1 = sorted([t1.time(u) for u in t1.nodes() if t1.time(u) > 0])
            c2 = sorted([t2.time(u) for u in t2.nodes() if t2.time(u) > 0])
            
            min_len = min(len(c1), len(c2))
            if min_len > 0:
                log_c1 = np.log(c1[:min_len])
                log_c2 = np.log(c2[:min_len])
                
                mse = np.mean((log_c1 - log_c2) ** 2)
                errors[i] = np.sqrt(mse)
            else:
                errors[i] = 0.0
                
        return errors.mean()

    def breakpoint_chamfer_distance(self, other: "PGTreeSequence") -> float:
        r"""
        Computes the symmetric Chamfer distance between the internal breakpoints 
        of the two tree sequences.

        A breakpoint is defined as the cumulative sum of SNPs at the boundary 
        between two marginal trees. The symmetric Chamfer distance calculates the 
        average distance from each breakpoint in Sequence 1 to its nearest neighbor 
        in Sequence 2, plus the same from Sequence 2 to Sequence 1, divided by 2.

        If a sequence has no internal breakpoints, the sequence boundaries 
        (0 and total_snps) are used as fallback reference points.

        Args:
            other (PGTreeSequence): The tree sequence to compare against.

        Returns:
            float: The symmetric mean Chamfer distance in SNP units.
        """
        self._check_compatibility(other)
        
        # Calculate internal breakpoints (excluding the final boundary)
        bp1 = np.cumsum(self.snps)[:-1]
        bp2 = np.cumsum(other.snps)[:-1]
        
        if len(bp1) == 0 and len(bp2) == 0:
            return 0.0
            
        # Fallback for sequences with no internal breakpoints:
        # Distance to the nearest sequence boundary
        if len(bp1) == 0:
            bp1 = np.array([0, self.total_snps])
        if len(bp2) == 0:
            bp2 = np.array([0, self.total_snps])
            
        # Compute pairwise absolute differences using broadcasting
        diffs = np.abs(bp1[:, np.newaxis] - bp2[np.newaxis, :])
        
        # Find the minimum distance from each breakpoint to the other set
        min_bp1_to_bp2 = np.min(diffs, axis=1)
        min_bp2_to_bp1 = np.min(diffs, axis=0)
        
        # Return symmetric mean Chamfer distance
        return float((np.mean(min_bp1_to_bp2) + np.mean(min_bp2_to_bp1)) / 2.0)
    
    def symmetric_kl_divergence(self, other: "PGTreeSequence", epsilon: float = 1e-9) -> np.ndarray:
        r"""
        Computes the symmetric Kullback-Leibler (KL) divergence of implied site 
        pattern distributions site-by-site.

        Each branch defines a clade $c$. The probability of observing a mutation 
        on that branch is modeled as $p(c) = \frac{l(c)}{\sum_{k} l(k)}$.
        Additive smoothing ($\epsilon$) is applied to the union of all clades 
        $C = C(T_1) \cup C(T_2)$ to avoid undefined logarithms, yielding smoothed 
        distributions $P$ and $Q$.
        
        The metric returns:
        $$D_{sKL}(T_1, T_2) = \frac{1}{2} \left( \sum_{c \in C} P(c) \log\left(\frac{P(c)}{Q(c)}\right) + \sum_{c \in C} Q(c) \log\left(\frac{Q(c)}{P(c)}\right) \right)$$

        Args:
            other (PGTreeSequence): The tree sequence to compare against.
            epsilon (float): The small constant used for additive smoothing.

        Returns:
            np.ndarray: A 1D float32 array containing the symmetric KL divergence at each site.
        """
        self._check_compatibility(other)
        divergences = np.zeros(self.total_snps, dtype=np.float32)

        def get_clade_branch_lengths(tree: tskit.Tree) -> dict:
            clades = {}
            for u in tree.nodes():
                # Ignore the root node itself as it doesn't have a branch above it
                if tree.parent(u) != tskit.NULL:
                    bl = tree.branch_length(u)
                    if bl > 0:
                        clade = frozenset(tree.samples(u))
                        clades[clade] = bl
            return clades

        for i, (t1, t2) in enumerate(zip(self._iter_site_trees(), other._iter_site_trees())):
            c1_lengths = get_clade_branch_lengths(t1)
            c2_lengths = get_clade_branch_lengths(t2)
            
            sum1 = sum(c1_lengths.values())
            sum2 = sum(c2_lengths.values())
            
            p1_init = {c: l / sum1 for c, l in c1_lengths.items()} if sum1 > 0 else {}
            p2_init = {c: l / sum2 for c, l in c2_lengths.items()} if sum2 > 0 else {}
            
            all_clades = set(p1_init.keys()).union(set(p2_init.keys()))
            
            if not all_clades:
                divergences[i] = 0.0
                continue
                
            # Apply additive smoothing and construct arrays
            P_smoothed = np.array([p1_init.get(c, 0.0) + epsilon for c in all_clades], dtype=np.float64)
            Q_smoothed = np.array([p2_init.get(c, 0.0) + epsilon for c in all_clades], dtype=np.float64)
            
            # Re-normalize to sum to 1
            P = P_smoothed / np.sum(P_smoothed)
            Q = Q_smoothed / np.sum(Q_smoothed)
            
            # Compute symmetric KL divergence
            kl_P_Q = np.sum(P * np.log(P / Q))
            kl_Q_P = np.sum(Q * np.log(Q / P))
            
            divergences[i] = 0.5 * (kl_P_Q + kl_Q_P)
            
        return divergences.mean()