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

import io
from dataclasses import dataclass, InitVar
from typing import Iterable, List, Iterator
import pickle

@dataclass
class PGTreeSequence:
    """
    A representation of a sequence of marginal trees mapped to SNP counts.
    Accepts standard trees on instantiation but compresses them internally 
    into native bytes to prevent OOM errors while preserving all node metadata.
    """
    trees: InitVar[Iterable[tskit.Tree]]
    snps: List[int]

    def __post_init__(self, trees: Iterable[tskit.Tree]):
        self.total_snps = sum(self.snps)
        self._tree_bytes = []
        
        # Compress each tree sequence into raw bytes using pickle.
        # This bypasses the C-level fileno requirement and safely stores it in RAM.
        for t in trees:
            self._tree_bytes.append(pickle.dumps(t.tree_sequence))
            
        if len(self._tree_bytes) != len(self.snps):
            raise ValueError("Length of 'trees' and 'snps' must be identical.")

    @classmethod
    def from_tskit(
        cls, 
        ts: tskit.TreeSequence, 
        samples_to_keep: Optional[List[int]] = None,
        positions: Optional[np.ndarray] = None 
    ) -> "PGTreeSequence":
        
        if samples_to_keep is not None:
            ts = ts.simplify(samples=samples_to_keep)

        if positions is not None:
            valid_positions = np.array(positions)
        elif ts.num_sites > 0:
            G = ts.genotype_matrix()
            valid_sites_mask = (G == 0).any(axis=1) & (G == 1).any(axis=1)
            valid_positions = ts.sites_position[valid_sites_mask]
        else:
            valid_positions = np.array([])
            
        snps = []
        for tree in ts.trees():
            left, right = tree.interval
            snps.append(int(np.sum((valid_positions > left) & (valid_positions <= right))))
            
        def generate_standardized_trees():
            for tree in ts.trees():
                tables = tskit.TableCollection(sequence_length=tree.span)
                old_to_new = {}
                
                samples = sorted(list(tree.samples()))
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
                yield new_ts.first(sample_lists=True)

        return cls(trees=generate_standardized_trees(), snps=snps)

    def _iter_site_trees(self) -> Iterator[tskit.Tree]:
        """Lazily reloads the tskit struct from pickled bytes only when iterating."""
        for t_bytes, num_snps in zip(self._tree_bytes, self.snps):
            if num_snps > 0:
                ts_reloaded = pickle.loads(t_bytes)
                tree = ts_reloaded.first(sample_lists = True)
                for _ in range(num_snps):
                    yield tree

    def save(self, filepath: str) -> None:
        """
        Saves the raw bytes instead of Newick to preserve internal node IDs.
        """
        packed_bytes = np.array(self._tree_bytes, dtype=object)
        snps_array = np.array(self.snps, dtype=np.int32)
        np.savez_compressed(filepath, tree_bytes=packed_bytes, snps=snps_array)

    @classmethod
    def load(cls, filepath: str) -> "PGTreeSequence":
        data = np.load(filepath, allow_pickle=True)
        snps = data['snps'].tolist()
        tree_bytes = data['tree_bytes'].tolist()
        
        # Bypass __init__ to avoid redundant byte re-serialization
        inst = cls.__new__(cls)
        inst.snps = snps
        inst.total_snps = sum(snps)
        inst._tree_bytes = tree_bytes
        return inst

    def _check_compatibility(self, other: "PGTreeSequence"):
        if self.total_snps != other.total_snps:
            raise ValueError("Both tree sequences must have the same total number of SNPs to compare.")

    def rf_distance(self, other: "PGTreeSequence") -> np.ndarray:
        self._check_compatibility(other)
        distances = np.zeros(self.total_snps, dtype=np.float32)
        
        for i, (t1, t2) in enumerate(zip(self._iter_site_trees(), other._iter_site_trees())):
            distances[i] = t1.rf_distance(t2)
            
        return distances.mean()

    @property
    def num_trees(self) -> int:
        return len(self._tree_bytes)

    def rf_distance(self, other: "PGTreeSequence") -> float:
        self._check_compatibility(other)
        total_dist = 0.0
        
        prev_t1, prev_t2 = None, None
        prev_dist = 0.0
        
        for t1, t2 in zip(self._iter_site_trees(), other._iter_site_trees()):
            # Only compute the metric if one of the tree topologies advanced
            if t1 is not prev_t1 or t2 is not prev_t2:
                prev_dist = t1.rf_distance(t2)
                prev_t1, prev_t2 = t1, t2
            total_dist += prev_dist
            
        return float(total_dist / self.total_snps)

    def kc_distance(self, other: "PGTreeSequence") -> float:
        self._check_compatibility(other)
        total_dist = 0.0
        
        prev_t1, prev_t2 = None, None
        prev_dist = 0.0
        
        for t1, t2 in zip(self._iter_site_trees(), other._iter_site_trees()):
            if t1 is not prev_t1 or t2 is not prev_t2:
                prev_dist = t1.kc_distance(t2)
                prev_t1, prev_t2 = t1, t2
            total_dist += prev_dist
            
        return float(total_dist / self.total_snps)

    def rms_log_coal_error(self, other: "PGTreeSequence") -> float:
        self._check_compatibility(other)
        total_error = 0.0
        
        prev_t1, prev_t2 = None, None
        prev_err = 0.0
        
        for t1, t2 in zip(self._iter_site_trees(), other._iter_site_trees()):
            if t1 is not prev_t1 or t2 is not prev_t2:
                c1 = sorted([t1.time(u) for u in t1.nodes() if t1.time(u) > 0])
                c2 = sorted([t2.time(u) for u in t2.nodes() if t2.time(u) > 0])
                
                min_len = min(len(c1), len(c2))
                if min_len > 0:
                    log_c1 = np.log(c1[:min_len])
                    log_c2 = np.log(c2[:min_len])
                    mse = np.mean((log_c1 - log_c2) ** 2)
                    prev_err = float(np.sqrt(mse))
                else:
                    prev_err = 0.0
                    
                prev_t1, prev_t2 = t1, t2
                
            total_error += prev_err
            
        return float(total_error / self.total_snps)

    def breakpoint_chamfer_distance(self, other: "PGTreeSequence") -> float:
        self._check_compatibility(other)
        
        bp1 = np.cumsum(self.snps)[:-1]
        bp2 = np.cumsum(other.snps)[:-1]
        
        if len(bp1) == 0 and len(bp2) == 0:
            return 0.0
            
        if len(bp1) == 0: bp1 = np.array([0, self.total_snps])
        if len(bp2) == 0: bp2 = np.array([0, self.total_snps])
            
        def directed_chamfer(a, b):
            # O(N log N) nearest neighbor search replacing the O(N^2) memory matrix
            idx = np.searchsorted(b, a)
            idx_left = np.clip(idx - 1, 0, len(b) - 1)
            idx_right = np.clip(idx, 0, len(b) - 1)
            
            dist_left = np.abs(a - b[idx_left])
            dist_right = np.abs(a - b[idx_right])
            
            return np.minimum(dist_left, dist_right).mean()
            
        return float((directed_chamfer(bp1, bp2) + directed_chamfer(bp2, bp1)) / 2.0)
    
    def symmetric_kl_divergence(self, other: "PGTreeSequence", epsilon: float = 1e-9) -> float:
        self._check_compatibility(other)
        total_div = 0.0

        def get_clade_branch_lengths(tree: tskit.Tree) -> dict:
            clades = {}
            for u in tree.nodes():
                if tree.parent(u) != tskit.NULL:
                    bl = tree.branch_length(u)
                    if bl > 0:
                        clade = frozenset(tree.samples(u))
                        clades[clade] = bl
            return clades

        prev_t1, prev_t2 = None, None
        prev_div = 0.0

        for t1, t2 in zip(self._iter_site_trees(), other._iter_site_trees()):
            if t1 is not prev_t1 or t2 is not prev_t2:
                c1_lengths = get_clade_branch_lengths(t1)
                c2_lengths = get_clade_branch_lengths(t2)
                
                sum1 = sum(c1_lengths.values())
                sum2 = sum(c2_lengths.values())
                
                p1_init = {c: l / sum1 for c, l in c1_lengths.items()} if sum1 > 0 else {}
                p2_init = {c: l / sum2 for c, l in c2_lengths.items()} if sum2 > 0 else {}
                
                all_clades = set(p1_init.keys()).union(set(p2_init.keys()))
                
                if not all_clades:
                    prev_div = 0.0
                else:
                    P_smoothed = np.array([p1_init.get(c, 0.0) + epsilon for c in all_clades], dtype=np.float64)
                    Q_smoothed = np.array([p2_init.get(c, 0.0) + epsilon for c in all_clades], dtype=np.float64)
                    
                    P = P_smoothed / np.sum(P_smoothed)
                    Q = Q_smoothed / np.sum(Q_smoothed)
                    
                    kl_P_Q = np.sum(P * np.log(P / Q))
                    kl_Q_P = np.sum(Q * np.log(Q / P))
                    
                    prev_div = 0.5 * (kl_P_Q + kl_Q_P)
                    
                prev_t1, prev_t2 = t1, t2
                
            total_div += prev_div
            
        return float(total_div / self.total_snps)