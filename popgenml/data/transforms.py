# -*- coding: utf-8 -*-
import numpy as np
import inspect
import warnings
from popgenml.data.functions import seriate_spectral, flip
from popgenml.data.functions import tree_to_distmat
from scipy.spatial.distance import pdist, squareform

import tskit
from typing import Any, Callable

class TSTransform:
    """
    Base transform class to convert a tskit tree sequence into a tensor representation.
    """
    def __init__(self):
        # Base initialization logic (if any is needed across all transforms later)
        pass

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        """
        Transforms the input tree sequence into a tensor.
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")

class SiteDistanceMatrixTransform(TSTransform):
    """
    Transforms a tree sequence into a tensor of shape (L, n choose 2), 
    where L is the number of sites, containing the condensed distance matrix 
    of the tree that covers each site.
    """
    def __init__(self):
        super().__init__() 

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        if not isinstance(ts, tskit.TreeSequence):
            raise TypeError(f"Expected tskit.TreeSequence, got {type(ts)}")

        L = ts.num_sites
        n = ts.num_samples
        n_choose_2 = n * (n - 1) // 2
        
        out_tensor = np.zeros((L, n_choose_2), dtype=np.float32)
        
        for tree in ts.trees():
            sites_in_tree = list(tree.sites())
            
            if not sites_in_tree:
                continue
                
            # Directly call the external tree_to_distmat function
            distmat = tree_to_distmat(tree)
            
            for site in sites_in_tree:
                out_tensor[site.id] = distmat
                
        return out_tensor

class AlignmentTransform:
    """Base class for alignment transformations."""
    def __call__(self, matrix, positions):
        raise NotImplementedError("Subclasses must implement __call__")
        
class Compose(AlignmentTransform):
    """
    Composes several alignment transforms together.
    
    Args:
        transforms (list of AlignmentTransform objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, matrix, positions):
        for transform in self.transforms:
            matrix, positions = transform(matrix, positions)
        return matrix, positions

class RelateTreeSequenceTransform(AlignmentTransform):
    def __init__(self):
        return

class FastSeriate(AlignmentTransform):
    def __init__(self, dist = 'cosine'):
        self.dist = dist
    
    def __call__(self, matrix, positions):
        D = squareform(pdist(matrix, metric = self.dist))
        
        matrix, _ = seriate_spectral(matrix, D)
        
        return matrix, positions

class Flip(AlignmentTransform):
    def __call__(self, matrix, positions):
        return flip(matrix), positions

class RandomSampleShuffle(AlignmentTransform):
    """
    Randomly shuffles the samples (rows) of the (n, l) alignment matrix.
    """
    def __call__(self, matrix, positions):
        n, l = matrix.shape
        
        # Generate a random permutation of row indices (0 to n-1)
        shuffled_indices = np.random.permutation(n)
        
        # Apply the permutation to the matrix rows
        shuffled_matrix = matrix[shuffled_indices, :]
        
        return shuffled_matrix, positions

class PadCrop(AlignmentTransform):
    """
    Symmetrically pads an (n, l) alignment to (n, l_new).
    If l > l_new, it randomly crops it down instead.
    """
    def __init__(self, l_new, matrix_pad_val=-1, pos_pad_val=-1.0):
        self.l_new = l_new
        self.matrix_pad_val = matrix_pad_val
        self.pos_pad_val = pos_pad_val

    def __call__(self, matrix, positions):
        n, l = matrix.shape
        
        # 1. Randomly crop if larger than l_new
        if l > self.l_new:
            start_idx = np.random.randint(0, l - self.l_new + 1)
            end_idx = start_idx + self.l_new
            return matrix[:, start_idx:end_idx], positions[start_idx:end_idx]
            
        # 2. Do nothing if exactly l_new
        elif l == self.l_new:
            return matrix, positions
            
        # 3. Symmetrically pad if smaller than l_new
        else:
            pad_total = self.l_new - l
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            # np.pad requires padding tuples for every dimension: ((dim0_before, dim0_after), (dim1_before, dim1_after))
            # We don't pad the 'n' dimension (0, 0), but we pad the 'l' dimension (pad_left, pad_right)
            padded_matrix = np.pad(
                matrix, 
                pad_width=((0, 0), (pad_left, pad_right)), 
                mode='constant', 
                constant_values=self.matrix_pad_val
            )
            
            # positions is 1D (l,), so we just pad its single dimension
            padded_positions = np.pad(
                positions, 
                pad_width=(pad_left, pad_right), 
                mode='constant', 
                constant_values=self.pos_pad_val
            )
            
            return padded_matrix, padded_positions

import popgenml.data.stats as pg_stats

STAT_FUNCS = {
    'theta_pi': pg_stats.theta_pi,
    'watterson_theta': pg_stats.watterson_theta,
    'sfs': pg_stats.sfs,
    'tajimas_d': pg_stats.tajimas_d,
    'ld_stats': pg_stats.ld_stats,
    'het_diversity': pg_stats.het_diversity
}

class WindowedStats(AlignmentTransform):
    """
    Slices an alignment into windows of a specified base pair length 
    and computes a list of population genetic statistics for each window.
    """
    def __init__(self, stat_names, window_size_bp, ploidy=2):
        self.stat_names = stat_names
        self.window_size_bp = window_size_bp
        self.ploidy = ploidy
        
        # Validate that requested stats exist in our dictionary
        for stat in stat_names:
            if stat not in STAT_FUNCS:
                raise ValueError(f"Unknown statistic '{stat}'. Available: {list(STAT_FUNCS.keys())}")

    def __call__(self, matrix, positions, L):
        n_haps, l_sites = matrix.shape
        
        L_bp = L
        
        # Convert relative floating positions [0, 1] to integer base pairs
        pos_bp = (positions * L_bp).astype(int)
        
        # Calculate the total number of windows
        num_windows = int(np.ceil(L_bp / self.window_size_bp))
        
        # Initialize an empty list for each stat
        results = {stat: [] for stat in self.stat_names}
        
        for w in range(num_windows):
            start_bp = w * self.window_size_bp
            end_bp = start_bp + self.window_size_bp
            
            # Mask to find sites falling within this specific window
            mask = (pos_bp >= start_bp) & (pos_bp < end_bp)
            win_matrix = matrix[:, mask]
            win_pos = pos_bp[mask]
            
            for stat in self.stat_names:
                func = STAT_FUNCS[stat]
                
                # If a window has fewer than 2 sites, most stats will fail.
                # We append a placeholder scalar NaN, which we will reshape later.
                if win_matrix.shape[1] < 2:
                    results[stat].append(np.nan)
                    continue
                    
                # Dynamically dispatch arguments based on what the function accepts
                sig = inspect.signature(func)
                kwargs = {}
                if 'pos' in sig.parameters:
                    kwargs['pos'] = win_pos
                if 'ploidy' in sig.parameters:
                    kwargs['ploidy'] = self.ploidy
                
                try:
                    # Suppress division-by-zero warnings that scikit-allel throws on edge cases
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        val = func(win_matrix, **kwargs)
                    results[stat].append(val)
                except Exception:
                    # Catch underlying math domain errors and fallback to NaN
                    results[stat].append(np.nan)
        
        # Cleanup: Convert lists to appropriately shaped NumPy arrays
        final_results = {}
        for stat in self.stat_names:
            # Find the first valid (non-NaN) window to determine the shape of the stat.
            # For example, SFS returns an array of shape (n_haps,), while Tajima's D returns a scalar.
            valid_shapes = [np.array(v).shape for v in results[stat] if not (np.isscalar(v) and np.isnan(v))]
            
            if not valid_shapes:
                # Edge case: If the entire genome is completely empty
                final_results[stat] = np.full(num_windows, np.nan)
            else:
                expected_shape = valid_shapes[0]
                
                cleaned_stat = []
                for v in results[stat]:
                    # Replace our placeholder NaNs with an array of NaNs of the correct shape
                    if np.isscalar(v) and np.isnan(v):
                        cleaned_stat.append(np.full(expected_shape, np.nan))
                    else:
                        cleaned_stat.append(v)
                        
                # Stack along the first dimension so shape is (num_windows, *stat_shape)
                final_results[stat] = np.stack(cleaned_stat)

        return final_results


