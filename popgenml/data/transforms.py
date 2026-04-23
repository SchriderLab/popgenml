# -*- coding: utf-8 -*-
import numpy as np
import inspect
import warnings

class AlignmentTransform:
    """Base class for alignment transformations."""
    def __call__(self, matrix, positions, L):
        raise NotImplementedError("Subclasses must implement __call__")

class RandomCrop(AlignmentTransform):
    """
    Randomly crops an (n, l) alignment down to (n, l_new).
    """
    def __init__(self, l_new):
        self.l_new = l_new

    def __call__(self, matrix, positions, L):
        n, l = matrix.shape
        
        # If the alignment is already smaller than or equal to the crop size, return as-is
        if l <= self.l_new:
            return matrix, positions, L
            
        # np.random.randint's upper bound is exclusive
        start_idx = np.random.randint(0, l - self.l_new + 1)
        end_idx = start_idx + self.l_new
        
        cropped_matrix = matrix[:, start_idx:end_idx]
        cropped_positions = positions[start_idx:end_idx]
        
        return cropped_matrix, cropped_positions, L

class Pad(AlignmentTransform):
    """
    Symmetrically pads an (n, l) alignment to (n, l_new).
    If l > l_new, it randomly crops it down instead.
    """
    def __init__(self, l_new, matrix_pad_val=-1, pos_pad_val=-1.0):
        self.l_new = l_new
        self.matrix_pad_val = matrix_pad_val
        self.pos_pad_val = pos_pad_val

    def __call__(self, matrix, positions, L):
        n, l = matrix.shape
        
        # 1. Randomly crop if larger than l_new
        if l > self.l_new:
            start_idx = np.random.randint(0, l - self.l_new + 1)
            end_idx = start_idx + self.l_new
            return matrix[:, start_idx:end_idx], positions[start_idx:end_idx], L
            
        # 2. Do nothing if exactly l_new
        elif l == self.l_new:
            return matrix, positions, L
            
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
            
            return padded_matrix, padded_positions, L

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
        
        # Convert length from Mb (megabases) to base pairs
        L_bp = L * 1e6
        
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


