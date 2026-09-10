# -*- coding: utf-8 -*-
import inspect
import warnings
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import tskit
from scipy.spatial.distance import pdist, squareform

from .functions import (
    flip,
    seriate_ortools,
    seriate_spectral,
    tree_to_distmat,
)
import .stats as pg_stats


class TSTransform:
    r"""
    Base interface for converting a tskit TreeSequence into a tensor representation.

    All tree sequence transformations must inherit from this class and implement
    the :meth:`__call__` method.
    """

    def __init__(self):
        pass

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        r"""
        Transform an input tree sequence into an array or tensor.

        Parameters
        ----------
        ts : tskit.TreeSequence
            The tree sequence object to be transformed.

        Returns
        -------
        np.ndarray
            The tensor representation of the tree sequence.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")


class SiteDistanceMatrixTransform(TSTransform):
    r"""
    Transform a tree sequence into condensed genealogical distance matrices per site.

    For every segregating site across the input tree sequence, this transform 
    extracts the local marginal tree enclosing the site, calculates all pairwise 
    genealogical distances between the $n$ samples, and records the condensed 
    upper-triangular distance vector of length $\binom{n}{2}$.

    Parameters
    ----------
    None

    See Also
    --------
    popgenml.data.functions.tree_to_distmat : Function computing the pairwise tree distance matrix.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        r"""
        Compute the condensed pairwise distance matrix for every site in the tree sequence.

        Parameters
        ----------
        ts : tskit.TreeSequence
            Input tree sequence containing $L$ sites and $n$ samples.

        Returns
        -------
        np.ndarray of shape (L, n * (n - 1) // 2)
            A 2D float32 tensor where row $i$ represents the condensed pairwise
            genealogical distance matrix for the tree spanning site ID $i$. Sites
            not covered by a valid tree will retain default zero entries.

        Raises
        ------
        TypeError
            If ``ts`` is not an instance of :class:`tskit.TreeSequence`.
        """
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
    r"""
    Base class for spatial and haplotype transformations on alignment matrices.

    Subclasses must implement :meth:`__call__` accepting a haplotype matrix and
    corresponding physical/relative genomic positions.
    """

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Apply transformation to alignment and coordinate vectors.

        Parameters
        ----------
        matrix : np.ndarray of shape (n, l)
            Binary or integer haplotype/genotype matrix where $n$ represents the
            number of sequences/samples and $l$ is the number of segregating sites.
        positions : np.ndarray of shape (l,)
            Chromosomal coordinates or relative positions in $[0, 1]$ corresponding 
            to the columns of ``matrix``.

        Returns
        -------
        matrix : np.ndarray
            Transformed haplotype matrix.
        positions : np.ndarray
            Transformed coordinate array.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class Compose(AlignmentTransform):
    r"""
    Compose a sequential chain of alignment transformations.

    Parameters
    ----------
    transforms : list of AlignmentTransform
        List of callable alignment transformation objects to apply in order.

    Examples
    --------
    >>> pipeline = Compose([
    ...     RandomSampleShuffle(),
    ...     PadCrop(l_new=128),
    ...     FastSeriate(dist='cosine')
    ... ])
    >>> mat_out, pos_out = pipeline(matrix, positions)
    """

    def __init__(self, transforms: List[AlignmentTransform]):
        self.transforms = transforms

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Execute each transform sequentially on the matrix and positions.

        Parameters
        ----------
        matrix : np.ndarray of shape (n, l)
            Haplotype matrix.
        positions : np.ndarray of shape (l,)
            Genomic coordinate positions.

        Returns
        -------
        matrix : np.ndarray
            Sequentially transformed matrix.
        positions : np.ndarray
            Sequentially transformed positions.
        """
        for transform in self.transforms:
            matrix, positions = transform(matrix, positions)
        return matrix, positions


class FastSeriate(AlignmentTransform):
    r"""
    Reorder matrix rows using spectral seriation on sample pairwise distances.

    Constructs a full distance matrix across haplotypes using the specified
    distance metric, computes the Fiedler vector via the normalized graph Laplacian,
    and sorts the samples to minimize coordinate distances between similar rows.

    Parameters
    ----------
    dist : str, default='cosine'
        Distance metric recognized by :func:`scipy.spatial.distance.pdist` 
        (e.g., `'cosine'`, `'euclidean'`, `'hamming'`, `'cityblock'`).
    """

    def __init__(self, dist: str = "cosine"):
        self.dist = dist

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Seriate rows of the alignment matrix based on spectral ordering.

        Parameters
        ----------
        matrix : np.ndarray of shape (n, l)
            Haplotype alignment matrix.
        positions : np.ndarray of shape (l,)
            Genomic coordinates (unmodified by row reordering).

        Returns
        -------
        matrix : np.ndarray of shape (n, l)
            Permuted haplotype matrix with sorted rows.
        positions : np.ndarray of shape (l,)
            Original position vector.
        """
        D = squareform(pdist(matrix, metric=self.dist))

        matrix, _ = seriate_spectral(matrix, D)

        return matrix, positions


class ORToolsSeriate(AlignmentTransform):
    r"""
    Reorder matrix rows by solving the Travelling Salesperson Problem (TSP) with OR-Tools.

    Solves an optimal or near-optimal linear ordering of sample rows that minimizes
    the total stepwise distance along the sequence of rows using Google OR-Tools.

    Parameters
    ----------
    dist : str, default='cosine'
        Distance metric passed to the underlying OR-Tools seriation solver.
    """

    def __init__(self, dist: str = "cosine"):
        self.dist = dist

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Seriate rows using an integer programming / routing solver.

        Parameters
        ----------
        matrix : np.ndarray of shape (n, l)
            Haplotype alignment matrix.
        positions : np.ndarray of shape (l,)
            Genomic coordinates.

        Returns
        -------
        matrix : np.ndarray of shape (n, l)
            Permuted haplotype matrix with rows ordered via TSP solution.
        positions : np.ndarray of shape (l,)
            Original position vector.
        """
        matrix, _ = seriate_ortools(matrix, self.dist)

        return matrix, positions


class Flip(AlignmentTransform):
    r"""
    Polarize or invert alleles within an alignment matrix.

    Applies the :func:`popgenml.data.functions.flip` transformation, commonly
    used to polarize derived versus ancestral states, standardize major/minor
    alleles, or augment training data via allele complementation ($0 \leftrightarrow 1$).
    """

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Invert or polarize alleles within the alignment matrix.

        Parameters
        ----------
        matrix : np.ndarray of shape (n, l)
            Haplotype alignment matrix.
        positions : np.ndarray of shape (l,)
            Genomic coordinates (unmodified).

        Returns
        -------
        matrix : np.ndarray of shape (n, l)
            Flipped or polarized haplotype matrix.
        positions : np.ndarray of shape (l,)
            Original position vector.
        """
        return flip(matrix), positions


class RandomSampleShuffle(AlignmentTransform):
    r"""
    Randomly permute the sample rows of an alignment matrix.

    Applies a uniform random permutation along the sample dimension ($n$). 
    This acts as a data augmentation technique and prevents machine learning 
    models from overfitting to arbitrary sample index orderings.
    """

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Uniformly permute rows in the matrix.

        Parameters
        ----------
        matrix : np.ndarray of shape (n, l)
            Haplotype matrix where rows represent samples.
        positions : np.ndarray of shape (l,)
            Genomic coordinates (unmodified).

        Returns
        -------
        shuffled_matrix : np.ndarray of shape (n, l)
            Matrix with row indices randomly permuted.
        positions : np.ndarray of shape (l,)
            Original position vector.
        """
        n, l = matrix.shape

        # Generate a random permutation of row indices (0 to n-1)
        shuffled_indices = np.random.permutation(n)

        # Apply the permutation to the matrix rows
        shuffled_matrix = matrix[shuffled_indices, :]

        return shuffled_matrix, positions


class PadCrop(AlignmentTransform):
    r"""
    Standardize the number of segregating sites via random cropping or symmetric padding.

    If the number of sites $l > l_{\text{new}}$, a contiguous window of length 
    $l_{\text{new}}$ is chosen uniformly at random. If $l < l_{\text{new}}$, constant 
    padding is symmetrically appended to both flanking ends of the alignment and position 
    vectors.

    Parameters
    ----------
    l_new : int
        Target number of columns (segregating sites) in the output matrix.
    matrix_pad_val : int or float, default=-1
        Fill value assigned to padded regions within the matrix.
    pos_pad_val : float, default=-1.0
        Fill value assigned to padded regions within the position array.
    """

    def __init__(
        self,
        l_new: int,
        matrix_pad_val: Union[int, float] = -1,
        pos_pad_val: float = -1.0,
    ):
        self.l_new = l_new
        self.matrix_pad_val = matrix_pad_val
        self.pos_pad_val = pos_pad_val

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Resize alignment columns and positions to exact target length `l_new`.

        Parameters
        ----------
        matrix : np.ndarray of shape (n, l)
            Alignment matrix.
        positions : np.ndarray of shape (l,)
            Corresponding chromosomal or relative positions.

        Returns
        -------
        matrix : np.ndarray of shape (n, l_new)
            Resized alignment matrix cropped or padded to $l_{\text{new}}$ sites.
        positions : np.ndarray of shape (l_new,)
            Resized position vector cropped or padded to $l_{\text{new}}$ entries.
        """
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

            # np.pad requires padding tuples for every dimension:
            # ((dim0_before, dim0_after), (dim1_before, dim1_after))
            padded_matrix = np.pad(
                matrix,
                pad_width=((0, 0), (pad_left, pad_right)),
                mode="constant",
                constant_values=self.matrix_pad_val,
            )

            # positions is 1D (l,), so we just pad its single dimension
            padded_positions = np.pad(
                positions,
                pad_width=(pad_left, pad_right),
                mode="constant",
                constant_values=self.pos_pad_val,
            )

            return padded_matrix, padded_positions


STAT_FUNCS = {
    "theta_pi": pg_stats.theta_pi,
    "watterson_theta": pg_stats.watterson_theta,
    "sfs": pg_stats.sfs,
    "tajimas_d": pg_stats.tajimas_d,
    "ld_stats": pg_stats.ld_stats,
    "het_diversity": pg_stats.het_diversity,
}


class WindowedStats(AlignmentTransform):
    r"""
    Compute population genetic summary statistics across non-overlapping physical windows.

    Partitions a genomic interval of length $L$ base pairs into discrete windows of size 
    ``window_size_bp`` and computes specified summary statistics on each window using 
    functions in :mod:`popgenml.data.stats`. Windows with fewer than two segregating sites 
    are imputed with ``np.nan``.

    Parameters
    ----------
    stat_names : list of str
        Names of statistics to compute. Must be keys in:
        ``['theta_pi', 'watterson_theta', 'sfs', 'tajimas_d', 'ld_stats', 'het_diversity']``.
    window_size_bp : int
        Window span in base pairs.
    ploidy : int, default=2
        Ploidy level forwarded to statistics that require it (e.g., heterozygosity).

    Raises
    ------
    ValueError
        If any requested statistic in ``stat_names`` is not supported.
    """

    def __init__(self, stat_names: List[str], window_size_bp: int, ploidy: int = 2):
        self.stat_names = stat_names
        self.window_size_bp = window_size_bp
        self.ploidy = ploidy

        # Validate that requested stats exist in our dictionary
        for stat in stat_names:
            if stat not in STAT_FUNCS:
                raise ValueError(
                    f"Unknown statistic '{stat}'. Available: {list(STAT_FUNCS.keys())}"
                )

    def __call__(
        self, matrix: np.ndarray, positions: np.ndarray, L: int
    ) -> Dict[str, np.ndarray]:
        r"""
        Calculate windowed statistics across the alignment.

        Parameters
        ----------
        matrix : np.ndarray of shape (n_haps, l_sites)
            Binary haplotype or genotype matrix.
        positions : np.ndarray of shape (l_sites,)
            Relative floating-point site positions scaled to the interval $[0, 1]$.
        L : int
            Total sequence length in physical base pairs.

        Returns
        -------
        dict of str to np.ndarray
            Dictionary mapping each statistic name to an array of shape 
            ``(num_windows, *stat_shape)``, where ``num_windows = ceil(L / window_size_bp)``.
            Scalar statistics yield shape ``(num_windows,)``, while vector statistics 
            (such as the SFS) yield shape ``(num_windows, stat_dim)``.
        """
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
                if "pos" in sig.parameters:
                    kwargs["pos"] = win_pos
                if "ploidy" in sig.parameters:
                    kwargs["ploidy"] = self.ploidy

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
            valid_shapes = [
                np.array(v).shape
                for v in results[stat]
                if not (np.isscalar(v) and np.isnan(v))
            ]

            if not valid_shapes:
                # Edge case: If the entire genome is completely empty
                final_results[stat] = np.full(num_windows, np.nan)
            else:
                expected_shape = valid_shapes[0]

                cleaned_stat = []
                for v in results[stat]:
                    # Replace placeholder NaNs with an array of NaNs of the correct shape
                    if np.isscalar(v) and np.isnan(v):
                        cleaned_stat.append(np.full(expected_shape, np.nan))
                    else:
                        cleaned_stat.append(v)

                # Stack along the first dimension so shape is (num_windows, *stat_shape)
                final_results[stat] = np.stack(cleaned_stat)

        return final_results

