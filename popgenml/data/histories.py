# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp, simpson
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.special import expit, logit
from numpy.polynomial.chebyshev import chebval

def precompute_kingman_lineages(n, tau_max=15.0, num_pts=1000):
    """Precomputes the expected number of surviving lineages over coalescent time tau."""
    tau_grid = np.linspace(0, tau_max, num_pts)
    
    def kingman_odes(tau, P):
        dP = np.zeros_like(P)
        for k in range(2, n + 1):
            rate_out = k * (k - 1) / 2.0
            dP[k] = -rate_out * P[k]
            if k < n:
                rate_in = (k + 1) * k / 2.0
                dP[k] += rate_in * P[k+1]
        return dP
    
    P_init = np.zeros(n + 1)
    P_init[n] = 1.0
    
    sol = solve_ivp(kingman_odes, [0, tau_max], P_init, t_eval=tau_grid, method='BDF')
    states = np.arange(2, n + 1)
    expected_lineages = np.sum(sol.y[2:] * states[:, None], axis=0)
    
    return interp1d(sol.t, expected_lineages, kind='cubic', bounds_error=False, fill_value=0.0)

class TargetedHistory:
    r"""
    Base class for calibrating demographic trajectories to a target expected tree length.

    This class samples or shifts continuous effective population size trajectories
    $N(t)$ such that the expected total branch length $E[L_n]$ matches a specified
    mutational budget.

    The trajectory is bounded strictly within $[N_{\min}, N_{\max}]$ via a logistic
    transform (expit), and root-finding (Brent's method) is applied across Kingman 
    coalescent integrals over a high-resolution mathematical time grid before 
    downsampling to a discrete simulation grid suitable for engines like msprime.

    Parameters
    ----------
    target_snps : int or float, default=20000
        Target number of segregating sites (SNPs) desired in the coalescent sample.
    n_haps : int, default=16
        Total sample size in haploid lineages ($n$).
    mu : float, default=1.5e-8
        Per-base, per-generation mutation rate ($\mu$).
    seq_len : float, default=2.5e6
        Total sequence length in base pairs.
    N_min : float, default=5000.0
        Lower bound for diploid effective population size $N(t)$.
    N_max : float, default=100000.0
        Upper bound for diploid effective population size $N(t)$.
    T_max_sim : int or float, default=200000
        Maximum time horizon for the simulation epoch grid.
    n_sim_epochs : int, default=100
        Number of discrete time breakpoints across the simulation grid.
    T_max_math : int or float, default=2_000_000
        Extended backwards time horizon used for numerical coalescent integrals.
    n_math_pts : int, default=1000
        Number of integration points across the extended mathematical grid.
    ploidy : int, default=2
        Ploidy of the organism.

    Attributes
    ----------
    target_Ln : float
        Target expected total tree length.
    t_math : ndarray of shape (n_math_pts,)
        Geometrically spaced time grid.
    x_math : ndarray of shape (n_math_pts,)
        Time values normalized to $[-1, 1]$.
    t_sim : ndarray of shape (n_sim_epochs,)
        Geometrically spaced time grid for simulation.
    expected_A_func : callable
        Precomputed lineage function.
    """

    def __init__(self, target_snps=20000, n_haps=16, mu=1.5e-8, seq_len=2.5e6,
                 N_min=5000.0, N_max=100000.0, T_max_sim=200000, n_sim_epochs=100,
                 T_max_math=2_000_000, n_math_pts=1000, ploidy=2):
        
        self.target_Ln = target_snps / (mu * seq_len)
        self.N_min = N_min
        self.N_max = N_max
        
        # 1. Math Grid: Massive horizon to ensure the coalescent integral completes
        self.t_math = np.geomspace(1, T_max_math + 1, n_math_pts) - 1 
        self.x_math = 2.0 * (self.t_math / T_max_math) - 1.0
        
        # 2. Simulation Grid: Coarser horizon for efficient msprime stepping
        self.t_sim = np.geomspace(1, T_max_sim + 1, n_sim_epochs) - 1
        
        self.ploidy = ploidy
        self.expected_A_func = precompute_kingman_lineages(n_haps)

    def _scale_and_bound(self, raw_log_shape):
        r"""
        Calibrate and project an unconstrained trajectory onto the simulation grid.

        Applies an additive log-shift $c$ to an arbitrary unconstrained trajectory 
        $f(t)$, maps it to bounded population space via a scaled logistic curve:
        
            N(t) = N_min + \text{expit}(c + f(t)) * (N_max - N_min)

        and solves for $c^*$ via Brent's method so that the integrated tree length 
        $E[L_n]$ equals `self.target_Ln`. The result is then interpolated onto 
        `self.t_sim`.

        Parameters
        ----------
        raw_log_shape : ndarray of shape (n_math_pts,)
            Arbitrary continuous curve evaluated along `self.t_math` prior 
            to bounding and scaling.

        Returns
        -------
        t_sim : ndarray of shape (n_sim_epochs,)
            Simulation time points in generations backwards from present.
        N_t_sim : ndarray of shape (n_sim_epochs,)
            Calibrated effective population sizes corresponding to `t_sim`.

        Raises
        ------
        ValueError
            If Brent's root-finding method cannot find a zero crossing within the 
            bracket $[-20.0, 20.0]$, indicating the target tree length cannot be 
            realized within the $[N_{\min}, N_{\max}]$ bounds.
        """
        def expected_Ln_error(log_c):
            squashed = expit(log_c + raw_log_shape)
            N_t = self.N_min + squashed * (self.N_max - self.N_min)
            
            # Using 1.0 / (2.0 * N_t) assumes msprime uses ploidy=2 
            inv_2N = 1.0 / (self.ploidy * N_t)
            Lambda_t = cumulative_trapezoid(inv_2N, self.t_math, initial=0)
            
            expected_Ln = simpson(y=self.expected_A_func(Lambda_t), x=self.t_math)
            return expected_Ln - self.target_Ln
            
        # Solve for the exact log-shift
        log_c_star = brentq(expected_Ln_error, -20.0, 20.0)
        
        # Generate the mathematically complete curve
        squashed_final = expit(log_c_star + raw_log_shape)
        N_t_math = self.N_min + squashed_final * (self.N_max - self.N_min)
        
        # Interpolate down to the fast simulation grid
        N_t_sim = np.interp(self.t_sim, self.t_math, N_t_math)
        
        return self.t_sim, N_t_sim


class ChebyshevHistory(TargetedHistory):
    r"""
    Demographic history sampler using randomized Chebyshev polynomial series.

    Generates smoothly oscillating historical demographic trajectories by drawing
    random coefficients for Chebyshev polynomials of the first kind $T_j(x)$.
    High-frequency oscillations are suppressed using decaying coefficient variance 
    proportional to $(j + 1)^{-0.5}$.

    Parameters
    ----------
    num_coeffs : int, default=13
        Number of Chebyshev terms (degrees $0$ through `num_coeffs - 1`) sampled.
    volatility : float, default=1.0
        Global scale factor for coefficient variance. Higher values produce larger 
        relative amplitudes between historical population peaks and troughs.
    **kwargs
        Arbitrary keyword arguments forwarded to `TargetedHistory.__init__`.
    """
    
    def __init__(self, num_coeffs=13, volatility=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_coeffs = num_coeffs
        self.volatility = volatility

    def sample_curve(self):
        r"""
        Sample a random Chebyshev trajectory and scale it to target specifications.

        Returns
        -------
        t_sim : ndarray of shape (n_sim_epochs,)
            Simulation time points in generations backwards from present.
        N_t_sim : ndarray of shape (n_sim_epochs,)
            Calibrated effective population sizes corresponding to `t_sim`.
        """
        # Decaying variance limits aggressive high-frequency oscillations
        variances = np.array([self.volatility / (j + 1)**0.5 for j in range(self.num_coeffs)])
        coeffs = np.random.randn(self.num_coeffs) * variances
        
        # Evaluate on the massive math grid
        raw_log_shape = chebval(self.x_math, coeffs)
        
        return self._scale_and_bound(raw_log_shape)


class ExponentialHistory(TargetedHistory):
    r"""
    Demographic history sampler modeling continuous exponential growth or decay.

    Draws an exponential rate $r$ log-uniformly over an interval and applies
    $N_{\text{raw}}(t) \propto \exp(-r \cdot t)$ backwards in time. When bounded 
    and scaled by the parent class, positive growth rates ($r > 0$) correspond 
    to recent forward-time population expansions that plateau into the past.

    Parameters
    ----------
    abs_r_min : float, default=1e-5
        Minimum absolute exponential rate per generation.
    abs_r_max : float, default=1e-3
        Maximum absolute exponential rate per generation.
    **kwargs
        Arbitrary keyword arguments forwarded to `TargetedHistory.__init__`.
    """
    
    def __init__(self, abs_r_min=1e-5, abs_r_max=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.abs_r_min = abs_r_min
        self.abs_r_max = abs_r_max

    def sample_curve(self):
        r"""
        Sample an exponential growth or decline trajectory and scale to target specifications.

        Draws $|r| \sim \text{LogUniform}(\text{abs\_r\_min}, \text{abs\_r\_max})$ and assigns
        a random directional sign ($\pm 1$). Backward-in-time trajectory is defined
        as $-r \cdot t$ before being calibrated by `_scale_and_bound`.

        Returns
        -------
        t_sim : ndarray of shape (n_sim_epochs,)
            Simulation time points in generations backwards from present.
        N_t_sim : ndarray of shape (n_sim_epochs,)
            Calibrated effective population sizes corresponding to `t_sim`.
        """
        # 1. Draw a random magnitude for the rate (log-uniform)
        if self.abs_r_min != self.abs_r_max:
            r_mag = np.exp(np.random.uniform(np.log(self.abs_r_min), np.log(self.abs_r_max)))
        else:
            r_mag = 0.
            
        # 2. Randomly assign a positive (growth) or negative (decay) sign
        sign = np.random.choice([-1.0, 1.0])
        r = r_mag * sign
        
        # 3. Looking backwards in time, the log-size scales linearly
        raw_log_shape = -r * self.t_math
        
        return self._scale_and_bound(raw_log_shape)

class PiecewiseConstantHistory(TargetedHistory):
    r"""
    Demographic history sampler modeling piecewise-constant population epochs.

    Samples epoch change points (knots) uniformly across the simulation time
    horizon and draws plateau population sizes uniformly between `N_min` and `N_max`.
    The trajectory is mapped to logit space so that it smoothly integrates with
    the parent class's logistic scaling and calibration mechanics.

    Parameters
    ----------
    num_epochs : int, default=5
        Total number of piecewise-constant epochs (resulting in `num_epochs - 1`
        internal knot points).
    T_knot_max : float, optional
        Maximum time horizon (in generations backwards) over which internal knots
        can be placed. If None, defaults to `T_max_sim`.
    **kwargs
        Arbitrary keyword arguments forwarded to `TargetedHistory.__init__`.
    """

    def __init__(self, num_epochs=5, T_knot_max=None, **kwargs):
        super().__init__(**kwargs)
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1.")
        self.num_epochs = num_epochs
        # Knot times default to the simulation horizon if not explicitly specified
        self.T_knot_max = self.t_sim[-1] if T_knot_max is None else float(T_knot_max)

    def sample_curve(self):
        r"""
        Sample a piecewise-constant step trajectory and scale to target specifications.

        Selects `num_epochs - 1` knot times uniformly over `(0, T_knot_max)` and
        `num_epochs` population sizes $N_k \sim \mathcal{U}(N_{\min}, N_{\max})$.
        The values are converted to logit space, evaluated across `self.t_math`,
        and calibrated via `_scale_and_bound`.

        Returns
        -------
        t_sim : ndarray of shape (n_sim_epochs,)
            Simulation time points in generations backwards from present.
        N_t_sim : ndarray of shape (n_sim_epochs,)
            Calibrated effective population sizes corresponding to `t_sim`.
        """
        # 1. Sample internal knot times uniformly across (0, T_knot_max)
        if self.num_epochs > 1:
            interior_knots = np.sort(
                np.random.uniform(0.0, self.T_knot_max, size=self.num_epochs - 1)
            )
            # Knot intervals: [0, k_1, k_2, ..., k_{E-1}, inf)
            knots = np.concatenate(([0.0], interior_knots, [np.inf]))
        else:
            knots = np.array([0.0, np.inf])

        # 2. Sample epoch population levels uniformly in (N_min, N_max)
        # Add small epsilon buffer to avoid infinite logit values at the exact boundaries
        eps = 1e-6
        N_levels = np.random.uniform(
            self.N_min + eps, self.N_max - eps, size=self.num_epochs
        )

        # 3. Convert target population levels to unconstrained logit space:
        #    squashed = (N_k - N_min) / (N_max - N_min) in (0, 1)
        normalized_levels = (N_levels - self.N_min) / (self.N_max - self.N_min)
        logit_levels = logit(normalized_levels)

        # 4. Map the step values onto self.t_math
        # np.digitize returns 1-based indices into knots
        epoch_idx = np.digitize(self.t_math, knots) - 1
        epoch_idx = np.clip(epoch_idx, 0, self.num_epochs - 1)
        raw_log_shape = logit_levels[epoch_idx]

        return self._scale_and_bound(raw_log_shape)
    
from scipy.special import comb
from scipy.optimize import minimize
    
    
def generate_polanski_kimmel_matrix(n):
    r"""
    Generate the Polanski and Kimmel (2005) combinatorial transformation matrix.

    Computes the transition matrix $\mathbf{W} \in \mathbb{R}^{(n-1) \times (n-1)}$ 
    that linearly maps expected coalescent epoch durations $\mathbb{E}[T_k]$ 
    (the total time interval during which ancestral genealogies retain exactly 
    $k$ distinct lineages, for $k \in \{2, \dots, n\}$) to expected branch 
    lengths $\mathbb{E}[t_i]$ subtending exactly $i$ leaf nodes in the 
    sample ($i \in \{1, \dots, n-1\}$):

    $$\mathbb{E}[t_i] = \sum_{k=2}^{n} W_{i, k} \, \mathbb{E}[T_k]$$

    Under the neutral Kingman coalescent with uniform topology probabilities,
    matrix entries correspond to:

    $$W_{i, k} = k \cdot \frac{\binom{n - i - 1}{k - 2}}{\binom{n - 1}{k - 1}} \quad \text{for } 2 \le k \le n - i + 1$$

    and $0$ otherwise.

    Parameters
    ----------
    n : int
        Sample size in haploid lineages ($n \ge 2$).

    Returns
    -------
    W : ndarray of shape (n - 1, n - 1)
        Topological projection matrix where row index $i-1$ represents the SFS 
        derived allele frequency bin $i \in \{1, \dots, n-1\}$ and column index 
        $k-2$ represents the ancestral lineage count epoch $k \in \{2, \dots, n\}$.

    References
    ----------
    .. [1] Polanski, A., & Kimmel, M. (2003). New findings on properties of 
       the spectrum of numbers of segregating sites in the coalescent with 
       recombination. *Theoretical Population Biology*, 64(2), 221-231.
    """
    W = np.zeros((n - 1, n - 1))
    
    for i_idx, i in enumerate(range(1, n)):          # SFS bins i = 1, ..., n-1
        for k_idx, k in enumerate(range(2, n + 1)):  # Epochs k = 2, ..., n
            # The probability formula for coalescent topologies
            if 2 <= k <= n - i + 1:
                W[i_idx, k_idx] = k * (comb(n - i - 1, k - 2) / comb(n - 1, k - 1))
                
    return W


class SFSTargetedHistory:
    r"""
    Abstract base class for fitting continuous demographic histories to an empirical SFS.

    Optimizes the parameters of an effective population trajectory $N(t)$ to match 
    a target Site Frequency Spectrum (SFS) using composite Poisson maximum 
    likelihood under the variable-population Kingman coalescent.

    Lineage state occupancy distributions $\mathbb{P}(A_n(t) = k)$ are solved 
    over coalescent units $\tau(t) = \int_0^t \frac{1}{\text{ploidy} \cdot N(u)} du$ 
    via Kingman death-process continuous-time Markov chain (CTMC) ODEs. The expected 
    epoch times $\mathbb{E}[T_k]$ are then projected into expected SFS entries 
    using the Polanski-Kimmel matrix $\mathbf{W}$.

    Parameters
    ----------
    target_sfs : array_like of shape (n_haps - 1,)
        Empirical site frequency spectrum counts across derived allele frequency 
        bins $i \in \{1, \dots, n_{\text{haps}} - 1\}$.
    n_haps : int
        Total number of sampled haploid lineages ($n$).
    mu : float, default=3.15e-9
        Per-base, per-generation mutation rate ($\mu$).
    seq_len : float, default=2.5e6
        Total sequence length in base pairs ($L$).
    T_max : float, default=2e6
        Maximum historical lookback horizon (in generations) for numerical integration.
    n_pts : int, default=1000
        Number of geometrically spaced integration points along the backwards time grid.
    ploidy : int, default=1
        Ploidy factor determining the coalescent rate denominator $\text{ploidy} \cdot N(t)$ 
        (use 1 for standard haploid effective units, 2 for diploid $2N(t)$ coalescent units).

    Attributes
    ----------
    target_sfs : ndarray of shape (n_haps - 1,)
        Array of observed SFS counts.
    n_haps : int
        Sample size in haploid lineages.
    mu_L : float
        Total locus-wide mutation parameter $\mu \times L$.
    T_max : float
        Maximum backwards time limit.
    ploidy : int
        Organism ploidy factor.
    t_grid : ndarray of shape (n_pts,)
        Geometrically spaced time grid spanning $[0, T_{\max}]$.
    W_matrix : ndarray of shape (n_haps - 1, n_haps - 1)
        Precomputed Polanski-Kimmel transition matrix.
    P_k_func : callable
        Interpolant returning the probability vector $[\mathbb{P}(A_n = 2), \dots, \mathbb{P}(A_n = n)]$ 
        at arbitrary cumulative coalescent times $\tau$.
    """

    def __init__(self, target_sfs, n_haps, mu=3.15e-9, seq_len=2.5e6, T_max=2e6, n_pts=1000, ploidy=1):
        self.target_sfs = np.array(target_sfs)
        self.n_haps = n_haps
        self.mu_L = mu * seq_len
        self.T_max = T_max
        
        self.ploidy = ploidy
        
        self.t_grid = np.geomspace(1, T_max + 1, n_pts) - 1 
        self.W_matrix = generate_polanski_kimmel_matrix(n_haps)
        self.P_k_func = self._precompute_kingman_states(n_haps)

    def _precompute_kingman_states(self, n, tau_max=15.0, num_pts=1000):
        r"""
        Solve the pure-death CTMC ODEs for Kingman coalescent lineage distributions.

        Solves the system $\frac{d P_k(\tau)}{d\tau} = -\binom{k}{2} P_k(\tau) + \binom{k+1}{2} P_{k+1}(\tau)$
        from initial condition $P_n(0) = 1.0$ down to absorbing state $k = 2$.

        Parameters
        ----------
        n : int
            Total initial sample size in haploid lineages.
        tau_max : float, default=15.0
            Maximum cumulative coalescent time horizon for ODE solution.
        num_pts : int, default=1000
            Number of evaluation points across $\tau \in [0, \tau_{\max}]$.

        Returns
        -------
        interpolant : scipy.interpolate.interp1d
            Cubic spline interpolator returning an array of shape `(n - 1, ...)` 
            corresponding to $[P_2(\tau), \dots, P_n(\tau)]$.
        """
        tau_grid = np.linspace(0, tau_max, num_pts)
        def kingman_odes(tau, P):
            dP = np.zeros_like(P)
            for k in range(2, n + 1):
                rate_out = k * (k - 1) / 2.0
                dP[k] = -rate_out * P[k]
                if k < n:
                    rate_in = (k + 1) * k / 2.0
                    dP[k] += rate_in * P[k+1]
            return dP
        
        P_init = np.zeros(n + 1)
        P_init[n] = 1.0
        sol = solve_ivp(kingman_odes, [0, tau_max], P_init, t_eval=tau_grid, method='BDF')
        return interp1d(sol.t, sol.y[2:, :], kind='cubic', bounds_error=False, fill_value=0.0)

    def build_Nt(self, coeffs):
        r"""
        Map optimization coefficients to an effective population size curve over `t_grid`.

        Parameters
        ----------
        coeffs : array_like
            Demographic parameters optimized by the fitting procedure.

        Returns
        -------
        N_t : ndarray of shape (n_pts,)
            Effective population trajectory evaluated along `self.t_grid`.

        Raises
        ------
        NotImplementedError
            Must be implemented in concrete subclasses.
        """
        raise NotImplementedError

    def forward_sfs(self, coeffs):
        r"""
        Compute the expected SFS and population trajectory for a given parameter set.

        Computes cumulative coalescent intensity $\Lambda(t)$, extracts state probabilities 
        $P_k(t)$, integrates across generations to determine expected epoch times 
        $\mathbb{E}[T_k]$, and projects via $\mu L \cdot \mathbf{W} \mathbb{E}[\mathbf{T}]$.

        Parameters
        ----------
        coeffs : array_like
            Demographic parameters passed to `build_Nt`.

        Returns
        -------
        expected_sfs : ndarray of shape (n_haps - 1,)
            Expected count of segregating sites across derived frequency bins $1$ to $n-1$.
        N_t : ndarray of shape (n_pts,)
            Effective population size trajectory evaluated over `self.t_grid`.
        """
        N_t = self.build_Nt(coeffs)
        
        inv_2N = 1.0 / (self.ploidy * N_t)
        Lambda_t = cumulative_trapezoid(inv_2N, self.t_grid, initial=0)
        
        P_k_t = self.P_k_func(Lambda_t) 
        E_T_k = simpson(y=P_k_t, x=self.t_grid, axis=1)
        
        expected_sfs = self.mu_L * (self.W_matrix @ E_T_k)
        return expected_sfs, N_t

    def _poisson_loss(self, coeffs):
        r"""
        Negative Poisson composite log-likelihood loss (omitting data log-factorials).

        $$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{n-1} \left( \lambda_i(\boldsymbol{\theta}) - S_i \ln \lambda_i(\boldsymbol{\theta}) \right)$$

        Parameters
        ----------
        coeffs : array_like
            Demographic parameters evaluated by the optimizer.

        Returns
        -------
        loss : float
            Total negative Poisson log-likelihood.
        """
        pred_sfs, _ = self.forward_sfs(coeffs)
        pred_sfs = np.clip(pred_sfs, 1e-9, None)
        return np.sum(pred_sfs - self.target_sfs * np.log(pred_sfs))

    def fit(self, init_coeffs, bounds=None):
        r"""
        Fit demographic parameters to the empirical SFS using L-BFGS-B optimization.

        Parameters
        ----------
        init_coeffs : array_like
            Initial parameter guess for the numerical optimizer.
        bounds : sequence of (float, float) or scipy.optimize.Bounds, optional
            Lower and upper optimization bounds for each parameter.

        Returns
        -------
        optimal_coeffs : ndarray
            Optimal parameter vector $\hat{\boldsymbol{\theta}}$ found by L-BFGS-B.
        optimal_Nt : ndarray of shape (n_pts,)
            Fitted demographic curve $N(t)$ evaluated across `self.t_grid`.
        optimal_sfs : ndarray of shape (n_haps - 1,)
            Model-predicted expected SFS under the fitted demographic history.
        """
        print("Starting generalized gradient descent...")
        res = minimize(self._poisson_loss, init_coeffs, bounds=bounds, 
                       method='L-BFGS-B', options={'disp': True})
        
        optimal_sfs, optimal_Nt = self.forward_sfs(res.x)
        return res.x, optimal_Nt, optimal_sfs


class ChebyshevSFSHistory(SFSTargetedHistory):
    r"""
    Demographic inference engine parameterizing $N(t)$ via Chebyshev polynomials.

    Fits smooth historical population size changes to an SFS by mapping the 
    log-transformed population trajectory into a series of Chebyshev polynomials 
    of the first kind $T_j(x)$ bounded strictly between $[N_{\min}, N_{\max}]$ via 
    the logistic sigmoid function.

    Parameters
    ----------
    target_sfs : array_like of shape (n_haps - 1,)
        Target empirical Site Frequency Spectrum counts.
    n_haps : int
        Sample size in haploid lineages ($n$).
    num_coeffs : int, default=12
        Number of Chebyshev series coefficients to fit.
    N_min : float, default=5000.0
        Enforced lower bound for effective population size $N(t)$.
    N_max : float, default=1e7
        Enforced upper bound for effective population size $N(t)$.
    **kwargs
        Arbitrary keyword arguments forwarded to `SFSTargetedHistory.__init__`.

    Attributes
    ----------
    x_grid : ndarray of shape (n_pts,)
        Time values from `t_grid` linearly scaled into the Chebyshev domain $[-1, 1]$.
    """

    def __init__(self, target_sfs, n_haps, num_coeffs=12, N_min=5000.0, N_max=1e7, **kwargs):
        super().__init__(target_sfs, n_haps, **kwargs)
        self.num_coeffs = num_coeffs
        self.N_min = N_min
        self.N_max = N_max
        
        # Map time grid [0, T_max] to Chebyshev domain [-1, 1]
        self.x_grid = 2.0 * (self.t_grid / self.T_max) - 1.0

    def build_Nt(self, coeffs):
        r"""
        Reconstruct the bounded demographic curve $N(t)$ from Chebyshev coefficients.

        Evaluates the Chebyshev polynomial expansion $f(x) = \sum_{j=0}^{M-1} c_j T_j(x)$
        and maps values to effective population space via:
        
        $$N(t) = N_{\min} + \sigma(f(x(t))) \cdot (N_{\max} - N_{\min})$$

        Parameters
        ----------
        coeffs : ndarray of shape (num_coeffs,)
            Polynomial coefficients optimized by the solver.

        Returns
        -------
        N_t : ndarray of shape (n_pts,)
            Bounded population sizes along `self.t_grid`.
        """
        # 1. Evaluate the Chebyshev polynomial across the mapped grid
        raw_log_shape = chebval(self.x_grid, coeffs)
        
        # 2. Squash and bound
        squashed = expit(raw_log_shape)
        N_t = self.N_min + squashed * (self.N_max - self.N_min)
        
        return N_t


class ExponentialSFSHistory(SFSTargetedHistory):
    r"""
    Demographic inference engine parameterizing $N(t)$ via exponential growth/decay.

    Fits a two-parameter model consisting of an initial vertical logit offset 
    and a continuous backwards-in-time exponential rate $r$, bounded between 
    $[N_{\min}, N_{\max}]$ via a logistic squashing transform.

    Parameters
    ----------
    target_sfs : array_like of shape (n_haps - 1,)
        Target empirical Site Frequency Spectrum counts.
    n_haps : int
        Sample size in haploid lineages ($n$).
    N_min : float, default=5000.0
        Enforced lower bound for effective population size $N(t)$.
    N_max : float, default=1e7
        Enforced upper bound for effective population size $N(t)$.
    **kwargs
        Arbitrary keyword arguments forwarded to `SFSTargetedHistory.__init__`.
    """

    def __init__(self, target_sfs, n_haps, N_min=5000.0, N_max=1e7, **kwargs):
        super().__init__(target_sfs, n_haps, **kwargs)
        self.N_min = N_min
        self.N_max = N_max

    def build_Nt(self, coeffs):
        r"""
        Reconstruct the bounded demographic curve $N(t)$ from exponential parameters.

        Computes the unconstrained linear shape in logit space backwards in time:
        
        $$f(t) = \log(c) - r \cdot t$$
        
        and squashes it into the population domain:
        
        $$N(t) = N_{\min} + \sigma(f(t)) \cdot (N_{\max} - N_{\min})$$

        Parameters
        ----------
        coeffs : sequence of length 2
            - `coeffs[0]` (`float`): Vertical logit shift parameter ($\log c$).
            - `coeffs[1]` (`float`): Exponential rate per generation backwards in time ($r$). 
              Positive values indicate forward growth (expansion from ancient size).

        Returns
        -------
        N_t : ndarray of shape (n_pts,)
            Bounded population sizes along `self.t_grid`.
        """
        log_c, r = coeffs
        
        # Linear in log-space looking backwards
        raw_log_shape = log_c - r * self.t_grid
        
        # Squash and bound
        squashed = expit(raw_log_shape)
        N_t = self.N_min + squashed * (self.N_max - self.N_min)
        
        return N_t