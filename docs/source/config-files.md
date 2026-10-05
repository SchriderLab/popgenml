# Targeted History Configuration Tutorial

The `TargetedHistory` framework generates dynamic population size trajectories $N(t)$ that are mathematically calibrated to yield a specific expected number of segregating sites (SNPs). These trajectories are defined via INI-style configuration files that natively parse `numpy.random` and `scipy.stats` functions, allowing you to easily define prior distributions for genetic simulators like `msprime` and `discoal`.

## Demographic Trajectory Models

All demographic shapes inherit from a base scaler that guarantees the resulting history stays within realistic population bounds while hitting exact mutational budgets.

* **`TargetedHistory` (Base Class):** Takes an arbitrary continuous trajectory $f(t)$ and bounds it strictly within $[N_{\min}, N_{\max}]$ using a scaled logistic transform. It mathematically integrates the Kingman coalescent expected tree length $E[L_n]$ across a high-resolution time grid, applying Brent's root-finding method to shift the trajectory until the resulting tree length perfectly matches your `target_snps` budget.

* **`ChebyshevHistory`:** Generates smoothly oscillating, non-standard demographic histories using randomized Chebyshev polynomials. The `volatility` parameter scales the variance of the polynomial coefficients; higher values create dramatic population booms and busts.

* **`ExponentialHistory`:** Models continuous exponential growth or decay. It draws an absolute exponential rate $r$ log-uniformly and randomly assigns a positive or negative sign.

* **`PiecewiseConstantHistory`:** Models instantaneous demographic shifts. It drops random time breakpoints (knots) across the simulation horizon and assigns a uniform random population size to each resulting epoch.

---

## Configuration File Manual

The simulation configuration file uses the INI format and is divided into main sections: `[base]` for global parameters, `[samples]` for defining sample populations, `[migration]` for defining migration rates between them, and `[discoal]` for simulator-specific overrides.

### The `[base]` Section

This section defines the global physical parameters of the simulation.

* **`mu`**: (Required) The per-base mutation rate per generation.
  * *Type*: Can be a fixed floating-point number or a `scipy.stats` distribution.
  * *Example (fixed)*: `mu = 1.25e-8`
  * *Example (distribution)*: `mu = stats.uniform(loc=1e-9, scale=2e-8)`

* **`r`**: (Required) The per-base recombination rate per generation.
  * *Type*: Can be a fixed floating-point number or a `scipy.stats` distribution.
  * *Example (fixed)*: `r = 1.007e-8`
  * *Example (distribution)*: `r = stats.loguniform(a=1e-9, b=5e-8)`

* **`L`**: (Required) The total length of the simulated sequence in base pairs.
  * *Type*: Must be a single, fixed integer.
  * *Example*: `L = 2500000`

### The `[samples]` Section

This section defines the properties of each population to be sampled. Each line represents a distinct population, identified by a custom name (e.g., `pop1`, `pop2`). The value for each population must be a dictionary-like string containing the following keys:

* **`n`**: (Required) The number of individuals to sample from the population.
  * *Type*: Must be an integer greater than zero.
  * *Example*: `'n': 50`

* **`ploidy`**: (Required) The ploidy of the sampled individuals. *(Note: Can also be defined globally in `[base]` depending on your parser setup).*
  * *Type*: Must be an integer, either `1` (haploid) or `2` (diploid).
  * *Example*: `'ploidy': 2`

* **`N0` or `Nt`**: (Required) A population size model must be specified using either `N0` for a constant size or `Nt` for a variable size history. If both are provided, `Nt` will be used and `N0` will be ignored.
  
  * **`N0`**: Defines a constant effective population size ($N_e$).
    * *Type*: Fixed number or a `scipy.stats` distribution.
    * *Example*: `'N0': 'stats.loguniform(a=1000, b=50000)'`
  
  * **`Nt`**: Defines a variable effective population size over time.
    * *Type*: Can be a `TargetedHistory` class instance or a direct list of `(size, time in generations)` tuples.
    * *Example (TargetedHistory)*: 
      `'Nt': 'ChebyshevHistory(target_snps=np.random.uniform(12000, 24000), n_haps=100, volatility=2.0)'`
    * *Example (Tuple list)*: 
      `'Nt': '[(10000, 0), (50000, 500), (10000, 2000)]'`

### The `[migration]` Section

This section defines the rate of migration between pairs of populations defined in the `[samples]` section.

* **Key Format**: The key defines the direction of migration. A key of `popA_popB` specifies the migration rate **from** `popB` **into** `popA`.
* **Mechanism**: The migration rate is the fraction of `popA` that is made up of migrants from `popB` in each generation. 
* **Value Format**: The value defines the migration rate over time, which can be constant or variable. A history like `[(m0, t0), (m1, t1)]` means the migration rate is $m_0$ until time $t_1$, at which point it becomes $m_1$.
  * *Example (Constant/Distribution)*: `pop1_pop2 = 10 ** np.random.uniform(-5, -3)`
  * *Example (Tuple list)*: `pop2_pop1 = [(0.0, 0), (0.001, 500), (0.0, 2000)]`

### The `[discoal]` Section

This section is for arguments specific to the `discoal` simulator (often used for modeling selective sweeps).

* **`args`**: 
  * *Type*: String with constant arguments passed directly to the `discoal` binary.
  * *Example*: `args = '-Pf 0.0 0.05 -Pc 0.5 1.0 -Pu 0.0 0.01 -ws 0'`

* **Variable Assignments**: You can define named prior distributions for sweep parameters (like selection coefficient `s` or sweep location `x`) using `scipy.stats`.
  * *Example*:
    ```ini
    s = stats.loguniform(1e-4, 1e-2)
    x = stats.uniform(loc = 0.05, scale = 0.9)
    ```
    
### Examples

Varying recombination rate according to a truncated exponential distribution:

```
[base]
mu = 1.5e-8
r = TruncatedExponential(a = 1e-8, b = 1e-6, lam = 4641167.677653745)
L = 200000
ploidy = 1

[samples]
# A diploid population with a variable size history defined by a spline
pop1 = {'N0': 'UniformFloatDiscrete([1e3, 2e3, 5e3, 1e4, 2e4, 5e4])', 'n': 32}
```


