The `TargetedHistory` framework generates dynamic population size trajectories $N(t)$ that are mathematically calibrated to yield a specific expected number of segregating sites (SNPs). These trajectories are defined via INI-style configuration files that natively parse `numpy.random` and `scipy.stats` functions, allowing you to easily define prior distributions for genetic simulators like `msprime` and `discoal`.

## Demographic Trajectory Models

All demographic shapes inherit from a base scaler that guarantees the resulting history stays within realistic population bounds while hitting exact mutational budgets.

*   **`TargetedHistory` (Base Class):** Takes an arbitrary continuous trajectory $f(t)$ and bounds it strictly within $[N_{\min}, N_{\max}]$ using a scaled logistic transform. It mathematically integrates the Kingman coalescent expected tree length $E[L_n]$ across a high-resolution time grid, applying Brent's root-finding method to shift the trajectory until the resulting tree length perfectly matches your `target_snps` budget.
*   **`ChebyshevHistory`:** Generates smoothly oscillating, non-standard demographic histories using randomized Chebyshev polynomials. The `volatility` parameter scales the variance of the polynomial coefficients; higher values create dramatic population booms and busts, while decaying variance suppresses unrealistic high-frequency jitter.
*   **`ExponentialHistory`:** Models continuous exponential growth or decay. It draws an absolute exponential rate $r$ log-uniformly and randomly assigns a positive or negative sign.
*   **`PiecewiseConstantHistory`:** Models instantaneous demographic shifts. It drops random time breakpoints (knots) across the simulation horizon and assigns a uniform random population size to each resulting epoch.

## Configuration File Structure

The configuration files dictate the fixed parameters and prior distributions for your simulations. The parser evaluates the string values as Python code, meaning any function from `np.random` or `stats` (from `scipy.stats`), as well as the `TargetedHistory` classes, can be executed directly in the config.

### `[base]`
Defines the biological constants and physical architecture of the simulated genomic region.

```ini
[base]
mu = 1.5e-8      # Per-base, per-generation mutation rate
r = 1.007e-8     # Per-base, per-generation recombination rate
L = 2500000      # Sequence length in base pairs
ploidy = 2       # Organism ploidy
```

### `[samples]`
Defines the populations present in the simulation, their sample sizes, and their specific demographic histories over time. 

```ini
[samples]
# Evaluates a Chebyshev history targeting between 12,000 and 24,000 SNPs
pop1 = {'Nt': 'ChebyshevHistory(target_snps=np.random.uniform(12000, 24000), n_haps=100, volatility = 2.0)', 'n': 50}
```
*   `Nt`: The demographic history class. The string is evaluated dynamically. You can parameterize `target_snps` with a random distribution to create a training dataset with diverse mutational densities.
*   `n_haps`: The haploid sample size passed to the history class.
*   `n`: The number of diploid individuals actually sampled by the simulator. Note that `n_haps` must equal `n * ploidy`.

### `[migration]`
Defines the per-generation probability that a lineage transfers between populations (gene flow) when simulating multiple populations.

```ini
[migration]
# Draws a migration rate log-uniformly between 1e-5 and 1e-3
pop1,pop2 = 10 ** np.random.uniform(-5, -3)
```
*   The key (`pop1,pop2`) specifies the directional or symmetric migration routes. 
*   The value evaluates to the continuous migration rate prior.

### `[discoal]`
Specific to simulations using `discoal` for modeling selective sweeps (adaptive introgression, hard/soft sweeps). 

```ini
[discoal]
args = '-Pf 0.0 0.05 -Pc 0.5 1.0 -Pu 0.0 0.01 -ws 0'
s = stats.loguniform(1e-4, 1e-2)
x = stats.uniform(loc = 0.05, scale = 0.9)
```
*   `args`: Raw command-line string passed directly to the `discoal` binary (e.g., setting fixation times or initial sweep frequencies).
*   **Variable Assignments:** You can define named prior distributions for sweep parameters using `scipy.stats`. In this example, `s` (the selection coefficient) is drawn log-uniformly, and `x` (the physical position of the sweep on the sequence) is drawn uniformly across the middle 90% of the simulated region.

