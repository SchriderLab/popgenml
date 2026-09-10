# Targeted Demographic Histories API Reference

The `targeted_history` module provides utilities for sampling continuous demographic trajectories $N(t)$ calibrated to match a predetermined mutational budget (expected segregating sites $S$) while enforcing rigorous population size bounds. 

This is particularly useful for coalescent simulation workflows (e.g., using [`msprime`](https://tskit.dev/msprime/)) where demographic scenarios need to remain computationally tractable, biologically plausible, and normalized in terms of genetic diversity.

---

## Mathematical Overview

### 1. Target Tree Length Calibration
Under the standard infinite sites model, the expected number of segregating sites $\mathbb{E}[S]$ in a sample of $n$ haploid lineages across a sequence of length $L$ with per-base per-generation mutation rate $\mu$ is:

$$\mathbb{E}[S] = \mu L \cdot \mathbb{E}[L_n]$$

where $\mathbb{E}[L_n]$ is the expected total genealogy branch length:

$$\mathbb{E}[L_n] = \int_0^\infty \mathbb{E}[A_n(t)] \, dt$$

Here, $\mathbb{E}[A_n(t)]$ denotes the expected number of ancestral lineages remaining at lookback time $t$, parameterized via the cumulative coalescent intensity $\Lambda(t)$:

$$\Lambda(t) = \int_0^t \frac{1}{\text{ploidy} \cdot N(u)} \, du$$

Given a desired target SNP count, the target expected branch length is computed as:

$$\mathbb{E}[L_n]^\ast = \frac{\text{target\_snps}}{\mu \times \text{seq\_len}}$$

### 2. Logistic Squashing and Root Finding
To enforce strict boundary conditions $N(t) \in [N_{\min}, N_{\max}]$, any raw trajectory $f(t)$ is passed through a logistic sigmoid transform scaled by an additive vertical log-shift $c$:

$$N(t; c) = N_{\min} + \sigma(c + f(t)) \cdot (N_{\max} - N_{\min})$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$. 

The class uses Brent's method (`scipy.optimize.brentq`) over $c \in [-20.0, 20.0]$ to find $c^\ast$ such that:

$$\int_0^{T_{\text{max\_math}}} \mathbb{E}[A_n(\Lambda(t; c^\ast))] \, dt - \mathbb{E}[L_n]^\ast = 0$$

### 3. Dual Time Grids
- **Mathematical Grid (`t_math`)**: High-resolution geometric grid spanning $[0, T_{\text{max\_math}}]$ (e.g., $2 \times 10^6$ generations) to ensure that the coalescent process goes to absorption ($\mathbb{E}[A_n(t)] \to 1$) and numerical integrals evaluate accurately.
- **Simulation Grid (`t_sim`)**: Downsampled epoch grid spanning $[0, T_{\text{max\_sim}}]$ (e.g., 100 epochs up to $2 \times 10^5$ generations) to plug directly into simulation software without excessive epoch overhead.

---

## Classes

### `TargetedHistory`

```python
class TargetedHistory(
    target_snps=20000,
    n_haps=16,
    mu=1.5e-08,
    seq_len=2500000.0,
    N_min=5000.0,
    N_max=100000.0,
    T_max_sim=200000,
    n_sim_epochs=100,
    T_max_math=2000000,
    n_math_pts=1000,
    ploidy=2
)
```

Base class for calibrating demographic trajectories to a target expected tree length.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_snps` | `int` \| `float` | `20000` | Target number of segregating sites (SNPs) desired across the sample. |
| `n_haps` | `int` | `16` | Total sample size in haploid lineages ($n$). |
| `mu` | `float` | `1.5e-8` | Per-base, per-generation mutation rate ($\mu$). |
| `seq_len` | `float` | `2.5e6` | Total sequence length in base pairs. |
| `N_min` | `float` | `5000.0` | Lower bound for effective population size $N(t)$. |
| `N_max` | `float` | `100000.0` | Upper bound for effective population size $N(t)$. |
| `T_max_sim` | `int` \| `float` | `200000` | Maximum time horizon (generations backwards in time) for the simulation grid. |
| `n_sim_epochs` | `int` | `100` | Number of discrete time intervals across the simulation grid. |
| `T_max_math` | `int` \| `float` | `2000000` | Extended backwards time horizon for numerical coalescent integrals. |
| `n_math_pts` | `int` | `1000` | Number of integration points across `t_math`. |
| `ploidy` | `int` | `2` | Organism ploidy (denominator coalescent factor $\text{ploidy} \cdot N(t)$). |

#### Attributes

- **`target_Ln`** (`float`): Target expected total branch length $\mathbb{E}[L_n]$.
- **`t_math`** (`numpy.ndarray`): Geometrically spaced evaluation grid of shape `(n_math_pts,)`.
- **`x_math`** (`numpy.ndarray`): Scaled domain in $[-1, 1]$ used for Chebyshev evaluations.
- **`t_sim`** (`numpy.ndarray`): Discrete epoch boundaries of shape `(n_sim_epochs,)`.
- **`expected_A_func`** (`callable`): Precomputed mapping $\Lambda \mapsto \mathbb{E}[A_n(\Lambda)]$.

#### Methods

##### `_scale_and_bound(raw_log_shape)`
Solves for the optimal shift scalar $c^\ast$ via Brent's method, squashes the shape between $[N_{\min}, N_{\max}]$, and linearly interpolates the trajectory onto `t_sim`.

- **Parameters:**
  - `raw_log_shape` (`numpy.ndarray` of shape `(n_math_pts,)`): Continuous shape evaluated on `t_math`.
- **Returns:**
  - `t_sim` (`numpy.ndarray`): Epoch change times (generations ago).
  - `N_t_sim` (`numpy.ndarray`): Effective population sizes at epoch boundaries.
- **Raises:**
  - `ValueError`: If the target $\mathbb{E}[L_n]$ cannot be satisfied within $[-20, 20]$ bounds.

---

### `ChebyshevHistory`

```python
class ChebyshevHistory(
    num_coeffs=13,
    volatility=1.0,
    **kwargs
)
```

*Inherits from [`TargetedHistory`](#targetedhistory)*.

Generates continuously fluctuating demographic histories using randomized truncated series of Chebyshev polynomials of the first kind $T_j(x)$:

$$f(t) = \sum_{j=0}^{\text{num\_coeffs}-1} a_j T_j(x(t)), \quad a_j \sim \mathcal{N}\left(0, \frac{\text{volatility}^2}{j + 1}\right)$$

The $(j + 1)^{-0.5}$ variance decay damps high-frequency oscillatory noise, creating smooth population fluctuations.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_coeffs` | `int` | `13` | Number of polynomial terms sampled ($0 \le j < \text{num\_coeffs}$). |
| `volatility` | `float` | `1.0` | Global scaling factor for polynomial coefficients. Higher values yield deeper fluctuations. |
| `**kwargs` | `dict` | | Forwarded to `TargetedHistory.__init__`. |

#### Methods

##### `sample_curve()`
Draws random coefficients, evaluates $f(t)$ on `x_math`, and calibrates via `_scale_and_bound`.

- **Returns:**
  - `t_sim` (`numpy.ndarray`): Simulation time breakpoints.
  - `N_t_sim` (`numpy.ndarray`): Calibrated population sizes.

---

### `ExponentialHistory`

```python
class ExponentialHistory(
    abs_r_min=1e-05,
    abs_r_max=0.001,
    **kwargs
)
```

*Inherits from [`TargetedHistory`](#targetedhistory)*.

Models continuous exponential growth or contraction phases transitioning into ancient plateaus.

A rate magnitude $|r|$ is sampled log-uniformly:

$$\ln |r| \sim \mathcal{U}(\ln(\text{abs\_r\_min}), \ln(\text{abs\_r\_max}))$$

with an equally probable growth or decay direction $\text{sign} \in \{-1, +1\}$. Backwards in time:

$$f(t) = -r \cdot t$$

When passed through the logistic transform, positive forward growth ($r > 0$) resembles an expansion from an ancient bottleneck, while negative rates describe forward declines.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `abs_r_min` | `float` | `1e-5` | Minimum absolute per-generation exponential rate. |
| `abs_r_max` | `float` | `1e-3` | Maximum absolute per-generation exponential rate. |
| `**kwargs` | `dict` | | Forwarded to `TargetedHistory.__init__`. |

#### Methods

##### `sample_curve()`
Samples rate $r$, constructs $-r \cdot t$, and calibrates to the target SNP expectation.

- **Returns:**
  - `t_sim` (`numpy.ndarray`): Simulation time breakpoints.
  - `N_t_sim` (`numpy.ndarray`): Calibrated population sizes.

---

## Example Usage with msprime

```python
import msprime
import numpy as np
import matplotlib.pyplot as plt
from targeted_history import ChebyshevHistory, ExponentialHistory

# Initialize generator targeting 30,000 SNPs in 20 haploid genomes
cheby_gen = ChebyshevHistory(
    target_snps=30000,
    n_haps=20,
    mu=1.25e-8,
    seq_len=5e6,
    N_min=2000,
    N_max=150000,
    num_coeffs=10,
    volatility=1.2
)

# Sample calibrated trajectory
t_sim, N_sim = cheby_gen.sample_curve()

# Construct msprime demographic model
demography = msprime.Demography()
demography.add_population(name="pop0", initial_size=N_sim[0])

for t, N in zip(t_sim[1:], N_sim[1:]):
    demography.add_population_parameters_change(time=t, initial_size=N, population="pop0")

# Simulate ancestry
ts = msprime.sim_ancestry(
    samples=10, # 10 diploids = 20 haplotypes
    demography=demography,
    sequence_length=5e6,
    recombination_rate=1e-8,
    random_seed=42
)

# Overlay mutations
mts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=42)
print(f"Simulated SNPs: {mts.num_mutations} (Target was ~30,000)")
```