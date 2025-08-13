# Configuration File Manual

This document outlines the format for the simulation configuration file. The file uses the INI format and is divided into three main sections: [base] for global parameters, [samples] for defining sample populations, and [migration] for defining migration rates between them.

## [base] Section

This section defines the global physical parameters of the simulation.

+ **mu**: (Required) The per-base mutation rate per generation.

    + Type: Can be a fixed floating-point number or a scipy.stats distribution.

    + Example (fixed): mu = 1.25e-8

    + Example (distribution): mu = stats.uniform(loc=1e-9, scale=2e-8)

+ **r**: (Required) The per-base recombination rate per generation.

    + Type: Can be a fixed floating-point number or a scipy.stats distribution.

    + Example (fixed): r = 1e-8

    + Example (distribution): r = stats.loguniform(a=1e-9, b=5e-8)

+ **L**: (Required) The total length of the simulated sequence in base pairs.

    + Type: Must be a single, fixed integer.

    + Example: L = 100000

## [samples] Section

This section defines the properties of each population to be sampled. Each line represents a distinct population, identified by a custom name (e.g., pop1).

The value for each population must be a dictionary-like string containing the following keys:

+ **n**: (Required) The number of individuals to sample from the population.

    + Type: Must be an integer greater than zero.

    + Example: 'n': 10

+ **ploidy**: (Required) The ploidy of the sampled individuals.

    Type: Must be an integer, either 1 (haploid) or 2 (diploid).

    Example: 'ploidy': 2

+ **N0 or Nt**: (Required) A population size model must be specified using either N0 for a constant size or Nt for a variable size history. If both are provided, Nt will be used and N0 will be ignored.

    + **N0**: Defines a constant effective population size (N_e).

        + Type: Can be a fixed number (integer or float) or a scipy.stats distribution.

        + Example: ``` 'N0': 50000 ```

        + Example (distribution): ``` 'N0': 'stats.loguniform(a=1000, b=50000)' ```

    + **Nt**: Defines a variable effective population size over time. The population size is piecewise constant, changing at specified time points.

        + Type: Can be a History class instance (like SplineHistory) or a direct list of (size, time) tuples.

        + Mechanism: A history ``` [(y0, t0), (y1, t1), ...] ``` means the population size is y0 until time t1, at which point it becomes y1, and so on.

        + Example (History class): ``` 'Nt': 'SplineHistory(N=stats.uniform(1000, 9000))' ```

        + Example (list of tuples): ``` 'Nt': '[(10000, 0), (50000, 500), (10000, 2000)]' ```

## [migration] Section

This section defines the rate of migration between pairs of populations defined in the [samples] section.

### Key Format: 

The key defines the direction of migration. A key of popA_popB specifies the migration rate from popB into popA.

### Value Format: The value defines the migration rate over time, which can be constant or variable.

+ Type: Can be a History class instance (like SplineHistory) or a direct list of (coefficient, time) tuples.

+ Mechanism: The migration rate is the fraction of popA that is made up of migrants from popB in each generation. A history ``` [(m0, t0), (m1, t1), ...] ``` means the migration rate is m0 until time t1, at which point it becomes m1, and so on.

+ Example (History class): ``` pop1_pop2 = SplineHistory(N=stats.uniform(0, 0.01)) ```

+ Example (list of tuples): ``` pop2_pop1 = [(0.0, 0), (0.001, 500), (0.0, 2000)] ```

## Full Example

Here is a complete example of a valid configuration file including a [migration] section:

```
[base]
mu = stats.uniform(loc=1e-9, scale=2e-8)
r = 1e-8
L = 100000

[samples]
# A diploid population with a variable size history defined by a spline
pop1 = {'Nt': 'SplineHistory(N=stats.uniform(loc=10000, scale=140000), max_k=10)', 'n': 50, 'ploidy': 2}

# A haploid population with a constant size drawn from a distribution
pop2 = {'N0': 'stats.loguniform(a=1000, b=50000)', 'n': 20, 'ploidy': 1}

# A diploid population with a fixed constant size
pop3 = {'N0': 10000, 'n': 100, 'ploidy': 2}

[migration]
# Migration from pop2 into pop1 starts at time 500 and stops at time 2000.
pop1_pop2 = [(0.0, 0), (0.001, 500), (0.0, 2000)]

# Migration from pop1 into pop2 is defined by a spline history.
pop2_pop1 = SplineHistory(N=stats.uniform(0, 0.005))
```
