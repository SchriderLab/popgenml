# Methods

Simulation of population genetic scenarios involving varying population sizes, migration, selection and other dynamics has become increasingly popular with 
the advent of machine learning.  It is often the first step in developing tools meant to infer quantities such as recombination or mutation rate or whether or not two populations
have had recent admixture.  popgenml hopes to streamline experiments around the development of such tools from within Python.  It has three overarching intended functions:

1. Simulation
    - Defining a demography or prior over demographies
    - Reading and writing of saved simulations
2. Formatting inputs and outputs
    - Genotype matrices
    - Popular pop-gen statistics
    - Inferred genealogical distance matrices
    - Embeddings for marginal trees
    - Graph (node / edge) representation for marginal trees
    - Conversion between representations
3. Training
    - Pre-built torch models
    - Training tools and visualization

## msprime

popgenml features a class named ```popgenml.data.simulators.BaseMSPrimeSimulator``` which is meant to be subclassed to create a desired prior over demographies i.e. demographic parameters 
such as effective population size etc.

Here is an example of a subclass of ```BaseMSPrimeSimulator``` included in the repo:

```
class BaseMSPrimeSimulator(BaseSimulator):
    # L is the size of the simulation in base pairs
    # specify mutation rate
    
    # immutable properties of the simulator are defined here:
    # L: the size of the simulation in base pairs
    # mu: mutation rate
    # r: recombination rate
    # whether or not diploid individuals are simulated (vs haploid)
    # the number of samples
    def __init__(self, L = int(1e5), mu = 1.5e-8, r = 1.007e-8, ploidy = 1, 
                 n_samples = [16], N = 75000):
        self.L = L
        self.mu = mu
        self.r = r
        self.n_samples = n_samples
        
        self.ploidy = ploidy
        
        self.sample_size = sum(n_samples)

        self.co = None
        self.demography = None
        
        self.N = N
        
        return

    # more definitions not shown here...

"""
Constant pop size simulator.
"""  
class SimpleCoal(BaseMSPrimeSimulator):
    def __init__(self, N = 75000, **kwargs):
        super().__init__(**kwargs)
        
        self.N = N
        
    def make_demography(self):
        demography = msprime.Demography()
        
        demography.add_population(name="A", initial_size=self.N)
        
        return demography
```

This creates a simple demography with a constant effective population size and will simulate it using msprime's default Hudson coalescent model.  Because we did not overwrite any of the default parameters defined in ```BaseMSPrimeSimulator```, 
they remain the same for this custom subclass and the resulting simulation will be over 1e5 base pairs and will sample 16 haploid individuals in the present day.  


