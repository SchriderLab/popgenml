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

## Simulation

### msprime

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

## Formatting data from simulation replicates

Like in many applications of machine learning, your choice of how to represent a replicate in your training set and what information to discard etc. can drastically affect the performance obtained.  For instance, if we were working with 3D objects as input to some model we might try to represent them as point clouds (a list of coordinates), as a binary 3d grid (voxels), or as a mesh (points with graph connectivity):

![image](https://miro.medium.com/v2/resize:fit:1158/1*n4uKWdVBwQGlB77Y3hsTPQ.png)

In population genetic simulations we often save alignments for a sample of individuals over some region of their genome.  We can make various choices, however, about how we represent the alignment and what if any post-processing steps such as the sorting of sampled individuals or tree sequence inference.  For instance we could choose for input to our model the alignment itself, inferred tree sequences as a sequence of graphs, the site frequence spectrum (histogram of allele counts), or use a tree matrix embedding (from https://www.pnas.org/doi/10.1073/pnas.1922851117):

![image](https://github.com/SchriderLab/popgenml/blob/devel/docs/images/pogenml_intro_fig1.png)

Each choice implies a set of architectures or models that can take it as input, and different choices may be better suited for different inference problems.  For instance, if we were to use inferred trees as a list of graphs (with node features and edges) as our input, then we can use graph convolution (https://en.wikipedia.org/wiki/Graph_neural_network) as part of our model, whereas an alignment matrix is suited for traditional 1d or 2d convolution.  

### Sorting of alignment matrices

We include two functions meant to "sort" individuals such that genetically similar individuals are near each other in the ordering.  
