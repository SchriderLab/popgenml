.. _popgenml:

API
==============

Simulators
---------------

The following classes handle the core simulation engines.

.. autoclass:: popgenml.data.simulators.MSPrimeSimulator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: popgenml.data.simulators.DiscoalSimulator
   :members:
   :undoc-members:
   :show-inheritance:


Stats
---------------

Functions for computing statistics on binary popgen alignments.

.. autosummary::
   :toctree: generated/

   popgenml.data.stats.sfs
   popgenml.data.stats.theta_pi
   popgenml.data.stats.watterson_theta
   popgenml.data.stats.tajimas_d
   popgenml.data.stats.ld_stats
   popgenml.data.stats.het_diversity
   

Prior specification
-------------------

.. autoclass:: popgenml.data.histories.TargetedHistory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: popgenml.data.histories.ChebyshevHistory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: popgenml.data.histories.ExponentialHistory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: popgenml.data.histories.PiecewiseConstantHistory
   :members:
   :undoc-members:
   :show-inheritance:
   
Transforms
---------------

Classes for transforming tree sequences, haplotype alignments, and calculating windowed summary statistics.

Tree Sequence Transforms
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: popgenml.data.transforms.TSTransform
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.SiteDistanceMatrixTransform
   :members:
   :special-members: __call__
   :show-inheritance:

Alignment Transforms
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: popgenml.data.transforms.AlignmentTransform
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.Compose
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.FastSeriate
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.ORToolsSeriate
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.Flip
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.RandomSampleShuffle
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.PadCrop
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: popgenml.data.transforms.WindowedStats
   :members:
   :special-members: __call__
   :show-inheritance:
   
Functions / Conversions
-------------------------

.. autosummary::
   :toctree: generated/

    popgenml.data.functions.newick_to_tree
    popgenml.data.functions.tree_to_graph
    popgenml.data.functions.graph_to_tree
    popgenml.data.functions.distmat_to_tree
    popgenml.data.functions.tree_to_distmat
    popgenml.data.functions.pad_sequences
    popgenml.data.functions.to_unique
    popgenml.data.functions.seriate_spectral
    
Relate
-------------------------

.. autosummary::
   :toctree: generated/

   popgenml.data.relate