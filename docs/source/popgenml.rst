.. _popgenml:

popgenml
==============

API reference.

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