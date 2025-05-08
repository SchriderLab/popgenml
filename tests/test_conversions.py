# -*- coding: utf-8 -*-
"""
For testing the conversion functions between TSKit Tree object, distance matrices, and node/edge represenations
"""

from popgenml.data.simulators import SimpleCoal
from popgenml.data.functions import tree_to_graph, graph_to_tree, tree_to_distmat, distmat_to_tree
import numpy as np

# Run a coalescent simulation and get the first tree
simulator = SimpleCoal()
ret = simulator.simulate()

ts = ret['ts']
tree = ts.first()

# ===========
print('testing tree to graph and back...')

times = sorted([tree.time(u) for u in tree.nodes()])
times = [u for u in times if u > 0]

x, edges = tree_to_graph(tree, n = 16)
ts_tree = graph_to_tree(x, edges)

times_ = sorted([ts_tree.time(u) for u in ts_tree.nodes()])
times_ = [u for u in times_ if u > 0]

times_ = np.array(times_)
times = np.array(times)

assert np.sum((times - times_) ** 2) == 0
assert tree.rf_distance(ts_tree) == 0

print('success!')
# ==========

# ===========
print('testing tree to distance matrix and back...')

D = tree_to_distmat(tree)
ts_tree = distmat_to_tree(D)

times_ = sorted([ts_tree.time(u) for u in ts_tree.nodes()])
times_ = [u for u in times_ if u > 0]

times_ = np.array(times_)

assert np.sum((times - times_) ** 2) == 0
assert tree.rf_distance(ts_tree) == 0
print('success!!')