# -*- coding: utf-8 -*-
import sys
sys.path.append('popgenml/data/')

from simulators import SimpleCoal
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

sim = SimpleCoal(n_samples = [11])
F, W, pop_vectors, coal_times, x, sites, ts = sim.simulate_fw(1000, 1e-6)

F = np.array(F)

Df = squareform(pdist(F, metric = 'cityblock'))
print(np.max(Df), F.shape[0])

plt.hist(squareform(Df))
plt.show()


D = np.zeros((ts.num_trees, ts.num_trees))

for i in range(ts.num_trees):
    for j in range(i, ts.num_trees):
        D[i, j] = ts.at_index(i).rf_distance(ts.at_index(j))        
        
ii = np.array(range(D.shape[0] - 1), dtype = np.int32)
plt.scatter(D[ii, ii + 1], Df[ii, ii + 1])
plt.show()