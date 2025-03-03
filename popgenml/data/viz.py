# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import msprime
import demesdraw
import numpy as np

def plot_demography(demography, log_time = True):
    graph = msprime.Demography.to_demes(demography)
    fig, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
    demesdraw.tubes(graph, ax=ax, seed=1, log_time = log_time)
    plt.show()
    
"""
Plots piecewise constant size history. Must call plt.show() or plt.savefig() after one more calls.
"""
def plot_size_history(Nt, max_t = None, color = 'k'):
    N = [u[0] for u in Nt]
    t = [u[1] for u in Nt]

    N = N[1:]
    t = np.log10(t[1:])

    for k in range(len(N) - 1):
        plt.plot([t[k], t[k + 1]], [N[k], N[k]], c = color)
        plt.plot([t[k + 1], t[k + 1]], [N[k], N[k + 1]], c = color)

    if max_t:
        plt.plot([t[-1], max_t], [N[-1], N[-1]], c = color)