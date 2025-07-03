# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import msprime
import demesdraw
import numpy as np
from matplotlib.axes import Axes
import tskit

def plot_tree_on_ax(tree: tskit.Tree, ax: Axes, title: str = None, **kwargs):
    """
    Plots a single tskit.Tree on a given Matplotlib Axes object.

    This function manually draws the tree by calculating node positions
    and then plotting nodes and edges. This replaces the deprecated
    `draw_mpl` function from older tskit versions.

    Args:
        tree: The tskit.Tree object to plot.
        ax: The Matplotlib Axes object to draw on.
        title: An optional title for the subplot.
        **kwargs: Additional keyword arguments. Supported: 'node_color'.
    """
    # --- 1. Calculate node positions using a layout algorithm ---
    # `pos` will be a numpy array where pos[u, 0] is the x-coord
    # and pos[u, 1] is the y-coord of node u.
    pos = np.zeros((tree.num_nodes, 2))

    # Assign x-coordinates to samples (leaves) first
    for i, u in enumerate(tree.samples()):
        pos[u, 0] = i

    # Calculate x-coords for internal nodes in a bottom-up traversal
    for u in tree.nodes(order="postorder"):
        if not tree.is_sample(u):
            child_x_coords = [pos[c, 0] for c in tree.children(u)]
            if child_x_coords:
                # Position internal nodes at the average x-position of their children
                pos[u, 0] = np.mean(child_x_coords)

    # Assign y-coordinates based on node time
    for u in tree.nodes():
        pos[u, 1] = tree.time(u)

    # --- 2. Draw the tree on the axes ---
    # Set plot limits and labels
    ax.set_ylim(0, tree.time(tree.root) * 1.05) # Add 5% padding to top
    ax.set_xlim(-0.5, len(list(tree.samples())) - 0.5)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Time Ago")

    # Draw vertical lines from each node to its parent's time
    for u in tree.nodes():
        if tree.parent(u) != tskit.NULL:
            parent_time = pos[tree.parent(u), 1]
            ax.plot(
                [pos[u, 0], pos[u, 0]],  # x-coords (a vertical line)
                [pos[u, 1], parent_time], # y-coords
                color='black', linewidth=1.0, zorder=1
            )

    # Draw horizontal lines connecting siblings under a parent
    for u in tree.nodes():
        if not tree.is_sample(u):
            children = tree.children(u)
            if children:
                min_child_x = min(pos[c, 0] for c in children)
                max_child_x = max(pos[c, 0] for c in children)
                ax.plot(
                    [min_child_x, max_child_x], # x-coords (a horizontal line)
                    [pos[u, 1], pos[u, 1]],     # y-coords (at the parent's time)
                    color='black', linewidth=1.0, zorder=1
                )

    # Draw the nodes themselves
    node_color = kwargs.get("node_color", "skyblue")
    ax.scatter(
        pos[:, 0], pos[:, 1],
        color=node_color, s=50, zorder=2, ec='black'
    )

    if title:
        ax.set_title(title)

def plot_demography(demography, log_time = True):
    graph = msprime.Demography.to_demes(demography)
    fig, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
    demesdraw.tubes(graph, ax=ax, seed=1, log_time = log_time)
    plt.show()
    
"""
Plots piecewise constant size history. Must call plt.show() or plt.savefig() after one or more calls.
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