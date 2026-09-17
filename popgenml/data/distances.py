# -*- coding: utf-8 -*-
import math
import numpy as np
import tskit

def history_to_tskit(history, H):
    """Converts a labeled history (sequence of merges) into a tskit.Tree."""
    n = len(history) + 1
    tables = tskit.TableCollection(sequence_length=1.0)
    
    # Add n sample nodes at time 0
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        
    block_to_node = {(i,): i for i in range(n)}
    
    # Add internal nodes and edges based on the merge sequence
    for k_idx, merge in enumerate(history):
        new_time = H[n - 1 - k_idx]
        new_node = tables.nodes.add_row(flags=0, time=new_time)
        u1, u2 = block_to_node[merge[0]], block_to_node[merge[1]]
        
        tables.edges.add_row(left=0.0, right=1.0, parent=new_node, child=u1)
        tables.edges.add_row(left=0.0, right=1.0, parent=new_node, child=u2)
        
        merged_block = tuple(sorted(merge[0] + merge[1]))
        block_to_node[merged_block] = new_node
        
    tables.sort()
    return tables.tree_sequence().first()

def tskit_to_history(tree):
    """Extracts the labeled history (ordered sequence of merges) from a tskit.Tree."""
    # Sort internal nodes strictly by time
    internal_nodes = sorted([u for u in tree.nodes() if not tree.is_sample(u)], 
                            key=lambda u: tree.time(u))
    
    # Map each node to the set of leaves it subtends
    node_leaves = {u: tuple(sorted(list(tree.samples(u)))) for u in tree.nodes()}
    
    history = []
    for u in internal_nodes:
        c1, c2 = tree.children(u)
        merge = tuple(sorted([node_leaves[c1], node_leaves[c2]]))
        history.append(merge)
        
    return tuple(history)

def smc_prime_P(n):
    """Computes exact SMC' transitions utilizing tskit for tree traversals."""
    # 1. Generate coalescent expectations
    H = {n: 0.0, 0: float('inf')}
    for k in range(n-1, 0, -1):
        H[k] = H[k+1] + 2.0 / ((k+1) * k)
        
    # 2. Enumerate histories
    def backtrack(current_partition, current_history, histories_list):
        if len(current_partition) == 1:
            histories_list.append(tuple(current_history))
            return
        for i in range(len(current_partition)):
            for j in range(i + 1, len(current_partition)):
                new_block = tuple(sorted(current_partition[i] + current_partition[j]))
                new_partition = sorted([current_partition[k] for k in range(len(current_partition)) if k not in (i,j)] + [new_block])
                merged_pair = tuple(sorted([current_partition[i], current_partition[j]]))
                backtrack(tuple(new_partition), current_history + [merged_pair], histories_list)

    histories = []
    backtrack(tuple([(i,) for i in range(n)]), [], histories)
    N = len(histories)
    P_matrix = np.zeros((N, N))
    history_to_idx = {h: i for i, h in enumerate(histories)}
    
    L_tot = sum(m * (H[m-1] - H[m]) for m in range(2, n+1))
    
    # 3. Compute transitions using tskit API
    for idx_T1, h in enumerate(histories):
        tree = history_to_tskit(h, H)
        
        # Iterate over cuttable branches (all nodes except the root)
        for b in tree.nodes():
            if b == tree.root: continue
            
            e_node = tree.parent(b)
            b_sib = [c for c in tree.children(e_node) if c != b][0]
            
            b_start = tree.time(b)
            b_end = tree.time(e_node)
            
            e = next(k for k in range(n, 0, -1) if abs(H[k-1] - b_end) < 1e-9)
            epochs_c = [c for c in range(n, 1, -1) if H[c] >= b_start - 1e-9 and H[c-1] <= b_end + 1e-9]
            
            for c in epochs_c:
                W_cut = (H[c-1] - H[c]) / L_tot
                
                for m in range(c, 0, -1):
                    # Transition Integration Math
                    delta_Hc = H[c-1] - H[c]
                    Kc = c - 1 if c >= e else c
                    P_escape_c = (1.0 - math.exp(-Kc * delta_Hc)) / (Kc * delta_Hc)
                    
                    if m == c:
                        prob = 1.0 - P_escape_c
                    else:
                        prob = P_escape_c
                        for j in range(c-1, m, -1):
                            Kj = j - 1 if j >= e else j
                            prob *= math.exp(-Kj * (H[j-1] - H[j]))
                        if m > 1:
                            Km = m - 1 if m >= e else m
                            prob *= (1.0 - math.exp(-Km * (H[m-1] - H[m])))
                            
                    # Find valid target branches using tskit
                    targets = []
                    for u in tree.nodes():
                        if u == b: continue
                        # Target must exist at time H[m]
                        if tree.time(u) <= H[m] + 1e-9 and (tree.parent(u) == tskit.NULL or tree.time(tree.parent(u)) >= H[m-1] - 1e-9):
                            y = b_sib if u == e_node else u
                            targets.append(y)
                            
                    Km = len(targets)
                    for y in targets:
                        # Build modified topology arrays to extract the new history
                        parents = {u: tree.parent(u) for u in tree.nodes()}
                        times = {u: tree.time(u) for u in tree.nodes()}
                        
                        old_parent_y = tree.parent(e_node) if y == b_sib else tree.parent(y)
                        
                        parents[b_sib] = tree.parent(e_node)
                        parents[b] = e_node
                        parents[y] = e_node
                        parents[e_node] = old_parent_y
                        
                        times[e_node] = H[1] + 1.0 if m == 1 else (H[m] + H[m-1]) / 2.0
                        
                        # Reconstruct the labeled history from the new parent/time arrays
                        internal_nodes = sorted([u for u in tree.nodes() if not tree.is_sample(u)], key=lambda x: times[x])
                        leaf_sets = {i: {i} for i in range(n)}
                        
                        new_h = []
                        for u in internal_nodes:
                            children = [v for v in tree.nodes() if parents[v] == u]
                            c1, c2 = children
                            leaf_sets[u] = leaf_sets[c1].union(leaf_sets[c2])
                            new_h.append(tuple(sorted([
                                tuple(sorted(list(leaf_sets[c1]))), 
                                tuple(sorted(list(leaf_sets[c2])))
                            ])))
                            
                        P_matrix[idx_T1, history_to_idx[tuple(new_h)]] += W_cut * prob / Km

    return P_matrix, histories

import matplotlib.pyplot as plt

# Verify row sums are exactly 1
P, h = smc_prime_P(5)
print("Row sums:", np.sum(P, axis=1))
    
plt.imshow(np.log(P))
plt.show()