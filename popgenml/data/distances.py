# -*- coding: utf-8 -*-
import numpy as np
import ot  # POT library for Earth Mover's Distance

def extract_tree_distribution(tree, normalize=True):
    """
    Extracts clades (patterns) and branch lengths (weights) from a tskit.Tree.
    
    Args:
        tree: A tskit.Tree object.
        normalize: If True, normalizes branch lengths to sum to 1.0, 
                   representing the probability distribution of site patterns.
    """
    patterns = []
    weights = []
    
    # Iterate over all nodes to calculate the branch length above them
    for u in tree.nodes():
        parent = tree.parent(u)
        
        # tskit.NULL is -1. The root has no parent, so we skip it.
        if parent != -1:  
            branch_length = tree.time(parent) - tree.time(u)
            
            # Only record branches with non-zero length
            if branch_length > 0:
                # Get the sample IDs under this node.
                # Converting to frozenset makes the clade hashable for dictionaries/sets
                clade = frozenset(tree.samples(u))
                
                patterns.append(clade)
                weights.append(branch_length)
                
    weights = np.array(weights)
    
    # Normalize weights so they represent a valid probability distribution
    if normalize and np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        
    return patterns, weights


# ==========================================
# Ground Cost Metrics (Your Original Functions)
# ==========================================
def cost_hamming(s1, s2):
    """Symmetric Difference: Size of clades not shared."""
    return len(s1.symmetric_difference(s2))

def cost_iou(s1, s2):
    """Jaccard Distance: 1.0 - (Intersection over Union)."""
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return 1.0 - (intersection / union)

def emd_dist(p1, w1, p2, w2, metric_func):
    """Computes Earth Mover's Distance using the provided ground metric."""
    C = np.zeros((len(p1), len(p2)))
    for i, s1 in enumerate(p1):
        for j, s2 in enumerate(p2):
            C[i, j] = metric_func(s1, s2)
    return ot.emd2(w1, w2, C)

def kl_dist(p1, w1, p2, w2, epsilon=1e-9, symmetric=True):
    """Computes the (Symmetric) KL divergence between two site pattern distributions."""
    all_patterns = set(p1).union(set(p2))
    dict1 = dict(zip(p1, w1))
    dict2 = dict(zip(p2, w2))
    
    P = np.array([dict1.get(pat, 0.0) for pat in all_patterns])
    Q = np.array([dict2.get(pat, 0.0) for pat in all_patterns])
    
    P = P + epsilon
    Q = Q + epsilon
    
    P /= np.sum(P)
    Q /= np.sum(Q)
    
    kl_pq = np.sum(P * np.log(P / Q))
    
    if not symmetric:
        return kl_pq
        
    kl_qp = np.sum(Q * np.log(Q / P))
    return (kl_pq + kl_qp) / 2.0


# ==========================================
# TSKit Wrappers
# ==========================================
def tree_emd(tree1, tree2, metric="hamming"):
    """
    Computes Earth Mover's Distance between two tskit trees.
    metric can be 'iou', 'hamming', or a custom function.
    """
    p1, w1 = extract_tree_distribution(tree1, normalize=True)
    p2, w2 = extract_tree_distribution(tree2, normalize=True)
    
    if metric == "iou":
        metric_func = cost_iou
    elif metric == "hamming":
        metric_func = cost_hamming
    else:
        metric_func = metric
        
    return emd_dist(p1, w1, p2, w2, metric_func)

def tree_kl(tree1, tree2, epsilon=1e-9, symmetric=True):
    """
    Computes Symmetric KL Divergence of site pattern distributions 
    between two tskit trees.
    """
    p1, w1 = extract_tree_distribution(tree1, normalize=True)
    p2, w2 = extract_tree_distribution(tree2, normalize=True)
    
    return kl_dist(p1, w1, p2, w2, epsilon, symmetric)