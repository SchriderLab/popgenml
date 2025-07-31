# Function Documentation

## `tree_to_graph(tree, n=200)`

Converts a TSKit tree into node features and an edge list suitable for graph-based processing.

### Parameters
- `tree`: `tskit.Tree`  
  A tree sequence object.
- `n`: `int`, default=200  
  Number of sample nodes.

### Returns
- `X`: `np.ndarray`, shape `(2n-1, 2)`  
  Node features: first column is node time, second is mutation count.
- `edge_index`: `np.ndarray`, shape `(2, 2n-2)`  
  Directed edges from parent to child.

---

## `graph_to_tree(x, edges)`

Converts graph-based node features and an edge list back into a TSKit tree.

### Parameters
- `x`: `np.ndarray`, shape `(2n-1, 2)`  
  Node features. First column must contain node times.
- `edges`: `np.ndarray`, shape `(2n-2, 2)`  
  Edge list as parent-child pairs.

### Returns
- `ts_tree`: `tskit.Tree`  
  Reconstructed TSKit tree.

---

## `distmat_to_tree(D, metric='euclidean')`

Constructs a tree from a distance matrix using hierarchical clustering.

### Parameters
- `D`: `np.ndarray`  
  Condensed or full distance matrix.
- `metric`: `str`, default=`'euclidean'`  
  Distance metric for clustering.

### Returns
- `ts_tree`: `tskit.Tree`  
  Reconstructed tree sequence object.

---

## `tree_to_distmat(tree)`

Converts a tree to a condensed pairwise genealogical distance matrix.

### Parameters
- `tree`: `tskit.Tree`  
  A TSKit tree object.

### Returns
- `D`: `np.ndarray`, shape `(n_samples * (n_samples - 1) / 2,)`  
  Condensed distance matrix.

---

## `pad_sequences(sequences, max_length=None, padding_value=0)`

Pads a list of 2D arrays to the same number of rows.

### Parameters
- `sequences`: `List[np.ndarray]`  
  Each element is an array of shape `(n_sites, n_samples)`.
- `max_length`: `int`, optional  
  Maximum sequence length to pad to.
- `padding_value`: `int` or `float`, default=0  
  Value to use for padding.

### Returns
- `np.ndarray`, shape `(batch_size, max_length, n_samples)`  
  Padded sequences.

---

## `format_matrix(x, pos, pop_sizes, y=None, out_shape=(2, 32, 128), metric='cosine', mode='seriate')`

Formats a genotype matrix for input to machine learning models.

### Parameters
- `x`: `np.ndarray`, shape `(n_ind, n_sites)`  
  Genotype matrix.
- `pos`: `np.ndarray`, shape `(n_sites,)`  
  Positions of SNPs.
- `pop_sizes`: `Tuple[int]`  
  Tuple specifying number of individuals in each population.
- `y`: `np.ndarray`, optional  
  Optional labels (same shape as `x`).
- `out_shape`: `Tuple[int]`, default=`(2, 32, 128)`  
  Output shape (n_pops, n_ind, n_sites).
- `metric`: `str`, default=`'cosine'`  
  Distance metric for ordering individuals.
- `mode`: `str`, one of `['seriate_match', 'seriate', 'pad']`  
  How to format and align individuals.

### Returns
- `x_formatted`: `np.ndarray`  
  Transformed genotype matrix.
- `pos_formatted`: `np.ndarray`  
  Transformed positions.
- `y_formatted`: `np.ndarray` or `None`  
  Transformed labels if provided.

---

## `to_unique(X)`

Computes a histogram of unique site configurations in a genotype matrix.

### Parameters
- `X`: `np.ndarray`, shape `(n_samples, n_sites)`  
  Genotype matrix.

### Returns
- `np.ndarray`, shape `(n_unique_sites, n_samples + 1)`  
  Unique site configurations with their frequency proportions.

---

## `seriate_spectral(x, C)`

Performs spectral seriation to reorder rows based on similarity matrix.

### Parameters
- `x`: `np.ndarray`, shape `(n_samples, n_features)`  
  Data matrix.
- `C`: `np.ndarray`, shape `(n_samples, n_samples)`  
  Similarity or affinity matrix.

### Returns
- `x_sorted`: `np.ndarray`  
  Reordered data matrix.
- `ix`: `np.ndarray`  
  Indices used to reorder the data.

---
