import numpy as np

from seriate import seriate
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linear_sum_assignment

from scipy.sparse.linalg import eigs
import networkx as nx
from scipy.cluster.hierarchy import linkage
import itertools
import tskit
import newick

def newick_to_tree(
    string, *, min_edge_length=0, span=1, time_units=None, node_name_key=None, multiplier = 1.
) -> tskit.TreeSequence:
    """
    Create a tree sequence representation of the specified newick string.

    The tree sequence will contain a single tree, as specified by the newick. All
    leaf nodes will be marked as samples (``tskit.NODE_IS_SAMPLE``). Newick names and
    comments will be written to the node metadata. This can be accessed using e.g.
    ``ts.node(0).metadata["name"]``.

    :param string string: Newick string
    :param float min_edge_length: Replace any edge length shorter than this value by this
        value. Unlike newick, tskit doesn't support zero or negative edge lengths, so
        setting this argument to a small value is necessary when importing trees with
        zero or negative lengths.
    :param float span: The span of the tree, and therefore the
        :attr:`~TreeSequence.sequence_length` of the returned tree sequence.
    :param str time_units: The value assigned to the :attr:`~TreeSequence.time_units`
        property of the resulting tree sequence. Default: ``None`` resulting in the
        time units taking the default of :attr:`tskit.TIME_UNITS_UNKNOWN`.
    :param str node_name_key: The metadata key used for the node names. If ``None``
        use the string ``"name"``, as in the example of accessing node metadata above.
        Default ``None``.
    :return: A tree sequence consisting of a single tree.
    """
    trees = newick.loads(string)
    if len(trees) > 1:
        raise ValueError("Only one tree can be imported from a newick string")
    if len(trees) == 0:
        raise ValueError("Newick string was empty")
    tree = trees[0]
    
    tables = tskit.TableCollection(span)
    if time_units is not None:
        tables.time_units = time_units
    if node_name_key is None:
        node_name_key = "name"
    nodes = tables.nodes
    nodes.metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "type": "object",
            "properties": {
                node_name_key: {
                    "type": ["string"],
                    "description": "Name from newick file",
                },
                "comment": {
                    "type": ["string"],
                    "description": "Comment from newick file",
                },
            },
        }
    )
            
    ii = 0
    n_leaf_nodes = 0
    for newick_node in tree.walk():
        if len(newick_node.descendants) == 0:
            n_leaf_nodes += 1
        newick_node.length *= multiplier

    id_map = {}

    def get_or_add_node(newick_node, time):
        if newick_node not in id_map:
            flags = tskit.NODE_IS_SAMPLE if len(newick_node.descendants) == 0 else 0
            metadata = {}
            if newick_node.name:
                metadata[node_name_key] = newick_node.name
            if newick_node.comment:
                metadata["comment"] = newick_node.comment
                
            if newick_node.name:
                id_map[newick_node] = tables.nodes.add_row(
                    flags=flags, time = time, metadata=metadata, individual = int(newick_node.name)
                )
            else:
                id_map[newick_node] = tables.nodes.add_row(
                    flags=flags, time = time, metadata=metadata, individual = -1
                )
        return id_map[newick_node]

    root = next(tree.walk())
    get_or_add_node(root, 0)
    
    edges = []
    for newick_node in tree.walk():
        node_id = id_map[newick_node]
        for child in newick_node.descendants:
            length = max(child.length, min_edge_length)
            if length <= 0:
                raise ValueError(
                    "tskit tree sequences cannot contain edges with lengths"
                    " <= 0. Set min_edge_length to force lengths to a"
                    " minimum size"
                )
            child_node_id = get_or_add_node(child, tables.nodes[node_id].time - length)
            tables.edges.add_row(0, span, node_id, child_node_id)
                        
    # Rewrite node times to fit the tskit convention of zero at the youngest leaf
    nodes = list(tables.nodes.copy())
    youngest = min(tables.nodes.time)
    
    tables.nodes.clear()
    individuals = tables.individuals
    
    for k in range(n_leaf_nodes):
        individuals.add_row(flags = 0)
    
    ii = n_leaf_nodes
    
    ids = []
    for node in nodes:
        if "name" in node.metadata.keys():
            id_ = int(node.metadata["name"])
        else:
            id_ = -1
            
        tables.nodes.append(node.replace(time=node.time - youngest + root.length))
        
    tables.sort()
    
    return tables.tree_sequence()

def tree_to_FW(tree: tskit.Tree):
    """
    Calculates F and W matrices from a tskit Tree object.

    This function is an adaptation of the original code that worked with
    skbio.TreeNode. It computes matrices used in population genetics,
    based on the coalescence times and topology of the tree.

    Args:
        tree: A tskit.Tree object representing a single coalescent tree.

    Returns:
        A tuple containing:
        - F (np.ndarray): A symmetric matrix related to the number of lineages.
        - W (np.ndarray): A vector of time interval lengths.
        - s (np.ndarray): A sorted array of unique coalescence times, plus 0.
    """

    # Get the unique coalescence times (ages of internal nodes) sorted descending
    coalescence_times = sorted(
        list(set(tree.time(u) for u in tree.nodes() if tree.is_internal(u))),
        reverse=True
    )
    s = np.array(coalescence_times + [0.0])

    n = len([u for u in tree.nodes() if tree.is_leaf(u)])

    # Initialize the F matrix
    # The size is (n-1) x (n-1) corresponding to the n-1 coalescence events
    F = np.zeros((n - 1, n - 1))

    # Get start and end times for each branch in the tree.
    # A branch is defined by a node and its parent. The time interval for a
    # branch is (time of parent, time of node).
    start_end = np.array([
        (tree.time(tree.parent(u)), tree.time(u))
        for u in tree.nodes() if tree.parent(u) != tskit.NULL
    ])

    # Fill the diagonal of F with the number of extant lineages [2, 3, ..., n]
    F[np.diag_indices_from(F)] = np.arange(2, n + 1)

    # Get indices for the lower triangle of F (excluding the diagonal)
    i, j = np.tril_indices(F.shape[0], -1)

    # Use NumPy broadcasting to efficiently count branches within time intervals
    # For each (i, j) pair, the time interval is [s[j], s[i]] because s is reverse sorted.
    start_end_b = np.tile(start_end, (len(i), 1, 1))
    start_b = np.tile(s[j], (start_end.shape[0], 1)).T
    end_b = np.tile(s[i], (start_end.shape[0], 1)).T

    # Count how many branches are fully contained within each interval [start, end]
    # A branch (parent_time, child_time) is contained if:
    # parent_time >= start_time AND child_time <= end_time
    # Note: The original code had the logic flipped: child_time <= end and parent_time >= start
    # which is what is implemented here.
    counts = np.sum((start_end_b[:, :, 1] <= end_b) & (start_end_b[:, :, 0] >= start_b), axis=-1)

    # Populate the lower and upper triangles of F with the counts
    F[i, j] = counts
    F[j, i] = counts

    # Calculate W vector using the lower triangle indices (including diagonal)
    i, j = np.tril_indices(F.shape[0])
    W = s[j] - s[i + 1]
    
    return F[i, j], W, s

def tree_to_graph(tree, n = 200):
    """
    Convert a TSKit tree into a node feature array and edge list (graph representation).

    This function converts a binary tree into a graph format suitable for machine learning models.
    It returns node times and mutation counts, along with directed edges.

    Parameters:
        tree (tskit.Tree): A binary coalescent tree.
        n (int): Number of sample (leaf) nodes. Assumes 2n-1 total nodes in the tree.

    Returns:
        x (np.ndarray): Array of shape (2n - 1, 2), where the first column contains node times
                              and the second column contains mutation counts.
        edge_index (np.ndarray): Array of shape (E, 2) specifying directed edges (parent → child).
    """
    tree = tree.split_polytomies()
    g = nx.DiGraph(tree.as_dict_of_dicts())
        
    mutations = list(tree.mutations())    
    
    for node in g.nodes():
        if g.out_degree(node) == 1:
            g.remove_node(node)
            break
    
    # include the sample nodes as the first n nodes
    sample_nodes = list(range(n))
    internal_nodes = sorted([u for u in g.nodes.keys() if u >= n], key = lambda u: tree.time(u))
    
    to = sample_nodes + list(range(n, 2*n - 1))
    
    nodes = sample_nodes + internal_nodes
    g = nx.relabel_nodes(g, dict(zip(nodes, to)))
    
    mut_counts = np.zeros((len(nodes),))
    for mu in mutations:
        ii = nodes.index(mu.node)
        mut_counts[ii] += 1

    X = []
    for node in nodes:
        t = tree.time(node)        
        X.append(t)
        
    X = np.concatenate([np.array(X).reshape(-1, 1), mut_counts.reshape(-1, 1)], 1)
            
    edge_index = np.array(g.edges())
    
    return np.array(X), edge_index

def graph_to_tree(x, edges, offset = 0.01):
    """
    Convert node features and edges into a TSKit tree.

    This function builds a tree from graph data, where node times are used to reconstruct coalescent events.

    Parameters:
        x (np.ndarray): Node feature matrix of shape (2n - 1, k). The first column must be node times.
        edges (array-like): Array of shape (E, 2) specifying edges (parent, child) in the tree.

    Returns:
        tskit.Tree: A TSKit tree object constructed from the node and edge data.
    """
    n_nodes = (x.shape[0] + 1) // 2
    
    x_ = x[:,0]
    x_ = x_[x_ != 0.]
    t_coal = np.sort(x_)
    
    # push branch lengths by a small amount to avoid TSKit errors
    diff = np.diff(t_coal)
    ii = np.where(diff == 0.)[0]
    
    if len(ii) > 0:
        for ii_ in ii:
            t_coal[ii_ + 1] += offset

    tables = tskit.TableCollection(sequence_length = 1000)
    node_table = tables.nodes  # set up an alias, for efficiency
    for k in range(n_nodes):
        node_table.add_row(flags = tskit.NODE_IS_SAMPLE, population = 0, individual = k, time = 0.)
        
    for k in range(n_nodes, n_nodes * 2 - 1):
        node_table.add_row(flags = 0, population = 0, individual = -1, time = t_coal[k - n_nodes])
        
    edge_table = tables.edges
    individuals = tables.individuals
    pops = tables.populations
    
    for k in range(n_nodes):
        individuals.add_row(flags = 0)
    
    pops.add_row(metadata = b"{'description': '', 'name': 'pop_000'}")

    for e0, e1 in edges:
        edge_table.add_row(left = 0., right = 1000., parent = e0, child = e1)
    
    tables.sort()
    ts_tree = tables.tree_sequence().first(sample_lists = True)
    
    return ts_tree

def distmat_to_tree(D, metric = 'euclidean', method = 'single', transform = None):
    """
    Construct a TSKit tree from a distance matrix using hierarchical clustering.

    Parameters:
        D (np.ndarray): Either a condensed 1D distance matrix or a 2D array of observations.
        metric (str): Distance metric to use for hierarchical clustering (default: 'euclidean').

    Returns:
        tskit.Tree: A binary coalescent tree reconstructed from the distance matrix.
    """
    edges = []
    Z = linkage(D, metric = metric, method = method)
    
    n = squareform(D).shape[0]
    
    if transform:
        Z[:,2] = transform(Z[:,2])
    
    # parents
    ii = np.array(range(n, 2 * n - 1)).reshape(-1, 1)
    
    edges.extend(np.concatenate([ii, Z[:,0].reshape(-1, 1)], 1).astype(np.int32))
    edges.extend(np.concatenate([ii, Z[:,1].reshape(-1, 1)], 1).astype(np.int32))
        
    x = np.zeros((n, 1))
    x = np.concatenate([x, np.expand_dims(Z[:,-2], -1) / 2.])
    
    # we use the distance divided by 2 as the branch length 
    return graph_to_tree(x, edges), Z

def tree_to_distmat(tree, node_dict = None):
    """
    Convert a TSKit tree into a condensed genealogical distance matrix.

    This function computes the pairwise genealogical distances between all sample nodes
    and returns the upper-triangular condensed form of the matrix.

    Parameters:
        tree (tskit.Tree): A binary coalescent tree with sample nodes.

    Returns:
        np.ndarray: A 1D condensed distance matrix suitable for input to functions like `scipy.cluster.hierarchy.linkage`.
    """
    tree = tree.split_polytomies()
            
    leaf_nodes = [u for u in tree.nodes() if tree.is_leaf(u)]

    if node_dict is None:
        leaf_nodes = range(len(leaf_nodes))
    else:
        leaf_nodes = [node_dict[u] for u in range(len(leaf_nodes))]
    D = np.zeros((len(leaf_nodes) * (len(leaf_nodes) - 1) // 2,))

    for ix, (i, j) in enumerate(itertools.combinations(leaf_nodes, 2)):
        D[ix] = tree.distance_between(i, j)
        
    # condense the matrix and return
    return D
    
def pad_sequences(sequences, max_length=None, padding_value=0):
    """
    Pad a list of 2D arrays (sequences) to the same number of rows (sites) using a given padding value.

    Parameters:
        sequences (list of np.ndarray): List of 2D arrays with shape (n_sites, n_samples).
        max_length (int, optional): Desired number of rows to pad to. If None, the max length across sequences is used.
        padding_value (float or int, optional): Value used for padding. Default is 0.

    Returns:
        np.ndarray: Array of shape (batch_size, max_sites, n_samples) with padded sequences.
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            ii = np.random.choice(range(len(seq) - max_length))
            seq = seq[ii:ii + max_length]
            padded_sequences.append(seq)
        else:
            padded_seq = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant', constant_values=padding_value)
            padded_sequences.append(padded_seq)

    return np.array(padded_sequences)


"""
x: (ind, sites) genotype matrix
pos: (sites,) array of positions
y: optional (same shape as x) for segmentation tasks
pop_sizes: tuple (n0, n1) or (n, )
out_shape: (n_pops, n_ind, n_sites) intended for the output.  If the genotype matrixs length > n_sites it is randomly cropped, 
    if < it is zero padded to n_sites 
metric: distance metric to use for sorting and/or matching
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
mode: [seriate_match (only for two populations, seriates the first population and matches it to chroms in the second),
       seriate (order individuals via the seriation algorithm and the given distance metric, see https://github.com/src-d/seriate),
       pad (pad the matrix on the site axis to the given size with no sorting. the number of individuals in outshape is ignored)]
"""
def format_matrix(x, pos, pop_sizes, y = None, 
                  out_shape = (2, 32, 128), 
                  metric = 'cosine', mode = 'seriate'):
    """
    Format genotype matrix into a standardized shape, optionally sorting or matching individuals using seriation.

    Parameters:
        x (np.ndarray): Genotype matrix of shape (n_individuals, n_sites).
        pos (array-like): Genomic site positions of shape (n_sites,).
        pop_sizes (tuple): Tuple indicating population sizes (s0, s1) or (s0,).
        y (np.ndarray, optional): Label or segmentation matrix with the same shape as x.
        out_shape (tuple): Target output shape (n_pops, n_ind, n_sites).
        metric (str): Distance metric for seriation and matching.
        mode (str): Mode of formatting, options are:
            - 'seriate_match': Seriate one population and match to another using the Hungarian algorithm.
            - 'seriate': Seriate individuals within one population.
            - 'pad': Only pad/crop without sorting.

    Returns:
        tuple:
            - np.ndarray: Formatted genotype matrix of shape (n_pops, n_ind, n_sites) or (n_ind, n_sites).
            - np.ndarray: Modified site positions of shape (n_sites,).
            - np.ndarray or None: Modified label/segmentation data if provided.
    """
    if len(pop_sizes) == 1:
        s0 = pop_sizes[0]
        s1 = 0
        
    else:         
        s0, s1 = pop_sizes
    n_pops, n_ind, n_sites = out_shape
            
    pos = np.array(pos)
    
    if x.shape[0] != s0 + s1:
        print('have x with incorrect shape!: {} vs expected {}'.format(x.shape[0], s0 + s1))
        return None, None
    
    if mode == 'seriate_match':
        x0 = x[:s0,:]
        x1 = x[s0:s0 + s1,:]
        
        if y is not None:
            y0 = y[:s0,:]
            y1 = y[s0:s0 + s1,:]
        
        # upsample to the number of individuals
        if s0 != n_ind:
            ii = np.random.choice(range(s0), n_ind)
            x0 = x0[ii,:]
            
            if y is not None:
                y0 = y0[ii,:]

        if s1 != n_ind:
            ii = np.random.choice(range(s1), n_ind)
            x1 = x1[ii,:]
            
            if y is not None:
                y1 = y1[ii,:]
 
        if x0.shape[1] > n_sites:
            ii = np.random.choice(range(x0.shape[1] - n_sites))
            
            x0 = x0[:,ii:ii + n_sites]
            x1 = x1[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
            
            if y is not None:
                y0 = y0[:,ii:ii + n_sites]
                y1 = y1[:,ii:ii + n_sites]
        else:
            to_pad = n_sites - x0.shape[1]
        
            if to_pad % 2 == 0:
                x0 = np.pad(x0, ((0,0), (to_pad // 2, to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2, to_pad // 2)))
                
                if y is not None:
                    y0 = np.pad(y0, ((0,0), (to_pad // 2, to_pad // 2)))
                    y1 = np.pad(y1, ((0,0), (to_pad // 2, to_pad // 2)))
                
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x0 = np.pad(x0, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                
                if y is not None:
                    y0 = np.pad(y0, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                    y1 = np.pad(y1, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
    
        # seriate population 1
        D = squareform(pdist(x0, metric = metric))
        D[np.isnan(D)] = 0.
        
        ii = seriate(D, timeout = 0.)
        
        x0 = x0[ii]
        
        if y is not None:
            y0 = y0[ii]
        
        D = cdist(x0, x1, metric = metric)
        D[np.isnan(D)] = 0.
        
        i, j = linear_sum_assignment(D)
        
        x1 = x1[j]
        
        if y is not None:
            y1 = y1[j]
        
        x = np.concatenate([np.expand_dims(x0, 0), np.expand_dims(x1, 0)], 0)
        if y is not None:
            y = np.concatenate([np.expand_dims(y0, 0), np.expand_dims(y1, 0)], 0)
        
    elif mode == 'pad':
        if x.shape[1] > n_sites:
            ii = np.random.choice(range(x.shape[1] - n_sites))
            
            x = x[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
        else:
            to_pad = n_sites - x.shape[1]

            if to_pad % 2 == 0:
                x = np.pad(x, ((0,0), (to_pad // 2, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x = np.pad(x, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
    
        
    elif mode == 'seriate': # one population
        if x.shape[1] > n_sites:
            ii = np.random.choice(range(x.shape[1] - n_sites))
            
            x = x[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
        else:
            to_pad = n_sites - x.shape[1]

            if to_pad % 2 == 0:
                x = np.pad(x, ((0,0), (to_pad // 2, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x = np.pad(x, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
                
        D = squareform(pdist(x, metric = metric))
        D[np.isnan(D)] = 0.
        
        ii = seriate(D, timeout = 0.)
        
        x = x[ii,:]
        
    return x, pos, y

"""
Converts a genotype matrix (n_samples, n_sites) to a unique site histogram where each row of the resulting array
is a unique column of the input and the last column of the array is the proportion of the data made up by that site.
"""
def to_unique(X):
    """
    Identify unique site patterns across a genotype matrix and compute their frequencies.

    Parameters:
        X (np.ndarray): Input matrix of shape (n_individuals, n_sites).

    Returns:
        np.ndarray: Array of unique patterns with frequencies concatenated with shape (n_unique_patterns, n_individuals + 1), where the last column is the normalized frequency.
    """
    site_hist = dict()
    
    ix = 0
    ii = dict()
    
    indices = []
    for k in range(X.shape[1]):
        x = X[:,k]
        h = ''.join(x.astype(str))
        if h in site_hist.keys():
            site_hist[h] += 1
        else:
            site_hist[h] = 1
            ii[h] = ix
            
            ix += 1
            
        indices.append(ii[h])
        
    site_hist = {v: k for k, v in site_hist.items()}
    
    ii = np.argsort(list(site_hist.keys()))[::-1]
    indices = [indices[u] for u in indices]
    
    v = sorted(list(site_hist.keys()), reverse = True)
    
    _ = []
    for v_ in v:
        x = site_hist[v_]
        x = np.array(list(map(float, [u for u in x])))
        
        _.append(x)
    
    x = np.array(_)
    v = np.array(v, dtype = np.float32).reshape(-1, 1)
    v /= np.sum(v)
        
    x = np.concatenate([x, v], -1)
    
    return x

def seriate_spectral(x, C): 
    """
    Reorder rows in a matrix using spectral seriation (Fiedler vector approach).

    Parameters:
        x (np.ndarray): Data matrix of shape (n_samples, n_features).

    Returns:
        tuple:
            - np.ndarray: Spectrally ordered matrix.
            - np.ndarray: Indices of the original rows in the new order.
    """
    C[np.where(np.isnan(C))] = 0.

    C = np.diag(C.sum(axis = 1)) - C
    _, v = eigs(C, k = 2, which = 'SM')

    f = v[:,1]
    ix = np.argsort(np.real(f))

    x = x[ix,:]
    
    return x, ix

def chamfer_distance_1d(set_a, set_b):
    """
    Calculates the 1D Chamfer distance between two sets of points.

    Args:
        set_a (np.ndarray): A 1D NumPy array representing the first set of points.
        set_b (np.ndarray): A 1D NumPy array representing the second set of points.

    Returns:
        float: The 1D Chamfer distance between set_a and set_b.
    """

    # Ensure inputs are NumPy arrays
    set_a = np.asarray(set_a).flatten()
    set_b = np.asarray(set_b).flatten()

    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0  # Or handle as an error, depending on desired behavior

    # Calculate distances from set_a to set_b
    dist_a_to_b = 0.0
    for point_a in set_a:
        min_dist = np.min(np.abs(point_a - set_b))
        dist_a_to_b += min_dist

    # Calculate distances from set_b to set_a
    dist_b_to_a = 0.0
    for point_b in set_b:
        min_dist = np.min(np.abs(point_b - set_a))
        dist_b_to_a += min_dist

    # The Chamfer distance is the sum of these two unidirectional distances
    chamfer_dist = dist_a_to_b + dist_b_to_a
    return chamfer_dist


    