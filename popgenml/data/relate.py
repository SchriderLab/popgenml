# -*- coding: utf-8 -*-

import os
import tempfile
from .io_ import write_to_ms
from .trees import PGTreeSequence

import time
import copy
import glob
import numpy as np
from skbio.tree import TreeNode
import importlib.resources

RELATE_PATH = 'Relate'

# Grab the file path from the package and convert the Traversable object to a string
rscript_path = str(importlib.resources.files('popgenml').joinpath('scripts', 'ms2haps.py'))

rcmd = 'cd {3} && python3 ' + rscript_path + ' {0} {1} {2}'
relate_cmd = 'cd {6} && ' + RELATE_PATH + ' --mode {7} -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'

def make_FW_rep(root, sample_sizes):
    if len(sample_sizes) > 1:
        topo_tips = [u for u in root.postorder() if u.is_tip()]
    
        pop_vector = [u.pop for u in topo_tips]
    else:
        pop_vector = None

    non_zero_ages = []

    ages = np.zeros(root.count())
    
    for node in root.traverse():
        ages[node.id] = node.age

        if node.age > 0.:
            non_zero_ages.append(node.age)

    non_zero_ages = sorted(non_zero_ages, reverse=True)

    # indexed by assinged id
    F = np.zeros((sum(sample_sizes) - 1, sum(sample_sizes) - 1))

    children = root.children
    c1, c2 = root.children

    extant = np.array(range(2, sum(sample_sizes) + 1))
    start_end = []

    todo = [root]
    while len(todo) != 0:
        root = todo[-1]
        del todo[-1]

        t_coal = ages[root.id]

        if root.has_children():
            if len(root.children) == 1:
                continue
            
            c1, c2 = root.children

            start_end.append((t_coal, c1.age))

            if not c1.is_tip():
                todo.append(c1)

            start_end.append((t_coal, c2.age))

            if not c2.is_tip():
                todo.append(c2)

    start_end = np.array(start_end)
    s = np.array(non_zero_ages + [0.])
    
    F[list(range(F.shape[0])), list(range(F.shape[0]))] = extant

    i, j = np.tril_indices(F.shape[0], -1)

    start_end_ = np.tile(start_end, (len(i), 1, 1))
    start = np.tile(s[j], (start_end.shape[0], 1)).T
    end = np.tile(s[i + 1], (start_end.shape[0], 1)).T

    _ = np.sum((start_end_[:,:,1] <= end) & (start_end_[:,:,0] >= start), axis = -1)

    F[i, j] = _
    F[j, i] = _
    
    i, j = np.tril_indices(F.shape[0])

    W = s[j] - s[i + 1]
    
    return F, W, pop_vector, s

def parse_line(line, s0, s1):
    nodes = []
    parents = []
    lengths = []
    n_mutations = []
    regions = []
    
    edges = []
    
    # new tree
    line = line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
    line = line.split(' ')[:-1]
    
    start_snp = int(line[0])
    
    sk_nodes = dict()
    mut_dict = dict()
    try:
        for j in range(2, len(line), 5):
            nodes.append((j - 1) // 5)
            
            p = int(line[j])
            if p not in sk_nodes.keys():
                sk_nodes[p] = TreeNode(name = str(p))
                
            length = float(line[j + 1])
            
            if (j - 1) // 5 not in sk_nodes.keys():
                sk_nodes[(j - 1) // 5] = TreeNode(name = str((j - 1) // 5), parent = sk_nodes[p], length = length)
                sk_nodes[p].children.append(sk_nodes[(j - 1) // 5])
            else:
                sk_nodes[(j - 1) // 5].parent = sk_nodes[p]
                sk_nodes[(j - 1) // 5].length = length
                sk_nodes[p].children.append(sk_nodes[(j - 1) // 5])
                
            parents.append(p)
            lengths.append(float(line[j + 1]))
            n_mutations.append(float(line[j + 2]))
            
            mut_dict[nodes[-1]] = n_mutations[-1]
            regions.append((int(line[j + 3]), int(line[j + 4])))
            
            edges.append((parents[-1], nodes[-1]))
    except:
        return
    
    lengths.append(0.)
    
    root = None
    for node in sk_nodes.keys():
        node = sk_nodes[node]
        if node.is_root():
            root = node
            break
        
    root = root.children[0]
    T_present = [u for u in root.traverse() if u.is_tip()]
    
    T_names = sorted([int(u.name) for u in root.postorder() if u.is_tip()])
    
    data = dict()
    
    # pop_labels + mutation
    if s1 > 0:
        for node in T_names[:s0]:
            data[node] = np.array([0., 1., 0., 0., mut_dict[node]])
        
        for node in T_names[s0:s0 + s1]:
            data[node] = np.array([0., 0., 1., 0., mut_dict[node]])
    else:
        for node in T_names:
            data[node] = np.array([0., 1., 0., mut_dict[node]])
    
    if s1 > 0:
        pop_vector = [data[u][1] for u in [int(u.name) for u in root.postorder() if u.is_tip()]]
    else:
        pop_vector = None

    edges = []
    while len(T_present) > 0:
        _ = []
        
        for c in T_present:
            c_ = int(c.name)
            branch_l = c.length
            
            p = c.parent
            
            if p is not None:
            
                    
                p = int(c.parent.name)
                if p < 0:
                    continue
                
                if p not in data.keys():
                    d = np.zeros(data[c_].shape)
                    # pop_label
                    d[-2] = 1.
                    # time
                    d[0] = data[c_][0] + branch_l
                    
                    if p in mut_dict.keys():
                        d[-1] = mut_dict[p]

                    data[p] = d
                
                    _.append(c.parent)
            
                edges.append((p, c_))
               
        T_present = copy.copy(_)
        
    X = []

    for node in nodes:
        X.append(data[node])
        sk_nodes[node].age = data[node][0]
                                  
    root.assign_ids()
            
    X = np.array(X)
    edges = edges[:X.shape[0]]

    return root, start_snp, X, edges, pop_vector, lengths

def read_anc(anc_file, pop_sizes = (40,0)):
    s0, s1 = pop_sizes
    sample_sizes = [u for u in pop_sizes if u != 0]
    
    anc_file = open(anc_file, 'r')
    
    # we're at the beginning of a block
    for k in range(3):
        line = anc_file.readline()
    
    while not '(' in line:
        line = anc_file.readline()
        if line.decode('utf-8') == '':
            break
        
    lines = []            
    while '(' in line:
        lines.append(line)
        line = anc_file.readline()
        
    X = []
    edge_indices = []
    branch_lengths = []
        
    snps = []
    for ij in range(len(lines)):
        line = lines[ij]
        root, snp, x, edges, pop_vector, lengths = parse_line(line, s0, s1)
        
        branch_lengths.append(lengths)
        
        snps.append(snp)
        
        X.append(x)
        edge_indices.append(edges)

    anc_file.close()

    return X, edge_indices, snps, branch_lengths

import tskit

def read_anc_to_tskit_trees(anc_file):
    tree_blocks = []
    
    with open(anc_file, 'r') as f:
        # Skip the first 3 lines (Relate .anc header blocks)
        for _ in range(3):
            f.readline()
            
        for line in f:
            if '(' not in line:
                continue
                
            # Clean and tokenize the line
            line_clean = line.replace(':', ' ').replace('(', '').replace(')', '').strip()
            if not line_clean:
                continue
                
            parts = line_clean.split()
            start_snp = int(parts[0])
            
            edges = []
            branch_lengths = {}
            
            for j in range(1, len(parts), 5):
                try:
                    child = (j - 1) // 5
                    parent = int(parts[j])
                    length = float(parts[j + 1])
                    
                    if parent >= 0:
                        edges.append((parent, child))
                        
                    branch_lengths[child] = length
                except IndexError:
                    break
                    
            tree_blocks.append({
                'start_snp': start_snp,
                'edges': edges,
                'lengths': branch_lengths
            })

    if not tree_blocks:
        return []

    # 1. Define bounds and scaling factor
    max_start = float(tree_blocks[-1]['start_snp'])
    seq_len = max_start + 1.0  
    
    # 2. Identify samples (tips)
    children = set()
    parents = set()
    for block in tree_blocks:
        for p, c in block['edges']:
            children.add(c)
            parents.add(p)
            
    samples = children - parents
    
    if not children and not parents:
        max_node_id = 0
    else:
        max_node_id = max(max(children), max(parents))

    # 3. Resolve consistent node times (tskit enforces: parent_time > child_time)
    node_times = {n: 0.0 for n in samples}
    for _ in range(max_node_id + 1):  
        changed = False
        for block in tree_blocks:
            for p, c in block['edges']:
                if c in node_times:
                    t = node_times[c] + max(block['lengths'][c], 1e-6)
                    if p not in node_times or node_times[p] < t:
                        node_times[p] = t
                        changed = True
        if not changed:
            break

    # 4. Build the tskit TableCollection
    tables = tskit.TableCollection(sequence_length=1.0)

    for i in range(max_node_id + 1):
        if i in samples:
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=node_times.get(i, 0.0))
        else:
            tables.nodes.add_row(flags=0, time=node_times.get(i, 0.0))

    # 5. Populate edges
    for i, block in enumerate(tree_blocks):
        curr_start = block['start_snp']
        left = curr_start / seq_len
        
        if i < len(tree_blocks) - 1:
            next_start = tree_blocks[i + 1]['start_snp']
            right = next_start / seq_len
        else:
            right = 1.0
            
        for p, c in block['edges']:
            tables.edges.add_row(left=left, right=right, parent=p, child=c)

    # Sort and finalize tree sequence
    tables.sort()
    ts = tables.tree_sequence()

    # 6. Yield Tree objects, their scaled spans, and dynamic SNP index lists
    # 6. Yield Tree objects, their scaled spans, and dynamic SNP index lists
    result = []
    for tree in ts.trees():
        local_tables = tskit.TableCollection(sequence_length=1.0)
        
        # Find strictly the nodes connected to the samples in THIS interval
        samples = list(tree.samples())
        active_nodes = set(samples)
        for u in samples:
            curr = u
            while tree.parent(curr) != tskit.NULL:
                curr = tree.parent(curr)
                active_nodes.add(curr)
                
        internal_nodes = [u for u in active_nodes if u not in samples]
        internal_nodes.sort(key=lambda u: tree.time(u))
        
        node_map = {}
        
        # 1. Map samples strictly to 0...n-1 so they match the simulated tree
        for u in samples:
            node_map[u] = local_tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=tree.time(u))
            
        # 2. Add connected internal nodes
        for u in internal_nodes:
            node_map[u] = local_tables.nodes.add_row(flags=0, time=tree.time(u))
            
        # 3. Add edges
        for u in active_nodes:
            parent = tree.parent(u)
            if parent != tskit.NULL and parent in active_nodes:
                local_tables.edges.add_row(
                    left=0.0, right=1.0,
                    parent=node_map[parent], child=node_map[u]
                )
                
        local_tables.sort()
        clean_ts = local_tables.tree_sequence()
        
        # Enable sample lists here
        clean_tree = clean_ts.first(sample_lists=True)
        
        # 4. Handle Relate uncoalesced lineages
        if clean_tree.num_roots > 1:
            dummy_tables = clean_ts.dump_tables()
            roots = [u for u in clean_tree.nodes() if clean_tree.parent(u) == tskit.NULL]
            
            # Place dummy root older than all existing roots
            max_time = max([clean_tree.time(r) for r in roots])
            dummy_root = dummy_tables.nodes.add_row(flags=0, time=max_time + 1.0)
            
            for r in roots:
                dummy_tables.edges.add_row(left=0.0, right=1.0, parent=dummy_root, child=r)
                
            dummy_tables.sort()
            
            # Enable sample lists here as well
            clean_tree = dummy_tables.tree_sequence().first(sample_lists=True)

        # Finalize spans and SNPs
        span_tuple = (tree.interval.left, tree.interval.right)
        start_idx = int(round(tree.interval.left * seq_len))
        end_idx = int(round(tree.interval.right * seq_len))
        snps = list(range(start_idx, end_idx))
        
        result.append((clean_tree, span_tuple, snps))
                
    return [u[0] for u in result], [u[1] for u in result], [u[2] for u in result]

# Example usage:
# tskit_trees = read_anc_to_tskit_trees("my_data.anc")
# for tree, (start, end) in tskit_trees:
#     print(f"Tree spans from {start:.3f} to {end:.3f}")
#     print(f"Total Roots: {tree.num_roots}, Total Edges: {tree.num_edges}")
    
def harmonic_number(n):
    return np.sum(np.array(range(1, n), dtype = np.float32) ** -1)

def get_haps_positions(filename):
    """
    Parses a .haps file and returns a list of SNP positions.
    Assumes standard Oxford format: 
    CHR ID POS REF ALT G1 G2 ...
    """
    positions = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            # Split by whitespace and take the 3rd element (index 2)
            parts = line.split()
            if len(parts) > 2:
                positions.append(int(parts[2]))
    return positions

def relate(X, sites, n_samples, mu, r, L, N = None, diploid = False, verbose = False,
           return_graph = False, odir = None):
    r"""
    Run RELATE (a genealogy-based inference method) on simulated or empirical binary haplotype data.

    This function writes the input data in `ms` format, constructs a genetic map, executes the RELATE
    command-line pipeline, and parses the output to return a tree sequence object.

    Parameters:
        X (np.ndarray): Binary haplotype array of shape (n_individuals, n_sites).
        sites (array-like): Array of genomic positions (length = n_sites).
        n_samples (int): Number of haploid samples (individuals * 2 for diploids).
        mu (float): Mutation rate per base pair.
        r (float): Recombination rate per base pair.
        L (int): Total sequence length in base pairs.
        N (int, optional): Effective population size. If not provided, estimated using Watterson's theta calculated from the number of segregating sites and mu.
        diploid (bool, optional): Whether the input samples are diploid (default is False).
        verbose (bool, optional): Whether to print RELATE's output to the terminal (default is False).
        return_graph (bool, optional): Placeholder (not currently used) for returning inferred ARG.
        odir (str, optional): Path to an output directory to save RELATE files. If None, a temporary directory is used and cleaned up automatically.

    Returns:
        PGTreeSequence: An object encapsulating the inferred tree sequence, containing the parsed tskit trees and their corresponding genomic intervals (spans).

    Notes:
        - If `odir` is not specified, this function creates and cleans up a temporary directory to run RELATE.
        - Assumes RELATE and helper binaries (`relate_cmd`, etc.) are properly configured and in scope.
        - Input data is written in ms-format; RELATE’s `.haps` and `.sample` files are auto-generated.
        - Genomic map is generated with a constant recombination rate.

    Requires:
        - External RELATE binary and pre-configured command templates: `rcmd`, `relate_cmd`.
        - Supporting functions: `write_to_ms`, `read_anc`.
    """
    if N is None:
        N = (X.shape[1] / (4 * mu * L)) / harmonic_number(X.shape[0])
    
    temp_dir = tempfile.TemporaryDirectory()
    
    if odir is not None:
        temp_dir.name = odir
    
    odir = os.path.join(temp_dir.name, 'relate')
    os.system('mkdir -p {}'.format(odir))
    
    ms_file = os.path.join(temp_dir.name, 'sim.msOut')
    write_to_ms(ms_file, X, sites, [0])
    time.sleep(0.001)
    
    tag = ms_file.split('/')[-1].split('.')[0]
    cmd_ = rcmd.format(os.path.abspath(ms_file), tag, L, odir)

    if verbose:
        print(cmd_)

    os.system(cmd_)
    
    map_file = ms_file.replace('.msOut', '.map')
    
    # for constant recombination rate across the entire chrom
    ofile = open(map_file, 'w')
    ofile.write('pos COMBINED_rate Genetic_Map\n')
    ofile.write('0 {} 0\n'.format(r * 10**8))
    ofile.write('{0} {1} {2}\n'.format(L, r * 10**8, r * L * 100))
    ofile.close()
    
    haps = list(map(os.path.abspath, sorted(glob.glob(os.path.join(odir, '*.haps')))))
    samples = list(map(os.path.abspath, [u.replace('.haps', '.sample') for u in haps if os.path.exists(u.replace('.haps', '.sample'))]))
    
    # we need to rewrite the haps files (for haploid organisms)
    if diploid:
        for sample in samples:
            f = open(sample, 'w')
            f.write('ID_1 ID_2 missing\n')
            f.write('0    0    0\n')
            for k in range(n_samples // 2):
                f.write('UNR{} UNR{} 0\n'.format(k + 1, k + 1))
                
            f.close()

    else:
        # we need to rewrite the haps files (for haploid organisms)
        for sample in samples:
            f = open(sample, 'w')
            f.write('ID_1 ID_2 missing\n')
            f.write('0    0    0\n')
            for k in range(int(n_samples)):
                f.write('UNR{} NA 0\n'.format(k + 1))
    
            f.close()
    
    ofile = haps[0].split('/')[-1].replace('.haps', '') + '_' + map_file.split('/')[-1].replace('.map', '').replace(tag, '').replace('.', '')
    if ofile[-1] == '_':
        ofile = ofile[:-1]
    
    cmd_ = relate_cmd.format(mu, 2 * N, haps[0], 
                             samples[0], os.path.abspath(map_file), 
                             ofile, odir, 'All')
    if not verbose:
        cmd_ += ' >/dev/null 2>&1'
        
    else:
        print(cmd_)
    
    os.system(cmd_)

    anc_file = os.path.join(odir, '{}.anc'.format(ofile))
    
    trees, intervals, snps = read_anc_to_tskit_trees(anc_file)
    
    if odir is None:
        temp_dir.cleanup()

    return PGTreeSequence(trees, intervals)
