# -*- coding: utf-8 -*-

import os
import tempfile
from popgenml.data.io_ import write_to_ms

import time
import copy
import glob
import numpy as np
from skbio import read
from skbio.tree import TreeNode
from pkg_resources import resource_filename

RELATE_PATH = 'Relate'
rscript_path = 'Rscript {}'.format(os.path.join(resource_filename('popgenml', 'scripts'), 'ms2haps.R'
                                                ))

rcmd = 'cd {3} && ' + rscript_path + ' {0} {1} {2}'
relate_cmd = 'cd {6} && ' + RELATE_PATH + ' --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'

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
    
    Fs = []
    Ws = []
    
    branch_lengths = []
        
    snps = []
    for ij in range(len(lines)):
        line = lines[ij]
        root, snp, x, edges, pop_vector, lengths = parse_line(line, s0, s1)
        
        branch_lengths.append(lengths)
        
        snps.append(snp)
        
        X.append(x)
        edge_indices.append(edges)
            
    Fs = np.array(Fs)
    Ws = np.array(Ws)
    
    anc_file.close()

    return X, edge_indices, snps, branch_lengths
    
def relate(X, sites, n_samples, mu, r, N, L, diploid = False, verbose = False,
           return_graph = False):
    """
    Run RELATE (a genealogy-based inference method) on simulated or empirical binary haplotype data.

    This function writes the input data in `ms` format, constructs a genetic map, executes the RELATE
    command-line pipeline, and parses the output to return coalescent statistics.

    Parameters:
        X (np.ndarray): Binary haplotype array of shape (n_individuals, n_sites).
        sites (array-like): Array of genomic positions (length = n_sites).
        n_samples (int): Number of haploid samples (individuals * 2 for diploids).
        mu (float): Mutation rate per base pair.
        r (float): Recombination rate per base pair.
        N (int): Effective population size.
        L (int): Total sequence length in base pairs.
        diploid (bool, optional): Whether the input samples are diploid (default is False).
        verbose (bool, optional): Whether to print RELATE's output to the terminal (default is False).
        return_graph (bool, optional): Placeholder (not currently used) for returning inferred ARG.

    Returns:
        tuple:
            Fs (np.ndarray): Tree sequence statistics (e.g., branch lengths or features from `.anc` file).
            Ws (np.ndarray): Inferred weights or local ancestries.
            snps (np.ndarray): SNP data extracted from the RELATE output.
            sites (np.ndarray): Positions corresponding to SNPs, copied from input.
            coal_times (np.ndarray): Inferred coalescent times from ancestral trees.

    Notes:
        - This function creates and cleans up a temporary directory to run RELATE.
        - Assumes RELATE and helper binaries (`relate_cmd`, etc.) are properly configured and in scope.
        - Input data is written in ms-format; RELATE’s `.haps` and `.sample` files are auto-generated.
        - Genomic map is generated with a constant recombination rate.

    Requires:
        - External RELATE binary and pre-configured command templates: `rcmd`, `relate_cmd`.
        - Supporting functions: `write_to_ms`, `read_anc`.

    """
    temp_dir = tempfile.TemporaryDirectory()
    
    odir = os.path.join(temp_dir.name, 'relate')
    os.mkdir(odir)
    
    ms_file = os.path.join(temp_dir.name, 'sim.msOut')
    write_to_ms(ms_file, X, sites, [0])
    time.sleep(0.001)
    
    tag = ms_file.split('/')[-1].split('.')[0]
    cmd_ = rcmd.format(os.path.abspath(ms_file), tag, L, odir)

    os.system(cmd_)
    
    map_file = ms_file.replace('.msOut', '.map')
    
    ofile = open(map_file, 'w')
    ofile.write('pos COMBINED_rate Genetic_Map\n')
    ofile.write('0 {} 0\n'.format(r * L))
    ofile.write('{0} {1} {2}\n'.format(L, r * L, r * 10**8))
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
                             ofile, odir)
    
    if not verbose:
        cmd_ += ' >/dev/null 2>&1'
    
    os.system(cmd_)
    
    anc_file = os.path.join(odir, '{}.anc'.format(ofile))
    X, edge_indices, snps, branch_lengths = read_anc(anc_file, pop_sizes = (n_samples, 0))
    
    temp_dir.cleanup()

    return X, edge_indices, snps, branch_lengths