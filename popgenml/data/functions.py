import gzip
import numpy as np
import tempfile

import os
import time
import glob
from skbio.tree import TreeNode
import copy

rscript_path = os.path.join(os.getcwd(), 'include/relate/bin/RelateFileFormats')
rcmd = 'cd {3} && ' + rscript_path + ' --mode ConvertFromVcf --haps {0} --sample {1} -i {2}'

relate_path = os.path.join(os.getcwd(), 'include/relate/bin/Relate')
relate_cmd = 'cd {6} && ' + relate_path + ' --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'

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
            
                edges.append((c_, p))
               
        T_present = copy.copy(_)
        
    X = []

    for node in nodes:
        X.append(data[node])
        sk_nodes[node].age = data[node][0]
                                  
    root.assign_ids()
            
    X = np.array(X)
    edges = edges[:X.shape[0]]

    return root, start_snp, X, edges, pop_vector

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
    
    current_day_nodes = list(range(sum(pop_sizes)))
        
    lines = []            
    while '(' in line:
        lines.append(line)
        line = anc_file.readline()
        
    try:
        iix = int(line.replace('\n', '').split()[-1]) - 1
    except:
        iix = 0
    
    
    Fs = []
    Ws = []
    coal_times = []
    
    pop_vectors = []
    
    t0 = time.time()
    
    snps = []
    for ij in range(len(lines)):
        line = lines[ij]
        root, snp, _, _, pop_vector = parse_line(line, s0, s1)
             
        snps.append(snp)
        
        F, W, _, t_coal = make_FW_rep(root, sample_sizes)
        coal_times.append(t_coal)
        
        i, j = np.tril_indices(F.shape[0])
        F = F[i, j]
        
        Fs.append(F)
        Ws.append(W)
        
        if pop_vector is not None:
            pop_vectors.append(pop_vector)
            
    Fs = np.array(Fs)
    Ws = np.array(Ws)
    
    anc_file.close()
    
    return Fs, Ws, snps, pop_vectors, np.array(coal_times)

def write_to_ms(ofile, X, sites, params):
    ofile = open(ofile, 'w')
    
    header = '// ' + ' '.join(['{0:04f}'.format(u) for u in params]) + '\n'
    ofile.write(header)
    
    n_segsites = X.shape[1]
    ofile.write('segsites: {}\n'.format(n_segsites))
    pos_line = 'positions: ' + ' '.join(['{0:08f}'.format(u) for u in sites]) + '\n'
    ofile.write(pos_line)
    
    for x in X:
        line = ''.join(list(map(str, list(x)))) + '\n'
        ofile.write(line)
        
    #ofile.write('\n')
    ofile.close()

# goes from *.msOut.gz to Relate (FW representation)
def relate(ifile, n_samples, mu, r, N, L):
    # assumes one simulation in the file
    X, Y, pos, params = load_data(ifile)
    
    X = X[0]
    sites = pos[0]
    
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
    for sample in samples:
        f = open(sample, 'w')
        f.write('ID_1 ID_2 missing\n')
        f.write('0    0    0\n')
        for k in range(n_samples // 2):
            f.write('UNR{} UNR{} 0\n'.format(k + 1, k + 1))
    
    f.close()
    
    ofile = haps[0].split('/')[-1].replace('.haps', '') + '_' + map_file.split('/')[-1].replace('.map', '').replace(tag, '').replace('.', '')
    if ofile[-1] == '_':
        ofile = ofile[:-1]
    
    cmd_ = relate_cmd.format(mu, 2 * N, haps[0], 
                             samples[0], os.path.abspath(map_file), 
                             ofile, odir) + ' >/dev/null 2>&1'
    
    os.system(cmd_)
    
    anc_file = os.path.join(odir, '{}.anc'.format(ofile))
    Fs, Ws, snps, coal_times = read_anc(anc_file)
    
    temp_dir.cleanup()

    return Fs, Ws, snps, np.array(sites), X

def make_FW_rep(root, sample_sizes):
    if len(sample_sizes) > 1:
        topo_tips = [u for u in root.postorder() if u.is_tip()]
        topo_ids = [u.id for u in topo_tips]
    
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
    end = np.tile(s[i], (start_end.shape[0], 1)).T

    _ = np.sum((start_end_[:,:,1] <= end) & (start_end_[:,:,0] >= start), axis = -1)

    F[i, j] = _
    F[j, i] = _

    i, j = np.tril_indices(F.shape[0])

    W = s[j] - s[i + 1]
    
    return F, W, pop_vector, s

def split(word):
    return [char for char in word]

######
# generic function for msmodified
# ----------------
# takes a gzipped ms file
# returns a list of genotype matrices, introgressed allele matrices (if *.anc file is provided),
# a list of position vectors, and a list of the parameters listed in the \\ line if any
def load_data(msFile, ancFile = None, n = None, leave_out_last = False):
    msFile = gzip.open(msFile, 'r')

    # no migration case
    if ancFile is not None:
        ancFile = gzip.open(ancFile, 'r')

    ms_lines = [u.decode('utf-8') for u in msFile.readlines()]

    if leave_out_last:
        ms_lines = ms_lines[:-1]

    if ancFile is not None:
        idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]
    else:
        idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]
        
            
    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1] += ['\n']

    if ancFile is not None:
        anc_lines = [u.decode('utf-8') for u in ancFile.readlines()]
    else:
        anc_lines = None
        
    X = []
    Y = []
    P = []
    intros = []
    params = []
    
    for chunk in ms_chunks:
        line = chunk[0]
        params_ = list(map(float, line.replace('\n', '').split('\t')[1:]))
        
        if len(params_) == 0:
            params_ = list(map(float, line.replace('\n', '').split()[1:]))
        
    
        if '*' in line:
            intros.append(True)
        else:
            intros.append(False)
        
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)
        _ = [list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]]
        _ = [u for u in _ if len(u) > 0]
        
        x = np.array(_, dtype = np.uint8)
        
        if x.shape[0] == 0:
            X.append(None)
            Y.append(None)
            P.append(None)
            params.append(None)
            continue
        
        # destroy the perfect information regarding
        # which allele is the ancestral one
        for k in range(x.shape[1]):
            if np.sum(x[:,k]) > x.shape[0] / 2.:
                x[:,k] = 1 - x[:,k]
            elif np.sum(x[:,k]) == x.shape[0] / 2.:
                if np.random.choice([0, 1]) == 0:
                    x[:,k] = 1 - x[:,k]
        
        if anc_lines is not None:
            y = np.array([list(map(int, split(u.replace('\n', '')))) for u in anc_lines[:len(pos)]], dtype = np.uint8)
            y = y.T
            
            del anc_lines[:len(pos)]
        else:
            y = np.zeros(x.shape, dtype = np.uint8)
            
        if len(pos) == x.shape[1] - 1:
            pos = np.array(list(pos) + [1.])
            
        assert len(pos) == x.shape[1]
        
        if n is not None:
            x = x[:n,:]
            y = y[:n,:]
            
        X.append(x)
        Y.append(y)
        P.append(pos)
        params.append(params_)
        
    return X, Y, P, params
