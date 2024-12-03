import gzip
import numpy as np
import tempfile

import os
import time
import glob
from skbio.tree import TreeNode
import copy

from io_ import write_to_ms, load_ms
import sys

from seriate import seriate
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linear_sum_assignment

rscript_path = 'Rscript {}'.format(os.path.abspath('src/data/ms2haps.R'))
rcmd = 'cd {3} && ' + rscript_path + ' {0} {1} {2}'

relate_path = os.path.join(os.getcwd(), 'include/relate/bin/Relate')
relate_cmd = 'cd {6} && ' + relate_path + ' --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'

def find_files(idir, exts = ('.msOut.gz')):
    matches = []
    
    if not os.path.isdir(idir):
        return matches
        
    for root, dirnames, filenames in os.walk(idir):
        filenames = [ f for f in filenames if os.path.splitext(f)[1] in exts ]
        for filename in filenames:
            matches.append(os.path.join(root, filename))
            
    return matches

def pad_sequences(sequences, max_length=None, padding_value=0):
    """Pads sequences to the same length."""

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
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

def parse_fixations(fixationLines):
    fixations = []
    mode = 0
    for line in fixationLines:
        if mode == 0:
            if line.startswith("Mutations"):
                mode = 1
        elif mode == 1:
            if line.startswith("Done with fixations"):
                break
            else:
                tempId, permId, mutType, pos, selCoeff, domCoeff, originSubpop, originGen, fixationGen = line.strip().split()
                fixations.append((mutType, int(pos)))
    return fixations

def process_slim_sample(sampleText, locs, genomes, sampleSize1):
    mode = 0
    idMapping, mutTypes, tempIdToPos = {}, {}, {}
    introgressedAlleles = []
    fixationLines = []
    for line in sampleText:
        # sys.stderr.write(line+"\n")
        if mode == 0:
            if line.startswith("Emitting fixations"):
                mode = 1
        elif mode == 1:
            fixationLines.append(line.strip())
            if line.startswith("Done with fixations"):
                fixations = parse_fixations(fixationLines)
                mode = 2
        elif mode == 2:
            if line.startswith("Mutations"):
                mode = 3
        elif mode == 3:
            if line.startswith("Genomes"):
                mode = 4
            else:
                tempId, permId, mutType, pos, selCoeff, domCoeff, originSubpop, originGen, numCopies = line.strip().split()
                pos, originSubpop, originGen = int(pos), int(
                    originSubpop.lstrip("p")), int(originGen)
                if mutType == "m4":
                    mutType = "m3"

                if mutType == "m3":
                    if not pos in locs:
                        locs[pos] = {}
                    locs[pos][permId] = 1
                elif mutType in ["m1", "m2"]:
                    tempIdToPos[tempId] = pos
                else:
                    print(tempId, permId, mutType, pos, selCoeff,
                          domCoeff, originSubpop, originGen, numCopies)
                    sys.exit(
                        "Encountered mutation type other than m1, m2, or m3. ARRRRGGGGHHHHHHH!!!!!!\n")
                idMapping[tempId] = permId
                mutTypes[tempId] = mutType

        elif mode == 4:
            line = line.strip().split()
            gId, auto = line[:2]
            mutLs = line[2:]

            introgressedAlleles.append([])
            for tempId in mutLs:
                if mutTypes[tempId] == "m1" and len(genomes) >= sampleSize1:
                    introgressedAlleles[-1].append(tempIdToPos[tempId])
                elif mutTypes[tempId] == "m2" and len(genomes) < sampleSize1:
                    introgressedAlleles[-1].append(tempIdToPos[tempId])
            for fixationType, fixationPos in fixations:
                if fixationType == "m1" and len(genomes) >= sampleSize1:
                    introgressedAlleles[-1].append(fixationPos)
                elif fixationType == "m2" and len(genomes) < sampleSize1:
                    introgressedAlleles[-1].append(fixationPos)

            genomes.append(set([idMapping[x]
                           for x in mutLs if mutTypes[x] == "m3"]))
    return introgressedAlleles

def getFreq(mutId, genomes):
    mutCount = 0
    for genome in genomes:
        if mutId in genome:
            mutCount += 1
    return mutCount

def processedIntrogressedAlleles(ls):
    ls.sort()
    if len(ls) == 0:
        return []
    else:
        runs = []
        runStart = ls[0]
        for i in range(1, len(ls)):
            if ls[i] > ls[i-1]+1:
                runEnd = ls[i-1]
                runs.append((runStart, runEnd))
                runStart = ls[i]
        runs.append((runStart, ls[-1]))
    return runs

def removeMonomorphic(allMuts, genomes):
    newMuts = []
    newLocI = 0
    for locI, loc, contLoc, mutId in allMuts:
        freq = getFreq(mutId, genomes)
        if freq > 0 and freq < len(genomes):
            newMuts.append((newLocI, loc, contLoc, mutId))
            newLocI += 1
    return newMuts

def buildMutationPosMapping(mutLocs, physLen):
    mutMapping = []
    mutLocs.sort()
    for i in range(len(mutLocs)):
        pos, mutId = mutLocs[i]
        mutMapping.append((i, pos, pos/physLen, mutId))
    return mutMapping

def buildPositionsList(muts, discrete=True):
    positions = []
    for locationIndex, locationDiscrete, locationContinuous, mutId in muts:
        if discrete:
            positions.append(locationDiscrete)
        else:
            positions.append(locationContinuous)
    return positions

def read_slim(output, sampleSize1, physLen, return_introgressed = True):
    numSamples = 1
    
    totSampleCount = 0
    for line in output.decode("utf-8").split("\n"):
        if line.startswith("Sampling at generation"):
            totSampleCount += 1
    samplesToSkip = totSampleCount-numSamples
    #sys.stderr.write("found {} samples and need to skip {}\n".format(totSampleCount, samplesToSkip))
    
    mode = 0
    samplesSeen = 0
    locs = {}
    genomes = []
    for line in output.decode("utf-8").split("\n"):
        if mode == 0:
            if line.startswith("migProb") or line.startswith("migTime"):
                sys.stderr.write(line+"\n")
            if line.startswith("splitTime"):
                splitTime = int(line.strip().split("splitTime: ")[1])
            if line.startswith("migTime"):
                migTime = int(line.strip().split("migTime: ")[1])
            if line.startswith("Sampling at generation"):
                samplesSeen += 1
                if samplesSeen >= samplesToSkip+1:
                    sampleText = []
                    mode = 1
        elif mode == 1:
            if line.startswith("Done emitting sample"):
                mode = 0
                if return_introgressed:
                    introgressedAlleles = process_slim_sample(
                        sampleText, locs, genomes, sampleSize1)
                
            else:
                sampleText.append(line)
        if "SEGREGATING" in line:
            sys.stderr.write(line+"\n")
            
    newMutLocs = []
    for mutPos in locs:
        if len(locs[mutPos]) == 1:
            mutId = list(locs[mutPos].keys())[0]
            newMutLocs.append((mutPos, mutId))
        else:

            for mutId in locs[mutPos]:
                newMutLocs.append((mutPos, mutId))

    allMuts = buildMutationPosMapping(newMutLocs, physLen)
    polyMuts = removeMonomorphic(allMuts, genomes)
    positions = buildPositionsList(polyMuts)
    haps = []
    for i in range(len(genomes)):
        haps.append([0]*len(polyMuts))

    for i in range(len(genomes)):
        for locI, loc, locCont, mutId in polyMuts:
            if mutId in genomes[i]:
                haps[i][locI] = 1

    if return_introgressed:
        return haps, positions, [processedIntrogressedAlleles(u) for u in introgressedAlleles]
    else:
        return haps, positions

def to_unique(X):
    site_hist = dict()
    
    ix = 0
    ii = dict()
    
    indices = []
    for k in range(X.shape[1]):
        x = X[:,k]
        #h = hashFor(x)
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
    print(len(T_present))
    
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

"""

"""
def relate(X, sites, n_samples, mu, r, N, L, diploid = False):

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
                             ofile, odir) + ' >/dev/null 2>&1'
    
    os.system(cmd_)
    
    anc_file = os.path.join(odir, '{}.anc'.format(ofile))
    Fs, Ws, snps, _, coal_times = read_anc(anc_file, pop_sizes = (n_samples, 0))
    
    temp_dir.cleanup()

    return Fs, Ws, snps, np.array(sites), coal_times

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
def load_data(msFile, ancFile = None, n = None, leave_out_last = False):
    msFile = gzip.open(msFile, 'r')

    # no migration case
    if ancFile is not None:
        ancFile = gzip.open(ancFile, 'r')

    ms_lines = [u.decode('utf-8') for u in msFile.readlines()]
    ms_lines = [u for u in ms_lines if not ('#' in u)]

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

