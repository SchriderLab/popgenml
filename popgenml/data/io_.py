# -*- coding: utf-8 -*-
import gzip
import numpy as np
import os

"""
need to test this...
"""
def find_files(idir, exts = ('.msOut.gz')):
    matches = []
    
    if not os.path.isdir(idir):
        return matches
        
    for root, dirnames, filenames in os.walk(idir):
        filenames = [ f for f in filenames if os.path.splitext(f)[1] in exts ]
        for filename in filenames:
            matches.append(os.path.join(root, filename))
            
    return matches

"""
Appends a simulation (genotype matrix X, (n_samples, n_sites) and positions (0 to 1)) to a text file 
in a way identical to how ms outputs simulations.
"""
def append_to_ms(ofile, X, sites, params = None):
    if params:
        header = '// ' + ' '.join(['{0:04f}'.format(u) for u in params]) + '\n'
    else:
        header = '//\n'
        
    ofile.write(header)
    n_segsites = X.shape[1]
    ofile.write('segsites: {}\n'.format(n_segsites))
    pos_line = 'positions: ' + ' '.join(['{0:08f}'.format(u) for u in sites]) + '\n'
    ofile.write(pos_line)
    
    for x in X:
        line = ''.join(list(map(str, list(x)))) + '\n'
        ofile.write(line)
        
    ofile.write('\n')

"""
Writes genotype matrix X, (n_samples, n_sites) and positions to a text file in a way identical to how ms
outputs simulations.
"""
def write_to_ms(ofile, X, sites, params = None):
    ofile = open(ofile, 'w')
    
    if params:
        header = '// ' + ' '.join(['{0:04f}'.format(u) for u in params]) + '\n'
    else:
        header = '//\n'
        
    ofile.write(header)
    n_segsites = X.shape[1]
    ofile.write('segsites: {}\n'.format(n_segsites))
    pos_line = 'positions: ' + ' '.join(['{0:08f}'.format(u) for u in sites]) + '\n'
    ofile.write(pos_line)
    
    for x in X:
        line = ''.join(list(map(str, list(x)))) + '\n'
        ofile.write(line)
        
    ofile.write('\n')
    ofile.close()
    
def split(word):
    return [char for char in word]
    
######
# generic function for msmodified
# ----------------
# takes a gzipped ms file
# returns a list of genotype matrices, introgressed allele matrices (if *.anc file is provided),
# a list of position vectors, and a list of the parameters listed in the \\ line if any
def load_ms(msFile, ancFile = None, n = None, flip_alleles = True):
    msFile = gzip.open(msFile, 'r')

    # no migration case
    if ancFile is not None:
        ancFile = gzip.open(ancFile, 'r')

    ms_lines = [u.decode('utf-8') for u in msFile.readlines()]
    ms_lines = [u for u in ms_lines if (not '#' in u)]

    idx_list = [idx for idx, value in enumerate(ms_lines) if ('//' in value)] + [len(ms_lines)]
        
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
        
        if flip_alleles:
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
            y = None
            
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

import sys

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