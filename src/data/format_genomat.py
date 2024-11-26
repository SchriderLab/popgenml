# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py

import glob
import numpy as np

from collections import deque

"""
"""

# temporary patch until package is finished
import sys
sys.path.append('popgenml/data/')

from functions import format_matrix, find_files, load_data

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--mode", default = "class", help = "[class | reg | seg] options for how to format the y variables, classification, regression of parameters, or segmentation of the genotype matrix")
    
    parser.add_argument("--pop_sizes", default = "20,14")
    parser.add_argument("--chunk_size", default = "4")
    parser.add_argument("--out_shape", default = "1,34,512")
    
    parser.add_argument("--classes", default = "None")
    parser.add_argument("--val_prop", default = "0.05")
    parser.add_argument("--sorting", default = "seriate_match")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    # ${odir_del_block}

    return args

import time

def main():
    from mpi4py import MPI
    
    args = parse_args()
    
    # configure MPI
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        ofile = h5py.File(args.ofile, 'w')
        if float(args.val_prop) > 0:
            ofile_val = h5py.File('/'.join(os.path.abspath(args.ofile).split('/')[:-1]) + '/' + os.path.abspath(args.ofile).split('/')[-1].split('.')[0] + '_val.hdf5', 'w')

    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    chunk_size = int(args.chunk_size)
    
    if args.mode == 'class':
        ifiles = []
        
        classes = sorted(os.listdir(args.idir))
        
        for c in classes:
            idir = os.path.join(args.idir, c)
            ifiles.extend([(c, u) for u in find_files(idir)])
            
        if comm.rank == 0:
            logging.info('have {} files to parse...'.format(len(ifiles)))
    
        counts = dict()
        counts_val = dict()
        for c in classes:
            counts[c] = 0
            counts_val[c] = 0
    elif args.mode == 'reg':
        ifiles = []
        classes = sorted(os.listdir(args.idir))
        
        for c in classes:
            idir = os.path.join(args.idir, c)
            ifiles.extend([(None, u) for u in find_files(idir)])
            
        count = 0
        count_val = 0
            
        if comm.rank == 0:
            logging.info('have {} files to parse...'.format(len(ifiles)))
    elif args.mode == 'seg':
        ifiles = []
    
        ms_files = sorted(glob.glob(os.path.join(args.idir, '*/*.msOut.gz')))
        anc_files = sorted(glob.glob(os.path.join(args.idir, '*/*.anc.gz')))
        
        print(ms_files, anc_files)
        
        ifiles.extend([(None, u) for u in list(zip(ms_files, anc_files))])
        
        count = 0
        count_val = 0
            
        if comm.rank == 0:
            logging.info('have {} files to parse...'.format(len(ifiles)))
    
    if comm.rank != 0:
        for ij in range(comm.rank - 1, len(ifiles), comm.size - 1):
            tag, ifile = ifiles[ij]
            logging.info('{}: working on {}...'.format(comm.rank, ifile))
            
            t0 = time.time()
            
            try:
                if type(ifile) == tuple:
                    ifile, anc_file = ifile
                
                    X, Y, P, params = load_data(ifile, anc_file)
                else:
                    X, Y, P, params = load_data(ifile)
            except:
                logging.info('could not read {}!'.format(ifile))
                comm.send([None, None, None], dest = 0)
                continue
            
            X_ = []
            P_ = []
            params_ = []
            ls = []
            
            for ix, x in enumerate(X):
                if x is None:
                    continue
                
                try:
                    ls.append(x.shape[1])
                except:
                    pass
                
                if args.mode != 'seg':
                    x, p, y = format_matrix(x, P[ix], pop_sizes, out_shape = tuple(map(int, args.out_shape.split(','))), mode = args.sorting)
                else:
                    x, p, y = format_matrix(x, P[ix], Y[ix], pop_sizes, out_shape = tuple(map(int, args.out_shape.split(','))), mode = args.sorting)
                    
                if x is not None:
                    
                    X_.append(x)
                    P_.append(p)
                    
                    if y is None:
                        params_.append(params[ix])
                    else:
                        params_.append(y)
            
            logging.info('sending {} matrices from {}...'.format(len(X_), ifile))    
            print('time: {}'.format(time.time() - t0))
            
            
            if args.mode == 'class':
                comm.send([X_, P_, tag], dest = 0)
            elif args.mode in ('reg', 'seg'):
                comm.send([X_, P_, params_], dest = 0)
            
            
            if len(ls) > 0:
                logging.info('have max shape: {}'.format(max(ls)))
    else:
        n_received = 0
                
        while n_received < len(ifiles):
            if args.mode == 'class':
                Xf, p, tag = comm.recv(source = MPI.ANY_SOURCE)
            else:
                Xf, p, y = comm.recv(source = MPI.ANY_SOURCE)
                
            if Xf is None:
                n_received += 1
                continue
            
            logging.info('have len {}'.format(len(Xf)))
            while len(Xf) >= chunk_size:
                if np.random.uniform() < float(args.val_prop):
                    if args.mode == 'class':
                        ofile_val.create_dataset('{}/{}/x'.format(tag, counts_val[tag]), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile_val.create_dataset('{}/{}/p'.format(tag, counts_val[tag]), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        counts_val[tag] += 1
                    else:
                        ofile_val.create_dataset('{}/x'.format(count_val), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile_val.create_dataset('{}/p'.format(count_val), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        ofile_val.create_dataset('{}/y'.format(count_val), data = np.array(y[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        count_val += 1
                          
                    ofile_val.flush()
                    
                else:
                    if args.mode == 'class':
                        ofile.create_dataset('{}/{}/x'.format(tag, counts[tag]), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile.create_dataset('{}/{}/p'.format(tag, counts[tag]), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        counts[tag] += 1
                    else:
                        ofile.create_dataset('{}/x'.format(count), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile.create_dataset('{}/p'.format(count), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        ofile.create_dataset('{}/y'.format(count), data = np.array(y[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        count += 1    
                    
                    ofile.flush()
            
                del Xf[-chunk_size:]
                del p[-chunk_size:]
                
                if not args.mode == 'class':
                    del y[-chunk_size:]
            
            n_received += 1
            if n_received % 10 == 0:
                logging.info('received {} files thus far...'.format(n_received))
                     
        if len(Xf) > 0:
            if args.mode == 'class':
                ofile.create_dataset('{}/{}/x'.format(tag, counts[tag]), data = np.array(Xf, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{}/{}/p'.format(tag, counts[tag]), data = np.array(p, dtype = np.float32), compression = 'lzf')
                
                counts[tag] += 1
            else:
                ofile.create_dataset('{}/x'.format(count), data = np.array(Xf, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{}/p'.format(count), data = np.array(p, dtype = np.float32), compression = 'lzf')
                ofile.create_dataset('{}/y'.format(count), data = np.array(y, dtype = np.float32), compression = 'lzf')
                
                count += 1    
        
        ofile.flush()
                
    if comm.rank == 0:
        ofile.close()
        if float(args.val_prop) > 0:
            ofile_val.close()
            
    # ${code_blocks}

if __name__ == '__main__':
    main()
