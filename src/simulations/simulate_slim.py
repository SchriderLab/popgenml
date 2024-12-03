# -*- coding: utf-8 -*-
import os
import argparse
import logging

import sys
sys.path.append('popgenml/data')

from simulators import SlimSimulator
from io_ import append_to_ms
import numpy as np

def array_to_binary_string(arr):
    binary_string = ''.join(str(num) for num in arr)
    return binary_string + '\n'

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "src/simulations/slim/introg_bidirectional.slim")
    parser.add_argument("--L", default = 10000, type = int)
    parser.add_argument("--n_replicates", default = 10, type = int)
    
    parser.add_argument("--args", default = "-d sampleSizePerSubpop={} -d donorPop={} -d st={} -d mt={} -d mp={} -d introS={}")
    parser.add_argument("--vals", default = "64,1,4,0.25,1,0")

    parser.add_argument("--odir", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()
    
    args_ = args.args
    if args_ == "None":
        args_ = None
    
    sim = SlimSimulator(args.ifile, args_)
    
    ms_ofile = open(os.path.join(args.odir, 'mig.msOut'), 'w')
    anc_ofile = open(os.path.join(args.odir, 'out.anc'), 'w')
    
    for ij in range(args.n_replicates):
        X, pos, y = sim.simulate(*args.vals.split(','))
        
        y = y.T.astype(np.uint8)
        
        append_to_ms(ms_ofile, X, pos)
        for y_ in y:
            line = array_to_binary_string(y_)
            anc_ofile.write(line)
        
    anc_ofile.close()
    ms_ofile.close()
    
    os.system('gzip {} && gzip {}'.format(os.path.join(args.odir, 'mig.msOut'), os.path.join(args.odir, 'out.anc')))
    
    # ${code_blocks}

if __name__ == '__main__':
    main()


