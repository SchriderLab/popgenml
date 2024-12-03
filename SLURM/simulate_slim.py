# -*- coding: utf-8 -*-
import os
import argparse
import logging

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
    
    parser.add_argument("--args", default = '"-d sampleSizePerSubpop={} -d donorPop={} -d st={} -d mt={} -d mp={} -d introS={}"')
    parser.add_argument("--vals", default = "32,1,4,0.25,1,0")
    
    parser.add_argument("--n_jobs", default = 100, type = int)

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
    
    cmd = "python3 src/simulations/simulate_slim.py --ifile {} --L {} --n_replicates {} --args {} --vals {} ".format(args.ifile, args.L, args.n_replicates, args.args, args.vals)

    for ix in range(args.n_jobs):
        odir = os.path.join(args.odir, "iter{0:05d}".format(ix))
        
        cmd_ = cmd + "--odir {}".format(odir)
        
        print(cmd_)
        os.system(cmd_)

if __name__ == '__main__':
    main()
