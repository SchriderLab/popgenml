# -*- coding: utf-8 -*-
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
    parser.add_argument("--n_jobs", default = "5000")
    parser.add_argument("--L", default = "1e8")
    
    parser.add_argument("--n_replicates", default = "20")

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
    
    cmd = 'sbatch -t 16:00:00 --mem=24G -n {0} --wrap "mpirun python3 src/simulations/simulate.py --n_replicates {0} --odir {1} --L {2}"'
    
    n_replicates = int(args.n_replicates)
    for ix in range(int(args.n_jobs)):
        ofile = os.path.join(args.odir, '{0:05d}'.format(ix))
        
        cmd_ = cmd.format(n_replicates, ofile, args.L)
        
        print(cmd_)
        os.system(cmd_)

    # ${code_blocks}

if __name__ == '__main__':
    main()



