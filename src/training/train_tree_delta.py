# -*- coding: utf-8 -*-
import os
import argparse
import logging

import sys
sys.path.append('/home/kilgoretrout/src/popgenml')

from popgenml.models.torchvision_mod_layers import resnet34
from popgenml.data.simulators import StepStoneSimulator
from popgenml.data.data_loaders import MSPrimeFWLoader

import argparse

import torch
import torch.nn.functional as F
import configparser
import torch.nn as nn

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel
from collections import deque

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
import copy
from torchvision import utils

import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None")
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
    
    
    
    

