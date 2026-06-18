# -*- coding: utf-8 -*-
from popgenml.data.simulators import MSPrimeSimulator
from popgenml.data.transforms import FastSeriate, PadCrop, Flip, Compose
from popgenml.data.datasets import LiveSimulationDataset

import torch
import numpy as np

# Use the included config (.ini) file which defines our prior
simulator = MSPrimeSimulator('recom.ini')
# specify a formatting pipeline for the binary popgen alignment

pipeline = Compose([Flip(), FastSeriate(), PadCrop(128)])

# let's compute the mean and std of our target, log-recombination rate
samples = simulator.r.rvs(size=10000)

log_samples = np.log(samples)

mu_y = np.mean(log_samples)
std_y = np.std(log_samples)

# we'll attempt to predict the log-recombination rate which is returned by our simulator
# as the 'r' entry in the dictionary
def parse_fn(result):
    x = result['x'] # numpy array
    pos = result['pos']
    r = result['r']
    
    r = (np.log(r) - mu_y) / std_y

    # make into torch Tensors, expand a channel dimension s.t. the returned shape is (1, 50, 128)
    return torch.FloatTensor(pipeline(x, pos)).unsqueeze(0), torch.FloatTensor(np.array([r]))
    
dataset = LiveSimulationDataset(simulator, parse_fn)

