# -*- coding: utf-8 -*-
from popgenml.data.simulators import MSPrimeSimulator
from popgenml.data.msmc2 import msmc2

simulator = MSPrimeSimulator('spline_neutral_n4_10Mb.ini')
ret = simulator.simulate()

x = ret['x']
pos = ret['pos']

msmc2(x, pos, simulator.L)

