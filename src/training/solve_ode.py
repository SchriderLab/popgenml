# -*- coding: utf-8 -*-
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

n_samples = 64

nc2 = np.array([n * (n - 1) / 2 for n in range(2, n_samples + 1)])[::-1]
n = np.array([n for n in range(1, n_samples + 1)])[::-1].copy()
Q = np.zeros((n_samples, n_samples))

ii = list(range(nc2.shape[0]))
ij = [u + 1 for u in ii]

Q[ii, ij] = nc2
Q[range(Q.shape[0]), range(Q.shape[0])] = -Q.sum(-1)

fractions = [1e-4, 1e-3, 1e-2, 1e-1]

N0 = 100
Nm = 0.1 * N0

def dy(y, t):
    return (Q * ((N0 + Nm * t) ** -1)).dot(y.reshape(n_samples, n_samples)).flatten()

def dy0(y, t):
    return (Q * (N0 ** -1)).dot(y.reshape(n_samples, n_samples)).flatten()


# values of time 
t = np.linspace(0,5,100) 

y0 = np.eye(n_samples)

# solving ODE 
y = odeint(dy, y0.flatten(), t)
y1 = odeint(dy0, y0.flatten(), t)

fig, axes = plt.subplots(ncols = 2)
axes[0].imshow(y[:,:n_samples].reshape(len(t), n_samples))
axes[1].imshow(y1[:,:n_samples].reshape(len(t), n_samples))

plt.show()