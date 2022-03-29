# This scripts shows it is possible to have an asymetric bridge dynamic response, in the frequency-domain even if the bridge is symetric.
# This is due to the off-diagonal terms of the cross spectral density matrices of modal forces and modal responses.
# This is useful to conclude on the vertical buffeting response under inhomogeneous winds

import numpy as np
import matplotlib.pyplot as plt

stop = False
while not stop:
    N = 20
    x = np.linspace(0, 2*np.pi, N)
    m1 = np.cos(0.5*x) - np.mean(np.cos(0.5*x))
    m2 = np.cos(x) - np.mean(np.cos(x))
    m = np.array([m1,m2])
    asym_weight = np.einsum('i,j->ij', np.linspace(0,1,N), np.linspace(0,1,N))
    F = np.random.rand(N,N) * asym_weight
    F_sym = F + F.T
    S_etaeta = m @ F_sym @ m.T
    if all(np.diag(S_etaeta) > 0):
        S_deltadelta = m.T @ S_etaeta @ m
        plt.plot(m[0], label='mode 1')
        plt.plot(m[1], label='mode 2')
        plt.legend()
        plt.show()
        plt.plot(np.diag(S_deltadelta), label='S_delta (diagonals only)')
        plt.legend()
        plt.show()
        stop = True

