# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:26:42 2018

@author: bernardc
"""
from scipy import signal
import numpy as np

# According to Etienne Cheynet:
# https://se.mathworks.com/matlabcentral/fileexchange/50041-wind-field-simulation-the-user-friendly-version

def coherence(X, Y, fs, window='hann', nperseg=256):
    
    # Remove linear trend
    X = signal.detrend(X)
    Y = signal.detrend(Y)
    
    # Cross power spectral density
    freq, pxy = signal.csd(X,Y, fs, window=window, nperseg=nperseg)
    
    # 1-point power spectral density
    pxx = signal.csd(X,X, fs, window=window, nperseg=nperseg)[1]
    pyy = signal.csd(Y,Y, fs, window=window, nperseg=nperseg)[1]

    # Normalize co-spectrum
    cocoh = np.real(pxy/np.sqrt(pxx * pyy)) # co-coherence
    quad = np.imag(pxy/np.sqrt(pxx * pyy)) # quad-coherence
    
    return{'freq':freq,
           'cocoh':cocoh,
           'quad':quad
           }
