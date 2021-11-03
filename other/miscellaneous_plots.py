import numpy as np
import matplotlib.pyplot as plt

def func(x_array):
    yLu = 37.26397209
    result = []
    for x in x_array:
        if x >= 0:
            result.append(np.exp(-x/yLu))
        if x < 0:
            result.append(np.exp(x/yLu))
    return result

x_array = np.linspace(-100,100,201)

plt.figure(figsize=(6,1.5))
plt.plot(x_array, func(x_array))
plt.ylabel(r'$\rho(\Delta s)$')
plt.xlabel(r'$\Delta s$')
plt.tight_layout()