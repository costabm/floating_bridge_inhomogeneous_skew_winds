import numpy as np

C = 1/2 * 1.25*0.4*np.pi/4*20**2
U = 50.38
sigma_u = 0.098*U
var_u = sigma_u**2

mean_F_aprox = C * U**2
mean_F = C * (U**2 + var_u)

var_F_aprox = C**2 * 4*U**2 * var_u
var_F = C**2 * (4*U**2*var_u + 2*var_u)

print(var_F_aprox/1000**2)
print(var_F/1000**2)

sigma_F_aprox = np.sqrt(var_F_aprox)
sigma_F = np.sqrt(var_F)

print(sigma_F_aprox/1000)
print(sigma_F/1000)
