import numpy as np
import matplotlib.pyplot as plt

L = 500
w1 = 0.8
w2 = 2.0
zf = 50

V = 30
n_freq = 128*2**4
w = np.linspace(0.1, 10, n_freq)
n_nodes = 100
n_modes = 2
x = np.linspace(0,500,n_nodes)
delta_x = np.abs(np.tile(x, (n_nodes,1)) - np.tile(x, (n_nodes,1)).transpose())

phi1 = np.array([np.zeros(n_nodes), np.sin(np.pi*x/L), np.zeros(n_nodes)])
phi2 = np.array([np.zeros(n_nodes), np.zeros(n_nodes), np.sin(np.pi*x/L)])

xr = L/2

CL = 0
CL_d = 5
CM = 0
CM_d = 1.5

Iw = 0.08
xLu = 162
xLw = xLu / 12

auto_spec = (1.5 * xLw / V)/(1+2.25*w*xLw/V)**(5/3)
Cwx = 6.5/2/np.pi
co_spec = np.exp(-Cwx*np.einsum('w,ij->wij', w, delta_x/V))

ro = 1.25
B = 20
D = 4
m1 = 10**4
m2 = 6*10**5
zeta1 = 0.005
zeta2 = 0.005

zeta_ae_11 = -39.06E-04 * V + 0j
zeta_ae_12 = 0 + 0j
zeta_ae_21 = -1.563E-04 * V + 0j
zeta_ae_22 = -0.1563E-04 * V**2 + 0j
kapa_ae_11 = 0 + 0j
kapa_ae_12 = 97.66E-04 * V**2 + 0j
kapa_ae_21 = 0 + 0j
kapa_ae_22 = 1.563E-04 * V**2 + 0j

zeta_ae = np.array([[zeta_ae_11, zeta_ae_12],
                    [zeta_ae_21, zeta_ae_22]])

kapa_ae = np.array([[kapa_ae_11, kapa_ae_12],
                    [kapa_ae_21, kapa_ae_22]])

H = np.linalg.inv( np.array([(np.identity(n_modes) - kapa_ae).tolist()]*n_freq) - np.einsum('w,ij->wij', w**2 , np.diag([w1**-2, w2**-2]), dtype='complex128') + \
      np.einsum('w,ij->wij', 2j*w, np.diag([w1**-1, w2**-1]) @ (np.diag([zeta1, zeta2])-zeta_ae ), dtype='complex128'))

H_test = np.zeros((n_freq, 2,2), dtype='complex128')
H1_1D = np.zeros(n_freq, dtype='complex128')  # neglect diagonals
H2_1D = np.zeros(n_freq, dtype='complex128')  # neglect diagonals
for i in range(n_freq):
    H_test[i] = np.linalg.inv(np.identity(n_modes) - kapa_ae - (w[i] * np.diag([1/w1, 1/w2]))**2 + 2j*w[i] *
                 np.diag([w1**-1, w2**-1]) @ (np.diag([zeta1, zeta2])-zeta_ae ))
    H1_1D[i] = (1 - kapa_ae_11 - w[i]**2 * w1**-2 + 2j*w[i] * w1**-1 * zeta1-zeta_ae_11)**-1  # neglect diagonals
    H2_1D[i] = (1 - kapa_ae_22 - w[i]**2 * w2**-2 + 2j*w[i] * w2**-1 * zeta2-zeta_ae_22)**-1  # neglect diagonals


det_H = np.zeros(n_freq, dtype='complex128')
det_H_test = np.zeros(n_freq, dtype='complex128')  # confirming einsum
for f in range(n_freq):
    # sign, logdet = np.linalg.slogdet(H[f])  # a more robust way to obtain determinant, to prevent overflow or underflow
    # det_H[f] = sign * np.exp(logdet)
    # sign, logdet = np.linalg.slogdet(H_test[f])
    # det_H_test[f] = sign * np.exp(logdet)
    # OLD:
    det_H[f] = np.linalg.det(H[f])
    det_H_test[f] = np.linalg.det(H_test[f])


H11 = 1- 0.1**2 * w1**-2 + 2j*0.1* w1**-1 * (zeta1-zeta_ae_11)



plt.plot(w, np.abs(det_H))
plt.plot(w, np.abs(det_H_test), 'rho')
plt.plot(w, np.abs(H1_1D*H2_1D), label='1D * 1D')
plt.grid()
plt.yscale('log')
plt.xscale('log')
# plt.ylim([0.1,100])
plt.legend()
plt.show()

# H1_1D_newtest = np.zeros(n_freq, dtype='complex128')
# H1_1D_newtest_abs = np.zeros(n_freq, dtype='complex128')
# for i in range(n_freq):
#     H1_1D_newtest[i] = np.reciprocal(1 - w[i] ** 2 * w1 ** -2 + 2j * w[i] * w1 ** -1 * zeta1 )
#     H1_1D_newtest_abs[i] = np.abs(H1_1D_newtest[i])
#
# i = 2048
# print((1 - w[i] ** 2 * w1 ** -2 + 2j * w[i] * w1 ** -1 * zeta1 ))
# print(np.abs((1 - w[i] ** 2 * w1 ** -2 + 2j * w[i] * w1 ** -1 * zeta1 ) ** -1))
#
#
# plt.scatter(w, H1_1D_newtest_abs)
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.ylim([0.1,100])
# plt.show()

