import numpy as np

U0 = 38  # m/s
ro = 1.25
A1 = np.pi*10**2
A2 = np.pi*7.5**2
CD = 0.4

C = 5.73*10**-2 * (1 + 0.15*U0)**0.5

def U(z):
    return U0*(1+C*np.log(z/10))
U1 = U(z=90)
U2 = U(z=60)

def Iu(z):
    return 0.06*(1+0.043*U0)*(z/10)**(-0.22)
I1 = Iu(z=90)
I2 = Iu(z=60)

u1_std = I1 * U1
u2_std = I2 * U2

F1_mean = 1/2 * ro * A1 * CD * U1**2
F1_std = F1_mean * 2 * u1_std / U1

F2_mean = 1/2 * ro * A2 * CD * U2**2
F2_std = F2_mean * 2 * u2_std / U2

corr = 1

# Monte Carlo 1 (assumes corr(u1,u2) = corr(F1,F2))
covs = [[F1_std**2, F1_std*F2_std*corr],
        [F2_std*F1_std*corr, F2_std**2]]
F = np.random.multivariate_normal([F1_mean, F2_mean], covs, 100000000)
F1 = F[:,0]
F2 = F[:,1]
Fb = F1 + F2
print(np.std(Fb)/1000)


# Monte Carlo 2
covs = [[u1_std**2, u1_std*u2_std*corr],
        [u2_std*u1_std*corr, u2_std**2]]
V = np.random.multivariate_normal([U1, U2], covs, 100000000)
V1 = V[:,0]
V2 = V[:,1]
F1 = F1_mean / U1**2 * V1**2
F2 = F2_mean / U2**2 * V2**2

Fb = F1 + F2
print(np.std(Fb)/1000)


# Analytical
cov_F1F2 = F1_mean*F2_mean*4*corr*I1*I2
Fb_std = np.sqrt(F1_std**2 + F2_std**2 + 2*cov_F1F2)/1000



