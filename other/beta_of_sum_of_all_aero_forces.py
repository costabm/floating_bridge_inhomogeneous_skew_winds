"""Estimates the beta (and theta) of the resultant single aerodynamic force vector, which is a SRSS of the 3 aerodynamic forces Fx, Fy, Fz"""


import pandas as pd
import numpy as np
from aero_coefficients import aero_coef, rad, deg
import matplotlib.pyplot as plt
import os
import copy

#####################################################################################################################
# RAW DATA FROM SOH
#####################################################################################################################
# Importing input file
path_df = os.path.join(os.getcwd(), r'aerodynamic_coefficients', 'aero_coef_experimental_data.csv')
df = pd.read_csv(path_df)  # raw original values

# Importing the angles
betas_uncorrected_SOH = rad(df['SOH_beta_uncorrected[deg]'].to_numpy()) # SOH initial skew angle, before performing rotation about bridge axis (which changes the beta angle).
alphas_SOH = rad(df['alpha[deg]'].to_numpy()) # Alpha: rotation about the bridge x-axis, which differs from the theta definition from L.D.Zhu and from SOH alpha (opposite direction).
betas_SOH = rad(df['beta[deg]'].to_numpy())
thetas_SOH = rad(df['theta[deg]'].to_numpy())

# # Importing the coefficients in Local normal wind coordinates "Lnw":
# Cx_Lnw  = df['Cx_Lnw']  # same as SOH Drag
# Cy_Lnw  = df['Cy_Lnw']  # axial direction (opposite to Ls_x, so positive with positive beta)
# Cz_Lnw  = df['Cz_Lnw']  # same as SOH Lift
# Cxx_Lnw = df['Cxx_Lnw']
# Cyy_Lnw = df['Cyy_Lnw']  # same as SOH moment
# Czz_Lnw = df['Czz_Lnw']

# Importing the coefficients in Local structural coordinates "Ls":
Cx_Ls  = df['Cx_Ls']
Cy_Ls  = df['Cy_Ls']
Cz_Ls  = df['Cz_Ls']
Cxx_Ls = df['Cxx_Ls']
Cyy_Ls = df['Cyy_Ls']
Czz_Ls = df['Czz_Ls']
#####################################################################################################################


betas = np.arange(rad(0), rad(91), rad(1))
thetas = np.ones(betas.shape)*rad(0)

Ci_Ls_cons = aero_coef(betas, thetas, method='2D_fit_cons', coor_system='Ls')
Ci_Ls_cons_2 = aero_coef(betas, thetas, method='2D_fit_cons_2', coor_system='Ls')
Ci_Ls_free = aero_coef(betas, thetas, method='2D_fit_free', coor_system='Ls')
Ci_Ls_cosr = aero_coef(betas, thetas, method='cos_rule', coor_system='Ls')

# Fitted aero coef in Ls system
Cx_cons, Cy_cons, Cz_cons = Ci_Ls_cons[:3]
Cx_cons_2, Cy_cons_2, Cz_cons_2 = Ci_Ls_cons_2[:3]
Cx_free, Cy_free, Cz_free = Ci_Ls_free[:3]
Cx_cosr, Cy_cosr, Cz_cosr = Ci_Ls_cosr[:3]

# SRSS of Cx and Cy
Cxy_cons = np.sqrt(Cx_cons**2 + Cy_cons**2)
Cxy_cons_2 = np.sqrt(Cx_cons_2**2 + Cy_cons_2**2)
Cxy_free = np.sqrt(Cx_free**2 + Cy_free**2)
Cxy_cosr = np.sqrt(Cx_cosr**2 + Cy_cosr**2)
Cxy_test = np.sqrt(Cx_Ls**2 + Cy_Ls**2)

# Resultant betas
beta_resultant_cons = -np.arccos(Cy_cons/Cxy_cons)*np.sign(Cx_cons)
beta_resultant_cons_2 = -np.arccos(Cy_cons_2/Cxy_cons_2)*np.sign(Cx_cons_2)
beta_resultant_free = -np.arccos(Cy_free/Cxy_free)*np.sign(Cx_free)
beta_resultant_cosr = -np.arccos(Cy_cosr/Cxy_cosr)*np.sign(Cx_cosr)
beta_resultant_test = -np.arccos(Cy_Ls/Cxy_test)*np.sign(Cx_Ls)

Cy_Cx_factor = np.max(abs(Cx_cons)) / np.max(abs(Cy_Ls[2::5]))  # Cx / Cy factor. To create a Cos+Sin Rule
beta_resultant_cos_sin = np.arctan(Cy_Cx_factor * np.sin(betas)**2 / (np.cos(betas)**2))

plt.figure(figsize=(4.4,4.4), dpi=200)
ax = plt.axes()
plt.title(r'$\beta_{\vec F}\/(\beta, \theta=0\degree)$')
ax.scatter(deg(betas_SOH)[2::5], deg(beta_resultant_test)[2::5], label=r'Measur. '+r'$(\theta=0)$', color='black', alpha=0.8, s=30)
ax.scatter(deg(betas_SOH), deg(beta_resultant_test), label=r'Measur. '+r'$(\theta\neq0)$', color='black', alpha=0.2, s=20, marker='x')
ax.plot(deg(beta_resultant_free), ls='-.', label=r'Free fit', linewidth=2, alpha=0.8, color='green')
ax.plot(deg(beta_resultant_cons), label=r'Constr. fit', linewidth=2, alpha=0.8, color='brown')
# ax.plot(deg(beta_resultant_cons_2), label=r'Constr. fitting 2', linewidth=2.2, alpha=0.6, color='brown')
ax.plot(deg(beta_resultant_cosr), label='2D approach', linewidth=2., alpha=0.8, linestyle='--', color='gold')
ax.plot(deg(beta_resultant_cos_sin), label='2D+1D approach', linewidth=2., alpha=0.8, linestyle=':', color='blue')
ax.plot(deg(betas),ls=(0, (3, 1.5, 1, 1.5, 1, 1.5)), label=r'$\beta_{\vec F}=\beta$', lw= 2,  alpha=0.6, color='grey')
plt.xlim([-2,92])
plt.ylim([-2,92])
plt.xticks(np.arange(0,91,15))
plt.yticks(np.arange(0,91,15))
plt.xlabel(r'$\beta\/[\degree]$')
plt.ylabel(r'$\beta_{\vec F}\/[\degree] $')
handles, labels = ax.get_legend_handles_labels()
handles = [handles[i] for i in [5,6,2,3,0,1,4]]
labels = [labels[i] for i in [5,6,2,3,0,1,4]]
plt.legend(handles,labels)
plt.tight_layout()
plt.savefig(r'other/beta_resultant_force.png')
plt.close()

# deg(thetas_SOH)
# plt.figure()
# plt.plot(deg(betas_SOH)[2::5], Cx_Ls[2::5])
# plt.plot(deg(betas_SOH)[2::5], Cy_Ls[2::5])
