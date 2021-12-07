# -*- coding: utf-8 -*-
"""
created: 2019
author: Bernardo Costa
email: bernamdc@gmail.com

This script is based on the following paper:
'Buffeting response of long-span cable-supported bridges under skew winds. Part 1: theory - L.D. Zhu, Y.L. Xu'
which is part of the book "Wind effects on cable-supported bridges - You-Lin Xu" and the PhD thesis from L.D. Zhu.

Notes:
1) Transformation matrices and rotation matrices are not the same. They are the inverse (or the transpose! or rotating
the opposite angle!) of each other.
One rotates the axes, the other rotates the vectors/points. A rotation matrix R =
[[cos(theta) -sin(theta)]
 [sin(theta)  cos(theta)]]
rotates a vector in xy-plane counterclockwise. These are known as active rotations of vectors counterclockwise in a
right-handed coordinate system (y counterclockwise from x) by pre-multiplication (R on the left) v'=Rv.
If any one of these is changed (such as rotating axes instead of vectors, a passive transformation),
then the inverse of the example matrix should be used, which coincides with its transpose.
The transformation T in v'=Tv == rotating the axes to which vector v will now (as v') be related to, by doing: v'=Tv,
which is the same as rotating v in the original axes, the opposite way (try to visualize this!): v'=inverse(R)v.

2) Understanding np.einsum:
np.einsum('ab,bc,cd->ad', M1, M2, M1) <=> M1 @ M2 @ M1
np.einsum('ab,bc,dc->ad', M1, M2, M1) <=> M1 @ M2 @ np.transpose(M1)

...and, the notation used in the np.einsum subscripts to improve readability is:
w - omegas, angular frequencies
t - time
m,n - g_nodes
M,N - Modes
d - dof
c - wind components
i,j,v - other...

How to transform M: (transp(T) @ M) or (transp(T) @ M @ T)? Answer: if M is just a collection of vectors (each one of
them having xyz information) then is the first case. If M is a matrix, where the diagonal has the xx, yy, zz information
and off-diagonals mean xy, yz, etc. then is the second case.

The bridge axis is oriented from 190deg (SSW) to 10deg (NNE) (Design Basis convention).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy
import time
import pandas as pd
from simple_5km_bridge_geometry import arc_length
from aero_coefficients import aero_coef, aero_coef_derivatives
from transformations import normalize, g_node_L_3D_func, g_elem_nodes_func, T_GsGw_func, T_LsGs_3g_func, T_LsGs_6g_func,\
    T_LrLs_func, T_LSOHLwbar_func, mat_Ls_node_Gs_node_all_func, rotate_v1_about_v2_func, vec_Ls_elem_Ls_node_girder_func, \
    T_LsLw_func, T_LsGs_full_2D_node_matrix_func, T_LsGs_6p_func, beta_within_minus_Pi_and_Pi_func, T_LnwLs_func, T_LwLnw_func,\
    M_3x3_to_M_6x6, T_LwGw_func, T_LsGw_func, T_GwLs_derivatives_func, T_LwLs_derivatives_func, theta_yz_bar_func, T_LnwLs_dtheta_yz_func,\
    C_Ci_Ls_to_C_Ci_Lnw, discretize_S_delta_local_by_equal_energies, T_GsNw_func, T_xyzXYZ_ndarray
from mass_and_stiffness_matrix import mass_matrix_func, stiff_matrix_func, geom_stiff_matrix_func
from modal_analysis import modal_analysis_func, simplified_modal_analysis_func
from damping_matrix import rayleigh_coefficients_func, rayleigh_damping_matrix_func, added_damping_global_matrix_func
from profiling import profile
import os

########################################################################################################################
# Global variables
########################################################################################################################
CS_width = 30  # m. CS width.
CS_height = 4  # m. CS height
rho = 1.25  # kg/m3. air density.
RP = 100  # years Return Period.
x_tower = 325  # m. x-coordinate of South tower for turbulence considerations.
theta_0 = 0  # it is 0 if wind is in the Global XY plane. theta will account for girder geometries (and static loads).
# Damping
damping_ratio = 0.005  # Structural damping
damping_Ti = 120  # period matching exactly the damping ratio (see Rayleigh damping)
damping_Tj = 2  # period matching exactly the damping ratio (see Rayleigh damping)  # this used to be 5 sec, but see AMC\Milestone 10\Appendix F - Enclosure 1, Designers format, K11-K14.zip

########################################################################################################################
# Auxiliary generic functions
########################################################################################################################
def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

def delta_array_func(array):
    """Gives array with consecutive differences between its element values. Half distance for first and last elements"""
    n_array = len(array)
    delta_array = np.zeros(n_array)
    delta_array[0] = (array[1]-array[0])/2
    delta_array[-1] = (array[-1]-array[-2])/2
    delta_array[1:-1] = np.array([array[f+1]-array[f] for f in range(1, n_array-1)])
    return delta_array

########################################################################################################################
# Functions dependant on wind direction and(or) node coordinates
########################################################################################################################
def beta_0_func(beta_DB):
    beta_0 = rad(100) - beta_DB  # [rad]. Global XYZ mean yaw angle, where both bridge ends fall on X-axis. Convention used as in Fig.1 (of the mentioned paper). beta_DB = 100 <=> beta_0 = 0. beta_DB = 80 <=> beta_0 = 20 [deg].
    beta_0 = np.where(beta_0<=rad(-180), rad(180) - (rad(-180) - beta_0), beta_0)  # converting to interval [rad(-180),rad(180)]. Confirm with: print('beta_DB = ', round(deg(beta_DB)), ' beta_0 = ', round(deg(beta_0)))
    return beta_0


def U_bar_func(g_node_coor, RP=RP):
    """ 10min mean wind """  #
    g_node_coor_z = g_node_coor[:, 2]  # m. Meters above sea level
    z = g_node_coor_z
    # Possible distributions of wind speed:
    # 1) Constant if same z. 2) Linearly varying from 0.6*V to V, from one end to the other. # NOTE: case 2 not done.

    # # 10min mean wind speed, at 10m height:
    # U10_1 = 22.9  # m/s at 10m height, for 1 year return period
    # U10_10 = 27.6
    # U10_100 = 31.7
    # U10_10000 = 38.4  # m/s at 10m height, for 10000 year return period
    #
    # z0 = 0.01  # roughness length
    # kt = 0.17  # terrain factor.
    #
    # # Wind speed, for each point and for each Return Period. Design Basis chapter 2.1.3.
    # U_1 = 1.07 * U10_1 * kt * np.log(nodes_coor_z / z0)
    # U_10 = 1.07 * U10_10 * kt * np.log(nodes_coor_z / z0)
    # U_100 = 1.07 * U10_100 * kt * np.log(nodes_coor_z / z0)
    # U_10000 = 1.07 * U10_10000 * kt * np.log(nodes_coor_z / z0)
    #
    # if (RP == 1):
    #     U = U_1
    # elif (RP == 10):
    #     U = U_10
    # elif (RP == 100):
    #     U = U_100
    # elif (RP == 10000):
    #     U = U_10000
    # return U

    # Newest version:
    # Alternative
    z0 = 0.01
    kt = 0.17
    K = 0.2
    n = 0.5
    p = 1 - np.exp(-1/RP)
    Cr = kt * np.log(z/z0)
    Cprob = ((1-K*np.log(-np.log(1-p)))/(1-K*np.log(-np.log(0.98))))**n
    V_1hour = Cr * Cprob * 24.3
    V_10min = V_1hour * 1.07
    return V_10min

def wind_vector_func(beta_0, theta_0):
    # Normalized. In Global Coordinates.
    return normalize(np.array([-np.sin(beta_0)*np.cos(theta_0), np.cos(beta_0)*np.cos(theta_0), np.sin(theta_0)]))

def beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha):
    """Returns the beta_bar and theta_bar at each node, as a mean of adjacent elements.
    Note that the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
    n_g_nodes = len(g_node_coor)
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
    T_GsGw = T_GsGw_func(beta_0, theta_0)
    T_LsGw = np.einsum('nij,jk->nik', T_LsGs, T_GsGw)
    U_Gw = np.array([np.ones(n_g_nodes), np.zeros(n_g_nodes), np.zeros(n_g_nodes)]).T
    U_Ls = np.einsum('nij,nj->ni', T_LsGw, U_Gw)
    Ux = U_Ls[:,0]
    Uy = U_Ls[:,1]
    Uz = U_Ls[:,2]
    Uxy = np.sqrt(Ux**2 + Uy**2)
    beta_bar = np.array([-np.arccos(Uy[i]/Uxy[i]) * np.sign(Ux[i]) for i in range(n_g_nodes)])
    theta_bar = np.array([np.arcsin(Uz[i]/ 1) for i in range(n_g_nodes)])
    return beta_bar, theta_bar

########################################################################################################################
# Wind properties
########################################################################################################################
def Ai_func(cond_rand_A):
    # Shall Au,Av,Aw parameters be varied +-40%?:
    if cond_rand_A:
        Au = random.uniform(6.8 - 0.4*6.8, 6.8 + 0.4*6.8)
        Av = random.uniform(9.4 - 0.4*9.4, 9.4 + 0.4*9.4)
        Aw = random.uniform(9.4 - 0.4*9.4, 9.4 + 0.4*9.4)
    else:
        Au = 6.8
        Av = 9.4
        Aw = 9.4
    Ai = np.array([Au, Av, Aw])
    return Ai

def Cij_func(cond_rand_C):
    # Shall Cij parameters be varied +-50%?:
    if cond_rand_C:
        Cux = random.uniform(3 - 0.5*3, 3 + 0.5*3)  # Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
        Cvx = random.uniform(3 - 0.5*3, 3 + 0.5*3)  # Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
        Cwx = random.uniform(3 - 0.5*3, 3 + 0.5*3)  # Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
        Cuy = random.uniform(10 - 0.5*10, 10 + 0.5*10)
        Cuz = Cuy
        Cvy = random.uniform(6.5 - 0.5*6.5, 6.5 + 0.5*6.5)
        Cvz = Cvy
        Cwy = Cvy
        Cwz = random.uniform(3 - 0.5*3, 3 + 0.5*3)
    else:
        Cux = 3.  # Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
        Cvx = 6.  # Taken from AMC Aerodynamics report, Table 4. Changed recently!
        Cwx = 3.  # Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
        Cuy = 10.
        Cvy = 6.5
        Cwy = 6.5
        Cuz = Cuy
        Cvz = Cvy
        Cwz = 3.
    Cij = np.array([[Cux, Cuy, Cuz],
                    [Cvx, Cvy, Cvz],
                    [Cwx, Cwy, Cwz]])
    return Cij

def iLj_func(g_node_coor):
    g_node_num = len(g_node_coor)
    g_node_coor_z = g_node_coor[:, 2]  # m. Meters above sea level
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0

    # Defining Integral Length Scales. Notation from N400 used.
    L1 = 100  # (m) reference length scale.
    z1 = 10  # (m) reference height.
    zmin = 1  # (m) for Terrain Cat. 0 and I.
    xLu = np.zeros(g_node_num)
    for n in g_nodes:
        xLu[n] = L1 * ( max(g_node_coor_z[n], zmin) / z1) ** 0.3
    yLu = 1 / 3 * xLu
    zLu = 1 / 5 * xLu
    xLv = 1 / 4 * xLu
    yLv = 1 / 4 * xLu
    zLv = 1 / 12 * xLu
    xLw = 1 / 12 * xLu
    yLw = 1 / 18 * xLu
    zLw = 1 / 18 * xLu
    return np.moveaxis(np.array([[xLu, yLu, zLu],
                                 [xLv, yLv, zLv],
                                 [xLw, yLw, zLw]]), -1, 0)

def reduc_coef_sector_func(beta_DB):
    # Sectors' directional reduction coefficient. 0/360 deg from North. 90 deg from East. NOTE: not being used!
    coef_sector_1 = 0.7  # 0-75 deg
    coef_sector_2 = 0.85 # 75-225 deg
    coef_sector_3 = 0.9  # 225-255 deg
    coef_sector_4 = 1    # 255-285 deg
    coef_sector_5 = 0.9  # 285-345 deg
    coef_sector_6 = 0.7  # 345-360 deg
    if 0 <= beta_DB <= 75:
        coef_sector = coef_sector_1
    if 75 < beta_DB <= 225:
        coef_sector = coef_sector_2
    if 225 < beta_DB <= 255:
        coef_sector = coef_sector_3
    if 255 < beta_DB < 285:
        coef_sector = coef_sector_4
    if 285 <= beta_DB < 345:
        coef_sector = coef_sector_5
    if 345 <= beta_DB < 360:
        coef_sector = coef_sector_6
    return coef_sector

def Ii_func(g_node_coor, beta_DB, Ii_simplified):
    # Turbulence. Accounts for different turbulence at different z & x, according to the Design Basis, PDF p.63 Table 11
    # Two possible Sectors. First from 150 to 210 deg:
    g_node_num = len(g_node_coor)
    g_node_coor_z = g_node_coor[:, 2]  # m. Meters above sea level
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0

    Iu = np.zeros(g_node_num)
    if Ii_simplified:
        # OLD VERSION. CHANGED ONLY ON (15/04/2021)
        # Iu = np.zeros(g_node_num)
        # for n in g_nodes:
        #     Iu[n] = 0.14
        # Iv = 0.85 * Iu
        # Iw = 0.55 * Iu
        Iu = np.zeros(g_node_num)
        for n in g_nodes:
            Iu[n] = 1 / np.log(g_node_coor_z[n]/0.01)  # Design basis rev 0, 2018, Chapter 2.2
        Iv = 0.84 * Iu  # Design basis rev 0, 2018, Chapter 2.2
        Iw = 0.60 * Iu  # Design basis rev 0, 2018, Chapter 2.2
    #
    # else:
    #     if 150 <= beta_DB <= 210:
    #         for n in g_nodes:
    #             if 0 <= g_node_coor[n, 0] <= x_tower:  # if point is South of Tower:
    #                 if 0 <= g_node_coor[n, 2] <= 50:  # if point is below z=50m, Iu=0.30
    #                     Iu[n] = 0.3
    #                 elif 50 < g_node_coor[n, 2] < 200:  # if point is above z=50m, interpolation:
    #                     Iu[n] = (g_node_coor[n, 2] - 50) * (0.15 - 0.3) / (200 - 50) + 0.30
    #                 elif 200 <= g_node_coor[n, 2]:  # if point is above z=200m, Iu=0.15
    #                     Iu[n] = 0.15
    #                 else:
    #                     print('z must be positive')
    #             elif x_tower < g_node_coor[n, 0] <= arc_length:  # if point is North of Tower:
    #                 if 0 <= g_node_coor[n, 2] <= 50:  # if point is below z=50m, Iu decreases linearly from 0.3 to 0.17
    #                     Iu[n] = 0.3 - (g_node_coor[n, 0] - x_tower) / (arc_length - x_tower) * (0.30 - 0.17)
    #                 elif 50 < g_node_coor[n, 2] < 200:  # if point is above z=50m, interpolation:
    #                     Iu[n] = (g_node_coor[n, 2] - 50) * (
    #                                 0.15 - (0.3 - (g_node_coor[n, 0] - x_tower) / (arc_length - x_tower) * (0.30 - 0.17))) / (200 - 50) + (
    #                                     0.3 - (g_node_coor[n, 0] - x_tower) / (arc_length - x_tower) * (0.30 - 0.17))
    #                 elif 200 <= g_node_coor[n, 2]:  # if point is above z=200m, Iu=0.15
    #                     Iu[n] = 0.15
    #                 else:
    #                     print('z must be positive')
    #             else:
    #                 print('x must be from 0m to the bridge girder length')
    #         Iv = 0.85 * Iu
    #         Iw = 0.55 * Iu
    #     # Then from from 0-150deg and 210-360deg:
    #     else:
    #         for n in g_nodes:
    #             if 0 <= g_node_coor[n, 2] <= 50:  # if point is below z=50m, Iu=0.14
    #                 Iu[n] = 0.14
    #             elif 50 < g_node_coor[n, 2] < 200:  # if point is above z=50m, interpolation:
    #                 Iu[n] = (g_node_coor[n, 2] - 50) * (0.12 - 0.14) / (200 - 50) + 0.14
    #             elif 200 <= g_node_coor[n, 2]:  # if point is above z=200m, Iu=0.12
    #                 Iu[n] = 0.12
    #             else:
    #                 print('z must be positive')
    #         Iv = 0.85 * Iu
    #         Iw = 0.55 * Iu
    return np.transpose(np.array([Iu, Iv, Iw]))

def S_a_func(g_node_coor, beta_DB, f_array, Ii_simplified):
    """
    n and n_hat needs to be in Hertz, not radians!
    """
    U_bar = U_bar_func(g_node_coor, RP=RP)
    Ii = Ii_func(g_node_coor, beta_DB, Ii_simplified)
    Ai = Ai_func(cond_rand_A=False)
    iLj = iLj_func(g_node_coor)

    sigma_n = np.einsum('na,n->na', Ii, U_bar)  # standard deviation of the turbulence, for each node and each component.

    # Autospectrum
    n_hat = np.einsum('f,na,n->fna', f_array, iLj[:, :, 0], 1 / U_bar)
    S_a = np.einsum('f,na,a,fna,fna->fna', 1/f_array, sigma_n ** 2, Ai, n_hat, 1 / (1 + 1.5 * np.einsum('a,fna->fna', Ai, n_hat)) ** (5 / 3))
    return S_a

def S_a_nondim_func(g_node_coor, f_array, plot_S_a_nondim=True):
    """
    In Hertz.
    """
    U_bar = U_bar_func(g_node_coor, RP=RP)
    Ai = Ai_func(cond_rand_A=False)
    iLj = iLj_func(g_node_coor)

    # Autospectrum
    n_hat = np.einsum('f,na,n->fna', f_array, iLj[:, :, 0], 1 / U_bar)
    S_a_nondim = np.einsum('a,fna,fna->fna', Ai, n_hat, 1/(1+1.5*np.einsum('a,fna->fna', Ai, n_hat))**(5/3))

    if plot_S_a_nondim:
        plt.title('Non-dimensional auto-spectrum.')
        plt.plot(np.outer(f_array, iLj[0,0,0])/U_bar[0], S_a_nondim[:,0,0], label ='u', alpha = 0.6)
        plt.plot(np.outer(f_array, iLj[0,1,0])/U_bar[0], S_a_nondim[:,0,1], label ='v', alpha = 0.6)
        plt.plot(np.outer(f_array, iLj[0,2,0])/U_bar[0], S_a_nondim[:,0,2], label ='w', alpha = 0.6)
        plt.legend()
        plt.xscale('log')
        plt.yticks(np.arange(0, 0.30, 0.05))
        plt.grid(which="both")
        plt.xlabel(r'$f$$\/$$/\/(L_n^{x}\/U) $')
        plt.ylabel(r'$\frac{n\/S_i}{\sigma_i^{2}}$', fontsize=15, rotation=0, position=((0,0.44)))

    return S_a_nondim

def S_aa_func(g_node_coor, beta_DB, f_array, Ii_simplified, cospec_type=2):
    """
    In Hertz. The input coordinates are in Global Structural Gs (even though Gw is calculated and used in this function)
    """
    U_bar = U_bar_func(g_node_coor, RP=RP)
    iLj = iLj_func(g_node_coor)
    Cij = Cij_func(cond_rand_C=False)
    S_a = S_a_func(g_node_coor, beta_DB, f_array, Ii_simplified)  # not necessary to change from Gs to Gw

    # Transforming coordinates, because the delta_xyz needs to be in Gw for the Cij operations to make sense.
    beta_0 = beta_0_func(beta_DB)
    T_GsGw = T_GsGw_func(beta_0, theta_0)
    node_coor_wind = np.einsum('ni,ij->nj', g_node_coor, T_GsGw)  # Nodes in wind coordinates. X along, Y across, Z vertical

    # Cross-spectral density of fluctuating wind components. You-Lin Xu's formulation. Note that coherence drops down do negative values, where it stays for quite some distance:
    U_bar_avg   = ( U_bar[:, None]  +  U_bar) /2  # [m * n] matrix
    iLj_avg = (iLj[:, None] + iLj[:])/2  #
    delta_xyz = np.absolute(node_coor_wind[:, None] - node_coor_wind[:])

    # Alternative 1: LD Zhu coherence and cross-spectrum. Developed in radians? So it is converted to Hertz in the end.
    if cospec_type == 1:  # NOTE: are we sure about the units hertz and radians here?
        nxa = math.gamma(5/6) / (2*np.sqrt(np.pi) * math.gamma(1/3)) * np.einsum('wmna,mn,mna->wmna',
                                                                                 np.sqrt(1 + 70.78 * np.einsum('w,mna,mn->wmna', f_array, iLj_avg[:, :, :, 0], 1 / U_bar_avg) ** 2),
                                                                                 U_bar_avg, 1 /iLj_avg[:,:,:,0])  # f_array input, w_array output. Check L.D.Zhu
        f_hat_aa = np.einsum('wmna,mna->wmna', nxa , np.divide(np.sqrt( (Cij[:,0]*delta_xyz[:,:,0,None])**2 + (Cij[:,1]*delta_xyz[:,:,1,None])**2 + (Cij[:,2]*delta_xyz[:,:,2,None])**2 )  ,  U_bar_avg[:,:,None] ))  # This is actually in omega units, not f_array, according to eq.(10.35b)! So: rad/s
        f_hat = f_hat_aa
        R_aa = (1-f_hat)*np.e**(-f_hat)  # phase spectrum is not included because usually there is no info. See book eq.(10.34)
        S_aa_omega = np.einsum('wmna,wmna->wmna' , np.sqrt( np.einsum('wma,wna->wmna', S_a, S_a)) , R_aa )  # S_a is only (3,) because assumed no cross-correlation between components
        S_aa = 2*np.pi * S_aa_omega  # not intuitive! S(f)*delta_f = S(w)*delta_w. See eq. (2.75) from Strommen.
    # Alternative 2: Coherence and cross-spectrum (adapted Davenport for 3D). Developed in Hertz!
    elif cospec_type == 2:
        f_hat_aa = np.einsum('f,mna->fmna', f_array , np.divide(np.sqrt( (Cij[:,0]*delta_xyz[:,:,0,None])**2 + (Cij[:,1]*delta_xyz[:,:,1,None])**2 + (Cij[:,2]*delta_xyz[:,:,2,None])**2 )  ,  U_bar_avg[:,:,None] ))  # This is actually in omega units, not f_array, according to eq.(10.35b)! So: rad/s
        f_hat = f_hat_aa  # this was confirmed to be correct with a separate 4 loop "f_hat_aa_confirm" and one Cij at the time
        R_aa = np.e**(-f_hat)  # phase spectrum is not included because usually there is no info. See book eq.(10.34)
        S_aa = np.einsum('fmna,fmna->fmna' , np.sqrt( np.einsum('fma,fna->fmna', S_a, S_a)) , R_aa )  # S_a is only (3,) because assumed no cross-correlation between components
    # Plotting coherence along the g_nodes, respective to some node
    # cross_spec_1 = []
    # cross_spec_2 = []
    # cross_spec_3 = []
    # for n in range(g_node_num):
    #     cross_spec_1.append([S_aa[25,n,0, 0]])
    #     cross_spec_2.append([S_aa[25, n, 0, 1]])
    #     cross_spec_3.append([S_aa[25, n, 0, 2]])
    # plt.plot(cross_spec_1)
    # plt.plot(cross_spec_2)
    # plt.plot(cross_spec_3)
    return S_aa


########################################################################################################################
# Inhomogeneous wind
########################################################################################################################

# Since it was a stupid idea to create a method _set_S_a(self, f_array), it is now duplicated here. (S_aa were too large to store for all Nw_cases...)
def Nw_S_a(g_node_coor, f_array, Nw_U_bar, Nw_Ii):
    """
    f_array and n_hat need to be in Hertz, not radians!
    """
    Ai = Ai_func(cond_rand_A=False)
    iLj = iLj_func(g_node_coor)
    sigma_n = np.einsum('na,n->na', Nw_Ii, Nw_U_bar)  # standard deviation of the turbulence, for each node and each component.
    # Autospectrum
    n_hat = np.einsum('f,na,n->fna', f_array, iLj[:, :, 0], 1 / Nw_U_bar)
    S_a = np.einsum('f,na,a,fna,fna->fna', 1/f_array, sigma_n ** 2, Ai, n_hat, 1 / (1 + 1.5 * np.einsum('a,fna->fna', Ai, n_hat)) ** (5 / 3))
    return S_a

def Nw_S_aa(g_node_coor, Nw_beta_0, Nw_theta_0, f_array, Nw_U_bar, Nw_Ii, cospec_type=2):
    """
    In Hertz. The input coordinates are in Global Structural Gs (even though Gw is calculated and used in this function)
    """
    S_a = Nw_S_a(g_node_coor, f_array, Nw_U_bar, Nw_Ii)
    Cij = Cij_func(cond_rand_C=False)
    n_g_nodes = len(g_node_coor)

    # Difficult part. We need a cross-transformation matrix T_GsNw_avg, which is an array with shape (n_g_nodes, n_g_nodes, 3) where each (i,j) entry is the T_GsNw_avg, where Nw_avg is the avg. between Nw_i (at node i) and Nw_j (at node j)
    T_GsNw = T_GsNw_func(Nw_beta_0, Nw_theta_0)  # shape (n_g_nodes,3,3)
    Nw_Xu_Gs = np.einsum('nij,j->ni', T_GsNw, np.array([1, 0, 0]))  # Get all Nw Xu vectors. We will later average these.
    Nw_Xu_Gs_avg_nonnorm = (Nw_Xu_Gs[:, None] + Nw_Xu_Gs) / 2  # shape (n_g_nodes, n_g_nodes, 3), so each entry m,n is an average of the Xu vector at node m and the Xu vector at node n
    Nw_Xu_Gs_avg = Nw_Xu_Gs_avg_nonnorm / np.linalg.norm(Nw_Xu_Gs_avg_nonnorm, axis=2)[:, :, None]  # Normalized. shape (n_g_nodes, n_g_nodes, 3)
    Z_Gs = np.array([0, 0, 1])
    Nw_Yv_Gs_avg = np.cross(Z_Gs[None, None, :], Nw_Xu_Gs_avg)
    Nw_Zw_Gs_avg = np.cross(Nw_Xu_Gs_avg, Nw_Yv_Gs_avg)
    X_Gs = np.repeat(np.repeat(np.array([1, 0, 0])[None, None, :], repeats=n_g_nodes, axis=0), repeats=n_g_nodes, axis=1)
    Y_Gs = np.repeat(np.repeat(np.array([0, 1, 0])[None, None, :], repeats=n_g_nodes, axis=0), repeats=n_g_nodes, axis=1)
    Z_Gs = np.repeat(np.repeat(np.array([0, 0, 1])[None, None, :], repeats=n_g_nodes, axis=0), repeats=n_g_nodes, axis=1)
    # T_GsNw_avg_WRONG = (T_GsNw[:, None] + T_GsNw) / 2
    # T_GsNw_avg_RIGHT_SLOW_VERSION = np.array([[T_xyzXYZ(np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), Nw_Xu_Gs_avg[m,n], Nw_Yv_Gs_avg[m,n], Nw_Zw_Gs_avg[m,n]) for m in range(n_g_nodes)] for n in range(n_g_nodes)])
    T_GsNw_avg = np.moveaxis(T_xyzXYZ_ndarray(x=X_Gs, y=Y_Gs, z=Z_Gs, X=Nw_Xu_Gs_avg, Y=Nw_Yv_Gs_avg, Z=Nw_Zw_Gs_avg), [0, 1], [-2, -1])
    # # Benchmark: confirmation of T_GsNw_avg by an easy-step-by-step method
    # X_Gs  = np.array([1, 0, 0])
    # Y_Gs  = np.array([0, 1, 0])
    # Z_Gs  = np.array([0, 0, 1])
    # Xu_Nw = np.array([1, 0, 0])
    # Yv_Nw = np.array([0, 1, 0])
    # Zw_Nw = np.array([0, 0, 1])
    # Nw_Xu_in_Gs = np.array([T_GsNw[i] @ Xu_Nw for i in range(n_g_nodes)])
    # Nw_Yv_in_Gs = np.array([T_GsNw[i] @ Yv_Nw for i in range(n_g_nodes)])
    # Nw_Zw_in_Gs = np.array([T_GsNw[i] @ Zw_Nw for i in range(n_g_nodes)])
    # Nw_Xu_in_Gs_avg = np.zeros((n_g_nodes, n_g_nodes, 3))
    # Nw_Yv_in_Gs_avg = np.zeros((n_g_nodes, n_g_nodes, 3))
    # Nw_Zw_in_Gs_avg = np.zeros((n_g_nodes, n_g_nodes, 3))
    # def cos(i,j):
    #     return np.dot(i, j) / (np.linalg.norm(i) * np.linalg.norm(j))
    # T_GsNw_avg_2 = np.zeros((n_g_nodes, n_g_nodes, 3, 3))
    # for i in range(n_g_nodes):
    #     for j in range(n_g_nodes):
    #         Nw_Xu_in_Gs_avg[i,j] = (Nw_Xu_in_Gs[i] + Nw_Xu_in_Gs[j]) / 2  # attention: this produces non-normalized vectors
    #         Nw_Yv_in_Gs_avg[i,j] = (Nw_Yv_in_Gs[i] + Nw_Yv_in_Gs[j]) / 2  # attention: this produces non-normalized vectors
    #         Nw_Zw_in_Gs_avg[i,j] = (Nw_Zw_in_Gs[i] + Nw_Zw_in_Gs[j]) / 2  # attention: this produces non-normalized vectors
    #         T_GsNw_avg_2[i,j] = np.array([[cos(X_Gs, Nw_Xu_in_Gs_avg[i,j]), cos(X_Gs, Nw_Yv_in_Gs_avg[i,j]), cos(X_Gs, Nw_Zw_in_Gs_avg[i,j])],
    #                                       [cos(Y_Gs, Nw_Xu_in_Gs_avg[i,j]), cos(Y_Gs, Nw_Yv_in_Gs_avg[i,j]), cos(Y_Gs, Nw_Zw_in_Gs_avg[i,j])],
    #                                       [cos(Z_Gs, Nw_Xu_in_Gs_avg[i,j]), cos(Z_Gs, Nw_Yv_in_Gs_avg[i,j]), cos(Z_Gs, Nw_Zw_in_Gs_avg[i,j])]])
    # np.allclose(T_GsNw_avg, T_GsNw_avg_2)
    # Calculating the distances between pairs of points, in the average Nw_avg systems, which are the average between the Nw of one point and the Nw of the other point
    delta_xyz_Gs = g_node_coor[:,None] - g_node_coor  # Note that delta_xyz is a linear operation that could be itself transformed
    delta_xyz_Nw = np.absolute(np.einsum('mni,mnij->mnj', delta_xyz_Gs, T_GsNw_avg))
    # # SLOW confirmation version:
    # delta_xyz_Nw_SLOW = np.zeros((n_g_nodes, n_g_nodes, 3))
    # for m in range(n_g_nodes):
    #     for n in range(n_g_nodes):
    #         delta_xyz_Nw_SLOW[m,n] = np.absolute(g_node_coor[m] @ T_GsNw_avg[m,n] - g_node_coor[n] @ T_GsNw_avg[m,n])
    Nw_U_bar_avg = (Nw_U_bar[:, None] + Nw_U_bar) / 2  # from shape (n_g_nodes) to shape (n_g_nodes,n_g_nodes)

    if cospec_type == 1:  # Alternative 1: LD Zhu coherence and cross-spectrum. Developed in radians? So it is converted to Hertz in the end.
        raise NotImplementedError
    if cospec_type == 2:  # Coherence and cross-spectrum (adapted Davenport for 3D). Developed in Hertz!
        f_hat_aa = np.einsum('f,mna->fmna', f_array,
                             np.divide(np.sqrt((Cij[:, 0] * delta_xyz_Nw[:, :, 0, None]) ** 2 + (Cij[:, 1] * delta_xyz_Nw[:, :, 1, None]) ** 2 + (Cij[:, 2] * delta_xyz_Nw[:, :, 2, None]) ** 2),
                                       Nw_U_bar_avg[:, :, None]))  # This is actually in omega units, not f_array, according to eq.(10.35b)! So: rad/s
        f_hat = f_hat_aa  # this was confirmed to be correct with a separate 4 loop "f_hat_aa_confirm" and one Cij at the time
        R_aa = np.e ** (-f_hat)  # phase spectrum is not included because usually there is no info. See book eq.(10.34)
        S_aa = np.einsum('fmna,fmna->fmna', np.sqrt(np.einsum('fma,fna->fmna', S_a, S_a)), R_aa)  # S_a is only (3,) because assumed no cross-correlation between components
    return S_aa









########################################################################################################################
# Aerodynamic properties and forces
########################################################################################################################
def C_Ci_func(beta, theta, aero_coef_method, n_aero_coef, coor_system):
    """
    Returns the aerodynamic coefficients, in a 3D skew wind approach (unless '2D Lnw' is chosen for the coor_system) with shape (6,n_nodes)
    """
    assert beta.ndim == 1
    assert beta.shape == theta.shape
    beta_num = len(beta)  # Usually == g_node_num. But in Time domain, we may have to flatten betas (for all t) so len(beta)>>g_node_num

    # Finding the aerodynamic coefficients in local normal wind coordinate system Lnw:
    # Obtaining the aerodynamic coefficients for each node (represented by beta and theta).

    # # OLD VERSION, IN Lnw COORDINATES.
    # # Transformation matrix
    # T_LwLnw = T_LwLnw_func(beta, theta, dim='6x6')
    # Cx_Lnw, Cy_Lnw, Cz_Lnw, Cxx_Lnw, Cyy_Lnw, Czz_Lnw = aero_coef(copy.deepcopy(beta), copy.deepcopy(theta), plot=False, method=aero_coef_method, coor_system='Lnw')
    # # Reducing the number of aerodynamic coefficients if desired:
    # if n_aero_coef == 3:  # then only: Drag, Lift and Moment
    #     Cy_Lnw, Cxx_Lnw, Czz_Lnw = np.zeros((3,beta_num))
    # elif n_aero_coef == 4:  # then only: Drag, Lift, Moment and Axial
    #     Cxx_Lnw, Czz_Lnw = np.zeros((2,beta_num))
    # # From Local normal wind (the same as the Lwbar system when rotated by beta back to non-skew position) to Local wind coordinates.
    # C_Ci = np.einsum('nij,jn->ni', T_LwLnw, np.array([Cx_Lnw, Cy_Lnw, Cz_Lnw, Cxx_Lnw, Cyy_Lnw, Czz_Lnw])).transpose() # See L.D.Zhu thesis eq(4-25b).
    if '2D' not in coor_system:
        Cx_Ls, Cy_Ls, Cz_Ls, Cxx_Ls, Cyy_Ls, Czz_Ls = aero_coef(copy.deepcopy(beta), copy.deepcopy(theta), method=aero_coef_method, coor_system='Ls')
        # Reducing the number of aerodynamic coefficients if desired:
        if n_aero_coef == 3:  # then only: Drag, Lift and Moment
            Cx_Ls, Cyy_Ls, Czz_Ls = np.zeros((3, beta_num))
        elif n_aero_coef == 4:  # then only: Drag, Lift, Moment and Axial
            Cyy_Ls, Czz_Ls = np.zeros((2, beta_num))
        C_Ci_Ls = np.array([Cx_Ls, Cy_Ls, Cz_Ls, Cxx_Ls, Cyy_Ls, Czz_Ls])
    if coor_system == 'Ls':
        return C_Ci_Ls
    elif coor_system == 'Gw':
        T_GwLs = np.transpose(T_LsGw_func(beta, theta, dim='6x6'), axes=(0, 2, 1))
        C_Ci_Gw = np.einsum('nij,jn->in', T_GwLs, C_Ci_Ls, optimize=True)
        return C_Ci_Gw
    elif coor_system == 'Lw':
        # From Local structural to Local wind coordinates.
        T_LwLs = np.transpose(T_LsLw_func(beta, theta, dim='6x6'), axes=(0, 2, 1))
        C_Ci_Lw = np.einsum('nij,jn->in', T_LwLs, C_Ci_Ls, optimize=True)  # See L.D.Zhu thesis eq(4-25b).
        return C_Ci_Lw

    if coor_system == '3D Lnw':
        C_Ci_Lnw = C_Ci_Ls_to_C_Ci_Lnw(beta, theta, C_Ci_Ls, CS_height, CS_width, C_Ci_Lnw_normalized_by='U', drag_normalized_by='H')  # normalizing this by Uyz would give very constant coefficients!
        return C_Ci_Lnw

    assert '2D' in coor_system  # from this point onward, only '2D' systems are contemplated
    thetayz = theta_yz_bar_func(beta, theta)
    C_Ci_Ls_beta0 = aero_coef(np.zeros(beta.shape), thetayz, method=aero_coef_method, coor_system='Ls')  # function of (b=0, theta_yz)
    if coor_system == '2D Ls':
        return C_Ci_Ls_beta0
    if coor_system == '2D Lnw':
        # Only beta=0 are used, so the coefs are implicitly normalized by U=Uyz.
        # 2D Lnw are obtained from a two-variate polynomial fitting in Ls! Not from a one-variate polynomial fitting on beta=0 as could be the case
        C_Ci_Lnw = C_Ci_Ls_to_C_Ci_Lnw(np.zeros(beta.shape), thetayz, C_Ci_Ls_beta0, CS_height, CS_width, C_Ci_Lnw_normalized_by='Uyz', drag_normalized_by='H')
        return C_Ci_Lnw

def C_Ci_derivatives_func(beta, theta, aero_coef_method, n_aero_coef, coor_system):
    """
    Returns the aerodynamic coefficient derivatives in a 3D skew wind approach (unless '2D Lnw' is chosen for the coor_system)
    If coor_system = 'Lnw', it returns the coefficient derivative w.r.t thetayz, for betas = 0
    """
    assert beta.ndim == 1
    assert beta.shape == theta.shape
    assert n_aero_coef in [3, 4, 6], "n_aero_coef needs to be 3, 4 or 6"
    beta_num = len(beta)

    # # OLD VERSION, IN Lnw COORDINATES.
    # # Transformation matrix
    # T_LwLnw = T_LwLnw_func(beta, theta, dim='6x6')
    # # Finding the aerodynamic coefficient derivatives in local normal wind coordinate system, for each node:
    # [Cx_Lnw_dbeta, Cy_Lnw_dbeta, Cz_Lnw_dbeta, Cxx_Lnw_dbeta, Cyy_Lnw_dbeta, Czz_Lnw_dbeta],\
    # [Cx_Lnw_dtheta, Cy_Lnw_dtheta, Cz_Lnw_dtheta, Cxx_Lnw_dtheta, Cyy_Lnw_dtheta, Czz_Lnw_dtheta] =\
    #     aero_coef_derivatives(copy.deepcopy(beta), copy.deepcopy(theta), method=aero_coef_method, coor_system='Lnw')
    # # Reducing the number of aerodynamic coefficients if desired:
    # if n_aero_coef == 3:  # then only: Drag, Lift and Moment
    #     Cy_Lnw_dbeta, Cy_Lnw_dtheta, Cxx_Lnw_dbeta, Cxx_Lnw_dtheta, Czz_Lnw_dbeta, Czz_Lnw_dtheta = np.zeros((6,beta_num))
    # elif n_aero_coef == 4:  # then only: Drag, Lift, Moment and Axial
    #     Cxx_Lnw_dbeta, Cxx_Lnw_dtheta, Czz_Lnw_dbeta, Czz_Lnw_dtheta = np.zeros((4,beta_num))
    # C_Ci_dbeta = np.einsum('nij,jn->ni', T_LwLnw, np.array([Cx_Lnw_dbeta, Cy_Lnw_dbeta, Cz_Lnw_dbeta, Cxx_Lnw_dbeta, Cyy_Lnw_dbeta, Czz_Lnw_dbeta])).transpose()
    # C_Ci_dtheta = np.einsum('nij,jn->ni', T_LwLnw, np.array([Cx_Lnw_dtheta, Cy_Lnw_dtheta, Cz_Lnw_dtheta, Cxx_Lnw_dtheta, Cyy_Lnw_dtheta, Czz_Lnw_dtheta])).transpose()

    # Finding the aerodynamic coefficient derivatives in local structural coordinate system, for each node:
    if '2D Lnw' not in coor_system:  # if Lnw, then other coefficients, for all beta=0, need to be obtained instead.
        [Cx_Ls_dbeta, Cy_Ls_dbeta, Cz_Ls_dbeta, Cxx_Ls_dbeta, Cyy_Ls_dbeta, Czz_Ls_dbeta],\
        [Cx_Ls_dtheta, Cy_Ls_dtheta, Cz_Ls_dtheta, Cxx_Ls_dtheta, Cyy_Ls_dtheta, Czz_Ls_dtheta] =\
            aero_coef_derivatives(copy.deepcopy(beta), copy.deepcopy(theta), method=aero_coef_method, coor_system='Ls')
        # Reducing the number of aerodynamic coefficients if desired:
        if n_aero_coef == 3:  # then only: Drag, Lift and Moment
            Cx_Ls_dbeta, Cx_Ls_dtheta, Cyy_Ls_dbeta, Cyy_Ls_dtheta, Czz_Ls_dbeta, Czz_Ls_dtheta = np.zeros((6,beta_num))
        elif n_aero_coef == 4:  # then only: Drag, Lift, Moment and Axial
            Cyy_Ls_dbeta, Cyy_Ls_dtheta, Czz_Ls_dbeta, Czz_Ls_dtheta = np.zeros((4,beta_num))
        C_Ci_Ls_dbeta = np.array([Cx_Ls_dbeta, Cy_Ls_dbeta, Cz_Ls_dbeta, Cxx_Ls_dbeta, Cyy_Ls_dbeta, Czz_Ls_dbeta])
        C_Ci_Ls_dtheta = np.array([Cx_Ls_dtheta, Cy_Ls_dtheta, Cz_Ls_dtheta, Cxx_Ls_dtheta, Cyy_Ls_dtheta, Czz_Ls_dtheta])

    if coor_system == 'Ls':
        return C_Ci_Ls_dbeta, C_Ci_Ls_dtheta
    elif coor_system == 'Gw':
        C_Ci_Ls = C_Ci_func(beta, theta, aero_coef_method, n_aero_coef, coor_system='Ls')
        T_GwLs_dbeta, T_GwLs_dtheta = T_GwLs_derivatives_func(beta, theta, dim='6x6')
        T_GwLs_6 = np.transpose(T_LsGw_func(beta, theta, dim='6x6'), axes=(0, 2, 1))
        C_Ci_Gw_dbeta = np.einsum('nij,jn->in', T_GwLs_dbeta, C_Ci_Ls) + np.einsum('nij,jn->in', T_GwLs_6 , C_Ci_Ls_dbeta)  # C_2 = T_21 @ C_1. Derive both sides: C_2_db = T_21_db @ C_1 + T_21 @ C_1_db
        C_Ci_Gw_dtheta = np.einsum('nij,jn->in', T_GwLs_dtheta, C_Ci_Ls) + np.einsum('nij,jn->in', T_GwLs_6 , C_Ci_Ls_dtheta)  # C_2 = T_21 @ C_1. Derive both sides: C_2_dt = T_21_dt @ C_1 + T_21 @ C_1_dt
        return C_Ci_Gw_dbeta, C_Ci_Gw_dtheta
    elif coor_system == 'Lw':  # Lw as in L.D.Zhu "Lwbar"
        C_Ci_Ls = C_Ci_func(beta, theta, aero_coef_method, n_aero_coef, coor_system='Ls')
        T_LwLs_dbeta, T_LwLs_dtheta = T_LwLs_derivatives_func(beta, theta, dim='6x6')
        T_LwLs_6 = np.transpose(T_LsLw_func(beta, theta, dim='6x6'), axes=(0, 2, 1))
        C_Ci_Lw_dbeta = np.einsum('nij,jn->in', T_LwLs_dbeta, C_Ci_Ls) + np.einsum('nij,jn->in', T_LwLs_6 , C_Ci_Ls_dbeta)  # C_2 = T_21 @ C_1. Derive both sides: C_2_db = T_21_db @ C_1 + T_21 @ C_1_db
        C_Ci_Lw_dtheta = np.einsum('nij,jn->in', T_LwLs_dtheta, C_Ci_Ls) + np.einsum('nij,jn->in', T_LwLs_6 , C_Ci_Ls_dtheta)  # C_2 = T_21 @ C_1. Derive both sides: C_2_dt = T_21_dt @ C_1 + T_21 @ C_1_dt
        return C_Ci_Lw_dbeta, C_Ci_Lw_dtheta
    elif coor_system == '2D Lnw':  # Local normal wind
        thetayz = theta_yz_bar_func(beta, theta)
        _, C_Ci_Ls_dthetayz = aero_coef_derivatives(np.zeros(beta.shape), copy.deepcopy(thetayz), method=aero_coef_method, coor_system='Ls')  # using beta = 0, so that theta = thetayz
        C_Ci_Ls_beta0 = aero_coef(np.zeros(beta.shape), thetayz, method=aero_coef_method, coor_system='Ls')  # function of (b=0, theta_yz)
        T_LnwLs_dthetayz = T_LnwLs_dtheta_yz_func(thetayz, dim='6x6')
        T_LnwLs_6 = T_LnwLs_func(beta, thetayz, dim='6x6')
        C_Ci_Lnw_dthetayz = np.einsum('nij,jn->in', T_LnwLs_dthetayz, C_Ci_Ls_beta0) + np.einsum('nij,jn->in', T_LnwLs_6 , C_Ci_Ls_dthetayz)  # C_2 = T_21 @ C_1. Derive both sides: C_2_dt = T_21_dt @ C_1 + T_21 @ C_1_dt
        C_Ci_Lnw_dthetayz[0,:] = C_Ci_Lnw_dthetayz[0,:] * CS_width / CS_height  # Normalizing Cd by H instead of B
        return C_Ci_Lnw_dthetayz

def Chi_Ci_func(coor_system='Gw'):
    # Aerodynamic admittance function
    if coor_system == 'Gw':
        Chi_Ci_u = np.ones(6)
        Chi_Ci_v = np.ones(6)
        Chi_Ci_w = np.ones(6)
        return Chi_Ci_u, Chi_Ci_v, Chi_Ci_w
    elif coor_system == 'Lnw':
        Chi_Ci_aD = np.ones(6)
        Chi_Ci_aA = np.ones(6)
        Chi_Ci_aL = np.ones(6)
        return Chi_Ci_aD, Chi_Ci_aA, Chi_Ci_aL

def A_bar_func(U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach, Chi_Ci='Chi function'):
    """ In Lw coordinates as in L.D. Zhu
    """
    # The following was numerically compared to be the same as the matrix by Zhu using the many s's and t's.
    if skew_approach == '3D':
        if Chi_Ci == 'Chi function':
            Chi_Ci_u, Chi_Ci_v, Chi_Ci_w = Chi_Ci_func(coor_system='Gw')
        elif Chi_Ci == 'ones':  # used in the QS motion-dependent Cse, when importing A_bar(Chi=1)
            Chi_Ci_u = np.ones(6)
            Chi_Ci_v = np.ones(6)
            Chi_Ci_w = np.ones(6)
        C_Ci_Gw = C_Ci_func(beta_bar, theta_bar, aero_coef_method, n_aero_coef, coor_system='Gw')
        C_Ci_Gw_dbeta, C_Ci_Gw_dtheta = C_Ci_derivatives_func(beta_bar, theta_bar, aero_coef_method, n_aero_coef, coor_system='Gw')
        A_Gw = 1 / 2 * rho * np.einsum('n,vcn->nvc', U_bar, np.array(
          [[   2*CS_width*C_Ci_Gw[0]*Chi_Ci_u[0],                                 CS_width*(C_Ci_Gw_dbeta[0]/np.cos(theta_bar)-C_Ci_Gw[1])*Chi_Ci_v[0],    CS_width*(C_Ci_Gw_dtheta[0]-C_Ci_Gw[2])*Chi_Ci_w[0]],
           [   2*CS_width*C_Ci_Gw[1]*Chi_Ci_u[1],    CS_width*(C_Ci_Gw[0]+C_Ci_Gw_dbeta[1]/np.cos(theta_bar)-C_Ci_Gw[2]*np.tan(theta_bar))*Chi_Ci_v[1],                 CS_width*C_Ci_Gw_dtheta[1]*Chi_Ci_w[1]],
           [   2*CS_width*C_Ci_Gw[2]*Chi_Ci_u[2],               CS_width*(C_Ci_Gw[1]*np.tan(theta_bar)+C_Ci_Gw_dbeta[2]/np.cos(theta_bar))*Chi_Ci_v[2],    CS_width*(C_Ci_Gw[0]+C_Ci_Gw_dtheta[2])*Chi_Ci_w[2]],
           [2*CS_width**2*C_Ci_Gw[3]*Chi_Ci_u[3],                              CS_width**2*(C_Ci_Gw_dbeta[3]/np.cos(theta_bar)-C_Ci_Gw[4])*Chi_Ci_v[3], CS_width**2*(C_Ci_Gw_dtheta[3]-C_Ci_Gw[5])*Chi_Ci_w[3]],
           [2*CS_width**2*C_Ci_Gw[4]*Chi_Ci_u[4], CS_width**2*(C_Ci_Gw[3]+C_Ci_Gw_dbeta[4]/np.cos(theta_bar)-C_Ci_Gw[5]*np.tan(theta_bar))*Chi_Ci_v[4],              CS_width**2*C_Ci_Gw_dtheta[4]*Chi_Ci_w[4]],
           [2*CS_width**2*C_Ci_Gw[5]*Chi_Ci_u[5],            CS_width**2*(C_Ci_Gw[4]*np.tan(theta_bar)+C_Ci_Gw_dbeta[5]/np.cos(theta_bar))*Chi_Ci_v[5], CS_width**2*(C_Ci_Gw[3]+C_Ci_Gw_dtheta[5])*Chi_Ci_w[5]]]), optimize=True)
        A_LwGw = np.einsum('ij,njk->nik', T_LwGw_func(dim='6x6'), A_Gw, optimize=True)
        return A_LwGw
    if skew_approach in ['2D','2D+1D','2D_cos_law']:
        if skew_approach in ['2D','2D+1D']:
            theta_bar_or_0 = copy.deepcopy(theta_bar)
        elif skew_approach == '2D_cos_law':
            theta_bar_or_0 = np.zeros(theta_bar.shape)
        if Chi_Ci == 'Chi function':
            Chi_Ci_aD, Chi_Ci_aA, Chi_Ci_aL = Chi_Ci_func(coor_system='Lnw')
        elif Chi_Ci == 'ones':  # used in the QS motion-dependent Cse, when importing A_bar(Chi=1)
            Chi_Ci_aD = np.ones(6)
            Chi_Ci_aA = np.ones(6)
            Chi_Ci_aL = np.ones(6)
        T_LsGw = T_LsGw_func(beta_bar, theta_bar_or_0, dim='3x3')
        U_bar_Gw = np.array([[U_bar[i], 0, 0] for i in range(len(U_bar))])
        U_bar_Ls = np.einsum('nij,nj->ni', T_LsGw, U_bar_Gw)
        U_yz = np.sqrt(U_bar_Ls[:,1]**2 + U_bar_Ls[:,2]**2)
        theta_yz = theta_yz_bar_func(beta_bar, theta_bar_or_0)
        C_Ci_Lnw_2D = C_Ci_func(beta_bar, theta_bar_or_0, aero_coef_method, n_aero_coef, coor_system='2D Lnw')  # beta_bar and theta_bar needed to calculate theta_yz inside the function. Then, betas = 0.
        C_Ci_Lnw_2D_dtheta = C_Ci_derivatives_func(beta_bar, theta_bar_or_0, aero_coef_method, n_aero_coef, coor_system='2D Lnw')
        zeros = np.zeros(len(beta_bar))
        A_Lnw = 1 / 2 * rho * np.einsum('n,vcn->nvc', U_yz, np.array(
          [[  2*CS_height*C_Ci_Lnw_2D[0]*Chi_Ci_aD[0], zeros,    (CS_height*C_Ci_Lnw_2D_dtheta[0] - CS_width*C_Ci_Lnw_2D[2])*Chi_Ci_aL[0]],
           [                                    zeros, zeros,                                                                       zeros],
           [   2*CS_width*C_Ci_Lnw_2D[2]*Chi_Ci_aD[2], zeros,    (CS_width*C_Ci_Lnw_2D_dtheta[2] + CS_height*C_Ci_Lnw_2D[0])*Chi_Ci_aL[2]],
           [                                    zeros, zeros,                                                                       zeros],
           [2*CS_width**2*C_Ci_Lnw_2D[4]*Chi_Ci_aD[4], zeros,                              CS_width**2*C_Ci_Lnw_2D_dtheta[4]*Chi_Ci_aL[4]],
           [                                    zeros, zeros,                                                                       zeros]]), optimize=True)
        T_LnwLs = T_LnwLs_func(beta_bar, theta_yz, dim='3x3')
        T_LnwLs_6 = T_LnwLs_func(beta_bar, theta_yz, dim='6x6')
        if skew_approach == '2D+1D':
            U_x = U_bar_Ls[:,0]
            Cx = C_Ci_func(np.ones(beta_bar.shape)*(-np.pi/2), np.zeros(theta_bar.shape), aero_coef_method, n_aero_coef, coor_system='Ls')[0]
            A_Ls_axial = 1 / 2 * rho * CS_width * np.einsum('n,vcn->nvc', np.abs(U_x), np.array(
                [[2*Cx*Chi_Ci_aA[1], zeros, zeros],
                 [            zeros, zeros, zeros],
                 [            zeros, zeros, zeros],
                 [            zeros, zeros, zeros],
                 [            zeros, zeros, zeros],
                 [            zeros, zeros, zeros]]), optimize=True)
            T_LsLnw = np.transpose(T_LnwLs, axes=(0, 2, 1))
            A_Lnw_axial = T_LnwLs_6 @ A_Ls_axial @ T_LsLnw
            A_Lnw = A_Lnw + A_Lnw_axial
        T_LnwGw = T_LnwLs @ T_LsGw
        A_LnwGw = np.einsum('nij,njk->nik',A_Lnw, T_LnwGw)
        T_LsLw = T_LsLw_func(beta_bar, theta_bar_or_0, dim='6x6')
        T_LwLnw = np.transpose(T_LnwLs_6 @ T_LsLw, axes=(0, 2, 1))
        A_LwGw = np.einsum('nij,njk->nik', T_LwLnw, A_LnwGw)
        return A_LwGw

def flutter_derivatives_func(U_bar, beta_bar, theta_bar, f_array, flutter_derivatives_type, aero_coef_method, n_aero_coef):
    w_array = f_array * 2 * np.pi
    n_freq = len(f_array)
    n_g_nodes = len(beta_bar)
    K_array = np.outer(w_array * CS_width, 1 / U_bar)  # Reduced frequency. (beginning of L.D.Zhu book page 4-23)

    b = beta_bar
    t = theta_bar
    K = K_array

    # Aerodynamic coefficients function
    C_Ci = C_Ci_func(beta_bar, theta_bar, aero_coef_method, n_aero_coef, coor_system='Lw')
    C_Ci_dbeta, C_Ci_dtheta = C_Ci_derivatives_func(beta_bar, theta_bar, aero_coef_method, n_aero_coef, coor_system='Lw')

    # Aerodynamic coefficients
    C_Cq = C_Ci[0]
    C_Dp = C_Ci[1]
    C_Lh = C_Ci[2]
    C_Malpha = C_Ci[3]
    C_Mgamma = C_Ci[4]
    C_Mphi = C_Ci[5]
    # ...and their partial derivatives with respect to mean wind yaw angle (beta) (eq.14):
    C_Cq_dbeta = C_Ci_dbeta[0]
    C_Dp_dbeta = C_Ci_dbeta[1]
    C_Lh_dbeta = C_Ci_dbeta[2]
    C_Malpha_dbeta = C_Ci_dbeta[3]
    C_Mgamma_dbeta = C_Ci_dbeta[4]
    C_Mphi_dbeta = C_Ci_dbeta[5]
    # ...or to mean angle of attack (theta).
    C_Cq_dtheta = C_Ci_dtheta[0]
    C_Dp_dtheta = C_Ci_dtheta[1]
    C_Lh_dtheta = C_Ci_dtheta[2]
    C_Malpha_dtheta = C_Ci_dtheta[3]
    C_Mgamma_dtheta = C_Ci_dtheta[4]
    C_Mphi_dtheta = C_Ci_dtheta[5]

    def cos(x):
        return np.cos(x)
    def sin(x):
        return np.sin(x)
    def cos2(x):
        return np.cos(x)**2
    def sin2(x):
        return np.sin(x)**2

    if 'Zhu' in flutter_derivatives_type or flutter_derivatives_type == '3D_Scanlan_confirm':
        P1_star = -1/K * (((2*cos2(t)-1)*sin(b)*cos(b)/cos(t))*C_Cq
                          +(1 + cos2(b) * cos2(t)) * C_Dp
                          -((sin2(b) + cos2(b) * cos2(t))*np.tan(t))*C_Lh
                          -(sin2(b)/cos(t))*C_Cq_dbeta-(sin(b)*cos(b)*sin(t))*C_Cq_dtheta
                          -(sin(b)*cos(b))*C_Dp_dbeta - (cos2(b)*sin(t)*cos(t))*C_Dp_dtheta
                          +(sin(b)*cos(b)*np.tan(t))*C_Lh_dbeta + (cos2(b)*sin2(t))*C_Lh_dtheta)
        P2_star = np.zeros((n_freq, n_g_nodes))
        P3_star = -1/(K**2) * ((sin(b)*cos(b))*C_Cq_dtheta + (cos2(b)*cos(t))*C_Dp_dtheta
                               -(cos2(b)*sin(t))*C_Lh_dtheta)
        if flutter_derivatives_type == '3D_Scanlan_confirm':  # the following difference between Zhu and BC comes from Sympy
           P3_star = P3_star + (-C_Dp_dbeta*sin(t)*cos(b)*cos(t) - C_Cq*sin(t)*cos(b) - C_Cq_dbeta*sin(b)*sin(t) - C_Lh*sin(b) + C_Lh_dbeta*sin(t)**2*cos(b))*sin(b)/(K**2*cos(t))
        P4_star = np.zeros((n_freq, n_g_nodes))
        P5_star = -1/K * ((2*sin(b)*sin(t))*C_Cq + (cos(b)*sin(t)*cos(t))*C_Dp
                          -((2-cos2(t))*cos(b))*C_Lh + (sin(b)*cos(t))*C_Cq_dtheta   # in L.D. Zhu it is instead "(sin(b)*cos(b))*C_Cq_dtheta". This is a typo in his theory. I found it in Sympy and corrected here.
                          +(cos(b)*cos2(t))*C_Dp_dtheta - (cos(b)*sin(t)*cos(t))*C_Lh_dtheta)
        if flutter_derivatives_type == '3D_Zhu_bad_P5':
            P5_star = -1/K * ((2*sin(b)*sin(t))*C_Cq + (cos(b)*sin(t)*cos(t))*C_Dp
                             -((2-cos2(t))*cos(b))*C_Lh + (sin(b)*cos(b))*C_Cq_dtheta   # in L.D. Zhu it is instead "(sin(b)*cos(b))*C_Cq_dtheta". This is a typo in his theory. I found it in Sympy and corrected here.
                             +(cos(b)*cos2(t))*C_Dp_dtheta - (cos(b)*sin(t)*cos(t))*C_Lh_dtheta)
        P6_star = np.zeros((n_freq, n_g_nodes))
        H1_star = -1/K * ((2-cos2(t))*C_Dp + cos2(t)*C_Lh_dtheta + (sin(t)*cos(t))*(C_Lh + C_Dp_dtheta))
        H2_star = np.zeros((n_freq, n_g_nodes))
        H3_star = -1/(K**2) * ((cos(b)*sin(t))*C_Dp_dtheta + (cos(b)*cos(t))*C_Lh_dtheta)
        if flutter_derivatives_type == '3D_Scanlan_confirm':  # the following difference between Zhu and BC comes from Sympy
            H3_star = H3_star + (-(C_Dp_dbeta*sin(t)**2/cos(t) - C_Cq + C_Lh_dbeta*sin(t))*sin(b)/K**2)
        H4_star = np.zeros((n_freq, n_g_nodes))
        H5_star = -1/K * ((cos(b)*sin(t)*cos(t))*C_Dp + (cos(b)*(1+cos2(t)))*C_Lh
                          -(sin(b)*np.tan(t))*C_Dp_dbeta - (cos(b)*sin2(t))*C_Dp_dtheta
                          -(sin(b))*C_Lh_dbeta - (cos(b)*sin(t)*cos(t))*C_Lh_dtheta)
        H6_star = np.zeros((n_freq, n_g_nodes))
        A1_star = -1/K*((2*cos(b)*sin(t))*C_Malpha - (sin(b)*sin(t)*cos(t))*C_Mgamma
                        +((2-cos2(t))*sin(b))*C_Mphi + (cos(b)*cos(t))*C_Malpha_dtheta
                        -(sin(b)*cos2(t))*C_Mgamma_dtheta + (sin(b)*sin(t)*cos(t))*C_Mphi_dtheta)
        A2_star = np.zeros((n_freq, n_g_nodes))
        # As by L.D. Zhu:
        A3_star = -1/(K**2) * ((cos2(b))*C_Malpha_dtheta - (sin(b)*cos(b)*cos(t))*C_Mgamma_dtheta
                               +(sin(b)*cos(b)*sin(t))*C_Mphi_dtheta)
        if flutter_derivatives_type == '3D_Scanlan_confirm':  # the following difference between Zhu and BC comes from Sympy
            A3_star = A3_star + (C_Mgamma_dbeta*sin(b)*sin(t)*cos(t) + C_Malpha*sin(b)*sin(t) - C_Malpha_dbeta*sin(t)*cos(b) - C_Mphi*cos(b) - C_Mphi_dbeta*sin(b)*sin(t)**2)*sin(b)/(K**2*cos(t))
        A4_star = np.zeros((n_freq, n_g_nodes))
        A5_star = - 1/K * (((sin2(b) + 2*cos2(b)*cos2(t))/cos(t))*C_Malpha
                           -(sin(b)*cos(b)*cos2(t))*C_Mgamma - (sin(b)*cos(b)*sin2(t)*np.tan(t))*C_Mphi
                           -(sin(b)*cos(b)/cos(t))*C_Malpha_dbeta - (cos2(b)*sin(t))*C_Malpha_dtheta
                           +(sin2(b))*C_Mgamma_dbeta + (sin(b)*cos(b)*sin(t)*cos(t))*C_Mgamma_dtheta
                           -(sin2(b)*np.tan(t))*C_Mphi_dbeta - (sin(b)*cos(b)*sin2(t)) * C_Mphi_dtheta)
        A6_star = np.zeros((n_freq, n_g_nodes))

    # DELETE? NOT IN USE
    if flutter_derivatives_type == 'QS non-skew':  # according to strommen. Independent of beta
        V_red = 1/K_array
        P1_star = -2*C_Dp * V_red
        P2_star = np.zeros((n_freq, n_g_nodes))
        P3_star = C_Dp_dtheta * V_red**2
        P4_star = np.zeros((n_freq, n_g_nodes))
        P5_star = (C_Lh-C_Dp_dtheta)* V_red
        P6_star = np.zeros((n_freq, n_g_nodes))
        H1_star = -(C_Lh_dtheta+C_Dp) * V_red
        H2_star = np.zeros((n_freq, n_g_nodes))
        H3_star = C_Lh_dtheta * V_red**2
        H4_star = np.zeros((n_freq, n_g_nodes))
        H5_star = -2*C_Lh * V_red
        H6_star = np.zeros((n_freq, n_g_nodes))
        A1_star = -C_Malpha_dtheta * V_red
        A2_star = np.zeros((n_freq, n_g_nodes))
        A3_star = C_Malpha_dtheta * V_red**2
        A4_star = np.zeros((n_freq, n_g_nodes))
        A5_star = -2*C_Malpha * V_red
        A6_star = np.zeros((n_freq, n_g_nodes))

    # DELETE? NOT IN USE
    if flutter_derivatives_type == 'QS cosine':  # according to L.D.Zhu.
        V_red = 1/K_array
        P1_star = -2*C_Dp * V_red
        P2_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)  # these cosines might give problems for beta = 90
        P3_star = C_Dp_dtheta * V_red**2 / np.cos(b)
        P4_star = np.zeros((n_freq, n_g_nodes))
        P5_star = (C_Lh-C_Dp_dtheta)* V_red / np.cos(b)
        P6_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)
        H1_star = -(C_Lh_dtheta+C_Dp) * V_red / np.cos(b)**2
        H2_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)**2
        H3_star = C_Lh_dtheta * V_red**2 / np.cos(b)**2
        H4_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)**2
        H5_star = -2*C_Lh * V_red / np.cos(b)
        H6_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)
        A1_star = -C_Malpha_dtheta * V_red / np.cos(b)**2
        A2_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)**2
        A3_star = C_Malpha_dtheta * V_red**2 / np.cos(b)**2
        A4_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)**2
        A5_star = -2*C_Malpha * V_red / np.cos(b)
        A6_star = np.zeros((n_freq, n_g_nodes)) / np.cos(b)

    return P1_star, P2_star, P3_star, P4_star, P5_star, P6_star, H1_star, H2_star, H3_star, \
           H4_star, H5_star, H6_star, A1_star, A2_star, A3_star, A4_star, A5_star, A6_star

def Kse_Cse_func(g_node_coor, U_bar, beta_bar, theta_bar, alpha, f_array, flutter_derivatives_type, aero_coef_method, n_aero_coef, skew_approach):
    """
    Linearized aerodynamic stiffness and damping matrices
    shape (n_freq,n_nodes,6,6)
    """
    if skew_approach == '2D_cos_law':
        theta_bar_or_0 = np.zeros(theta_bar.shape)
    else:
        theta_bar_or_0 = copy.deepcopy(theta_bar)
    g_node_num = len(g_node_coor)
    g_node_L_3D = g_node_L_3D_func(g_node_coor)
    T_LsGs_6 = T_LsGs_6g_func(g_node_coor, alpha)
    T_GsLs_6 = np.transpose(T_LsGs_6, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LsLw_6 = T_LsLw_func(beta_bar, theta_bar_or_0, dim='6x6')
    T_LwLs_6 = np.transpose(T_LsLw_6, axes=(0, 2, 1))
    if skew_approach == '3D':
        assert '3D' in flutter_derivatives_type
        if flutter_derivatives_type in ['3D_Zhu', '3D_Zhu_bad_P5', '3D_Scanlan_confirm']:
            w_array = f_array * 2 * np.pi
            K_array = np.outer(w_array * CS_width, 1 / U_bar)  # Reduced frequency. (beginning of L.D.Zhu book page 4-23)

            flutter_derivatives = flutter_derivatives_func(U_bar, beta_bar, theta_bar, f_array, flutter_derivatives_type, aero_coef_method, n_aero_coef)
            P1_star, P2_star, P3_star, P4_star, P5_star, P6_star, H1_star, H2_star, H3_star, H4_star, H5_star, H6_star, \
            A1_star, A2_star, A3_star, A4_star, A5_star, A6_star = flutter_derivatives
            # Aerodynamic stiffness of Malpha, Dp, Lh (eq. 4-88):
            As = 1 / 2 * rho * U_bar ** 2 * K_array ** 2 * np.array(
                [[CS_width ** 2 * A3_star, CS_width * A6_star, CS_width * A4_star],
                 [CS_width * P3_star, P4_star, P6_star],
                 [CS_width * H3_star, H6_star, H4_star]])
            # Aerodynamic damping of Malpha, Dp, Lh (eq. 4-89):
            Ad = 1 / 2 * rho * U_bar * CS_width * K_array * np.array(
                [[CS_width ** 2 * A2_star, CS_width * A5_star, CS_width * A1_star],
                 [CS_width * P2_star, P1_star, P5_star],
                 [CS_width * H2_star, H5_star, H1_star]])
            # Converting the eq. 4-88 from (3x3) (Malpha, Dp, Lh), to (6x6) (qph) system. Confirmation: np.round(np.transpose(T_MDL_to_qph) @ As[:,:,0,0] @ T_MDL_to_qph)
            # This can be obtained by the N_SE_interpolation_matrix_func when xi is 0 or 1.
            T_MDL_to_qph = np.array([[0., 0., 0., 1., 0., 0.],
                                     [0., 1., 0., 0., 0., 0.],
                                     [0., 0., 1., 0., 0., 0.]])
            # Final aeroelastic stiffness matrix. Global nodal matrix (girder only) (not elements as in L.D.Zhu).
            Kse = np.einsum('ij,jkwm,kl->wmil', np.transpose(T_MDL_to_qph), As, T_MDL_to_qph,
                            optimize=True)  # MDL = moment, drag, lift
            Kse = - np.einsum('wmjl,m->wmjl', Kse, g_node_L_3D, optimize=True)
            Kse = np.einsum('mij,wmjk,mkl->wmil', T_GsLs_6, Kse, T_LsGs_6,
                            optimize=True)  # from local xyz to global XYZ coordinates. There is no dependency between g_nodes so it can be done for each node, not at a huge (g_nodes x g_nodes) matrix.
            # Final aeroelastic damping matrix. Global nodal matrix (girder only) (not elements as in L.D.Zhu).
            Cse = np.einsum('ij,jkwm,kl->wmil', np.transpose(T_MDL_to_qph), Ad, T_MDL_to_qph, optimize=True)
            Cse = - np.einsum('wmjl,m->wmjl', Cse, g_node_L_3D, optimize=True)
            Cse = np.einsum('mij,wmjk,mkl->wmil', T_GsLs_6, Cse, T_LsGs_6,
                            optimize=True)  # from local xyz to global XYZ coordinates. There is no dependency between g_nodes so it can be done for each node, not at a huge (g_nodes x g_nodes) matrix.
            return Kse, Cse

        # If flutter derivatives are "QS" = quasi-static, there's actually no need for a frequency dimension
        if flutter_derivatives_type in ['3D_full','3D_Scanlan']:
            C_Ci_Gw = C_Ci_func(beta_bar, theta_bar, aero_coef_method, n_aero_coef, coor_system='Gw')
            C_Ci_Gw_dbeta, C_Ci_Gw_dtheta = C_Ci_derivatives_func(beta_bar, theta_bar, aero_coef_method, n_aero_coef, coor_system='Gw')
            zeros = np.zeros(g_node_num)
            A_delta_Gw = 1/2 * rho * np.einsum('n,ijn->nij', U_bar**2 , np.array([
                [zeros,zeros,zeros,                     zeros,    CS_width * C_Ci_Gw_dtheta[0],                                       -CS_width * C_Ci_Gw_dbeta[0] / np.cos(theta_bar)],
                [zeros,zeros,zeros,    -CS_width * C_Ci_Gw[2],    CS_width * C_Ci_Gw_dtheta[1],    -CS_width * (C_Ci_Gw_dbeta[1] - C_Ci_Gw[2] * np.sin(theta_bar)) / np.cos(theta_bar)],
                [zeros,zeros,zeros,     CS_width * C_Ci_Gw[1],    CS_width * C_Ci_Gw_dtheta[2],    -CS_width * (C_Ci_Gw_dbeta[2] + C_Ci_Gw[1] * np.sin(theta_bar)) / np.cos(theta_bar)],
                [zeros,zeros,zeros,                     zeros, CS_width**2 * C_Ci_Gw_dtheta[3],                                    -CS_width**2 * C_Ci_Gw_dbeta[3] / np.cos(theta_bar)],
                [zeros,zeros,zeros, -CS_width**2 * C_Ci_Gw[5], CS_width**2 * C_Ci_Gw_dtheta[4], -CS_width**2 * (C_Ci_Gw_dbeta[4] - C_Ci_Gw[5] * np.sin(theta_bar)) / np.cos(theta_bar)],
                [zeros,zeros,zeros,  CS_width**2 * C_Ci_Gw[4], CS_width**2 * C_Ci_Gw_dtheta[5], -CS_width**2 * (C_Ci_Gw_dbeta[5] + C_Ci_Gw[4] * np.sin(theta_bar)) / np.cos(theta_bar)]]), optimize=True)
            A_delta_Lw = T_LwGw_func(dim='6x6') @ A_delta_Gw @ np.transpose(T_LwGw_func(dim='6x6'))
            A_delta_d_LwGw = np.zeros((g_node_num,6,6))
            A_delta_d_LwGw[:,:,:3] = - A_bar_func(U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach='3D', Chi_Ci='ones')
            A_delta_Ls = T_LsLw_6 @ A_delta_Lw @ T_LwLs_6
            A_delta_d_Ls = T_LsLw_6 @ A_delta_d_LwGw @ np.transpose(T_LwGw_func(dim='6x6')) @ T_LwLs_6
            if flutter_derivatives_type == '3D_Scanlan':
                A_delta_Ls_2 = np.zeros((g_node_num,6,6))        # deleting all entries except the ones corresponding to the traditional Pi,Hi,Ai for i=1,2,3,4,5
                A_delta_Ls_2[:,1:4,1:4] = A_delta_Ls[:,1:4,1:4]  # deleting all entries except the ones corresponding to the traditional Pi,Hi,Ai for i=1,2,3,4,5
                A_delta_Ls = A_delta_Ls_2                        # deleting all entries except the ones corresponding to the traditional Pi,Hi,Ai for i=1,2,3,4,5
                A_delta_d_Ls_2 = np.zeros((g_node_num,6,6))         # deleting all entries except the ones corresponding to the traditional Pi,Hi,Ai for i=1,2,3,4,5
                A_delta_d_Ls_2[:,1:4,1:4] = A_delta_d_Ls[:,1:4,1:4] # deleting all entries except the ones corresponding to the traditional Pi,Hi,Ai for i=1,2,3,4,5
                A_delta_d_Ls = A_delta_d_Ls_2                       # deleting all entries except the ones corresponding to the traditional Pi,Hi,Ai for i=1,2,3,4,5

    elif skew_approach in ['2D','2D+1D','2D_cos_law']:
        assert flutter_derivatives_type in ['2D_full','2D_in_plane']
        T_LsGw = T_LsGw_func(beta_bar, theta_bar_or_0, dim='3x3')
        U_bar_Gw = np.array([[U_bar[i], 0, 0] for i in range(len(U_bar))])
        U_bar_Ls = np.einsum('nij,nj->ni', T_LsGw, U_bar_Gw)
        U_yz = np.sqrt(U_bar_Ls[:,1]**2 + U_bar_Ls[:,2]**2)
        theta_yz = theta_yz_bar_func(beta_bar, theta_bar_or_0)
        T_LnwLs_6 = T_LnwLs_func(beta_bar, theta_yz, dim='6x6')
        T_LsLnw_6 = np.transpose(T_LnwLs_6, axes=(0, 2, 1))
        C_Ci_Lnw_2D = C_Ci_func(beta_bar, theta_bar_or_0, aero_coef_method, n_aero_coef, coor_system='2D Lnw')   # beta_bar and theta_bar needed to calculate theta_yz inside the function. Then, betas = 0.
        C_Ci_Lnw_2D_dtheta = C_Ci_derivatives_func(beta_bar, theta_bar_or_0, aero_coef_method, n_aero_coef, coor_system='2D Lnw')
        zeros = np.zeros(len(beta_bar))
        signs = np.sign(np.cos(beta_bar))
        A_delta_Lnw = 1 / 2 * rho * np.einsum('n,ijn->nij', U_yz**2, np.array([
                [zeros,zeros,zeros,  signs*(CS_width*C_Ci_Lnw_2D[2]-CS_height*C_Ci_Lnw_2D_dtheta[0])*np.sin(beta_bar)*np.cos(theta_bar_or_0)*U_bar/U_yz,   CS_height*C_Ci_Lnw_2D_dtheta[0],   2*signs*CS_height*C_Ci_Lnw_2D[0]*np.sin(beta_bar)*np.cos(theta_bar_or_0)*U_bar/U_yz],
                [zeros,zeros,zeros,                                                                                            -CS_width*C_Ci_Lnw_2D[2],                             zeros,                                                              CS_height*C_Ci_Lnw_2D[0]],
                [zeros,zeros,zeros, -signs*(CS_height*C_Ci_Lnw_2D[0]+CS_width*C_Ci_Lnw_2D_dtheta[2])*np.sin(beta_bar)*np.cos(theta_bar_or_0)*U_bar/U_yz,    CS_width*C_Ci_Lnw_2D_dtheta[2],    2*signs*CS_width*C_Ci_Lnw_2D[2]*np.sin(beta_bar)*np.cos(theta_bar_or_0)*U_bar/U_yz],
                [zeros,zeros,zeros,                                                                                                               zeros,                             zeros,                                                           -CS_width**2*C_Ci_Lnw_2D[4]],
                [zeros,zeros,zeros,                         -signs*CS_width**2*C_Ci_Lnw_2D_dtheta[4]*np.sin(beta_bar)*np.cos(theta_bar_or_0)*U_bar/U_yz, CS_width**2*C_Ci_Lnw_2D_dtheta[4], 2*signs*CS_width**2*C_Ci_Lnw_2D[4]*np.sin(beta_bar)*np.cos(theta_bar_or_0)*U_bar/U_yz],
                [zeros,zeros,zeros,                                                                                          CS_width**2*C_Ci_Lnw_2D[4],                             zeros,                                                                               zeros]]), optimize=True)
        if flutter_derivatives_type == '2D_in_plane':
            # Then only the dependencies on delta_M are accounted, as in e.g. Strommen.
            A_delta_Lnw[:,:,3] = np.zeros(6)  # erasing dependencies on delta_rD
            A_delta_Lnw[:,:,5] = np.zeros(6)  # erasing dependencies on delta_rL
        A_delta_Ls = T_LsLnw_6 @ A_delta_Lnw @ T_LnwLs_6
        A_delta_d_LwGw = np.zeros((g_node_num, 6, 6))
        A_delta_d_LwGw[:, :, :3] = - A_bar_func(U_bar, beta_bar, theta_bar_or_0, aero_coef_method, n_aero_coef, skew_approach='2D', Chi_Ci='ones')
        A_delta_d_Ls = T_LsLw_6 @ A_delta_d_LwGw @ np.transpose(T_LwGw_func(dim='6x6')) @ T_LwLs_6

        if skew_approach == '2D+1D':
            assert flutter_derivatives_type in ['2D_full','2D_in_plane']
            # if flutter_derivatives_type is '2D_in_plane', then only aero damping is included (""in-axis""). If '2D_full' then also aero stiff is included (""in- and out-of-axis"")
            U_x = U_bar_Ls[:, 0]
            U_y = U_bar_Ls[:, 1]
            U_z = U_bar_Ls[:, 2]
            Cx = C_Ci_func(np.ones(beta_bar.shape)*(-np.pi/2), np.zeros(theta_bar.shape), aero_coef_method, n_aero_coef, coor_system='Ls')[0]  # beta_bar and theta_bar needed to calculate theta_yz inside the function. Then, betas = 0.
            A_delta_d_Ls_axial = -1 * 1/2 * rho * CS_width * np.einsum('n,vcn->nvc', np.abs(U_x), np.array(
                [[ 2*Cx, zeros, zeros, zeros, zeros, zeros],
                 [zeros, zeros, zeros, zeros, zeros, zeros],
                 [zeros, zeros, zeros, zeros, zeros, zeros],
                 [zeros, zeros, zeros, zeros, zeros, zeros],
                 [zeros, zeros, zeros, zeros, zeros, zeros],
                 [zeros, zeros, zeros, zeros, zeros, zeros]]), optimize=True)
            A_delta_d_Ls = A_delta_d_Ls + A_delta_d_Ls_axial
            if flutter_derivatives_type == '2D_full':
                A_delta_Ls_axial = 1/2 * rho * CS_width * np.einsum('n,vcn->nvc', np.abs(U_x), np.array(
                    [[zeros, zeros, zeros, zeros, -2*U_z*Cx, 2*U_y*Cx],
                     [zeros, zeros, zeros, zeros,     zeros,   U_x*Cx],
                     [zeros, zeros, zeros, zeros,   -U_x*Cx,    zeros],
                     [zeros, zeros, zeros, zeros,     zeros,    zeros],
                     [zeros, zeros, zeros, zeros,     zeros,    zeros],
                     [zeros, zeros, zeros, zeros,     zeros,    zeros]]), optimize=True)
                A_delta_Ls = A_delta_Ls + A_delta_Ls_axial
    A_delta_Gs = T_GsLs_6 @ A_delta_Ls @ T_LsGs_6
    A_delta_d_Gs = T_GsLs_6 @ A_delta_d_Ls @ T_LsGs_6
    Kse_QS = - np.einsum('n,nij->nij', g_node_L_3D, A_delta_Gs)  # quasi-static (frequency-independent)
    Cse_QS = - np.einsum('n,nij->nij', g_node_L_3D, A_delta_d_Gs)  # quasi-static (frequency-independent)
    Kse = np.repeat(Kse_QS[np.newaxis, :, :, :], len(f_array), axis=0)
    Cse = np.repeat(Cse_QS[np.newaxis, :, :, :], len(f_array), axis=0)
    return Kse, Cse

def Pb_func(g_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach, Chi_Ci='Chi function'):
    """
    Coefficient matrix of nodal buffeting forces with respect to the global structural XYZ-system, to be multiplied with
    wind turbulences in the global wind coordinates XuYvZw. Note: Pb * a = (T_GsLw * A) * a = T_GsLw * (A * a), and
    A(beta,theta) is set up so that (A * a) goes from wind speeds in global wind Gw to forces in local wind Lw?
    """
    A_bar = A_bar_func(U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach, Chi_Ci)
    g_node_num = len(g_node_coor)
    g_node_L_3D = g_node_L_3D_func(g_node_coor)

    # Transformation matrices.
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)  # to be used for the bridge girder elements
    T_GsLs = np.transpose(T_LsGs, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LrLs = T_LrLs_func(g_node_coor)  # to be used for the bridge girder elements
    T_LrLs_T = np.transpose(T_LrLs, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LrLwbar = T_LsLw_func(beta_bar, theta_bar, dim='3x3')
    T_GsLw = T_GsLs @ T_LrLs_T @ T_LrLwbar
    T_GsLw_6 = np.zeros((g_node_num, 6, 6))
    T_GsLw_6[:, :3, :3] = T_GsLw
    T_GsLw_6[:, 3:, 3:] = T_GsLw
    Pb = np.einsum('n,nvc->nvc', g_node_L_3D, T_GsLw_6 @ A_bar)
    return Pb

def Fsw_func(g_node_coor, p_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef):
    """Get static wind forces in a full 1D global vector, shape:(dof_all)"""
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)

    B_diag = np.diag((CS_width, CS_width, CS_width, CS_width ** 2, CS_width ** 2, CS_width ** 2))
    C_Ci_bar = C_Ci_func(beta_bar, theta_bar, aero_coef_method, n_aero_coef, coor_system='Lw')
    f_sw_Lwbar = 0.5 * rho * np.einsum('n,ij,jn->ni', U_bar ** 2, B_diag, C_Ci_bar)  # static wind only
    g_node_L_3D = g_node_L_3D_func(g_node_coor)

    T_LrLwbar_6 = T_LsLw_func(beta_bar, theta_bar, dim='6x6')
    T_LsGs_6 = T_LsGs_6g_func(g_node_coor, alpha)
    T_GsLs_6 = np.transpose(T_LsGs_6, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))

    f_sw_Ls = np.einsum('nij,nj->ni', T_LrLwbar_6, f_sw_Lwbar)
    f_sw_Gs = np.einsum('nij,nj->ni', T_GsLs_6, f_sw_Ls)  # Global. Also, reshaping from nit to tni.
    F_sw_Gs = np.einsum('n,ni->ni', g_node_L_3D, f_sw_Gs)  # Local.
    F_sw_Gs = np.reshape(F_sw_Gs, (g_node_num * 6))  # reshaping from 'tnd' (3D) to 't(n*d)' (2D) so it resembles the stiffness matrix shape of (n*d)*(n*d)
    F_sw_Gs = np.concatenate((F_sw_Gs, np.zeros(p_node_num * 6)))  # adding Fb = 0 to all remaining dof at the pontoon nodes
    return F_sw_Gs

def Fad_or_Fb_all_t_Taylor_hyp_func(g_node_coor, p_node_coor, alpha, beta_0, theta_0, beta_bar, theta_bar, windspeed, aero_coef_method, n_aero_coef, skew_approach, which_to_get):
    """Get buffeting forces (without mean wind) in a full 2D global matrix shape:(time, dof_all),
    with Taylor's hyphotesis on the linearized C_Ci coefficients. SE excited forces are not included here as they
    are thus eventually included in K and C as Kse and Cse."""
    windspeed_u = windspeed[1, :, :]
    windspeed_v = windspeed[2, :, :]
    windspeed_w = windspeed[3, :, :]
    time_array_length = len(windspeed_u[0])  # length of U+u at first node
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)
    # Nodal buffeting force coefficients in global structural XYZ-system
    Pb = Pb_func(g_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach)
    # Wind speeds and Buffeting forces
    a = np.array([windspeed_u, windspeed_v, windspeed_w])  # shape: (3, g_node_num, len(timepoints))
    Fb = np.einsum('ndi,int->tnd', Pb, a)  # (N). Global buffeting force vector. See Paper from LD Zhu, eq. (24)
    Fb = np.reshape(Fb, (time_array_length, g_node_num * 6))  # reshaping from 'tnd' (3D) to 't(n*d)' (2D) so it resembles the stiffness matrix shape of (n*d)*(n*d)
    Fb = np.concatenate((Fb, np.zeros((time_array_length, p_node_num * 6))), axis=1)  # adding Fb = 0 to all remaining dof at the pontoon g_nodes
    if which_to_get == 'Fb':
        return Fb
    elif which_to_get == 'Fad':
        F_sw_Gs = Fsw_func(g_node_coor, p_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef)  # static wind forces
        Fad_Gs = Fb + F_sw_Gs[np.newaxis, :]  # buffeting only forces. New time dimension for static wind.
        return Fad_Gs
    else:
        raise ValueError("parameter 'which_to_get' needs to be 'Fad' or 'Fb'")

def Fad_or_Fb_all_t_C_Ci_NL_no_SE_func(g_node_coor, p_node_coor, alpha, beta_0, theta_0, beta_bar, theta_bar, windspeed, aero_coef_method, n_aero_coef, skew_approach, which_to_get):
    """Get aerodynamic or buffeting forces in a full 2D global matrix, shape:(time, dof_all),
    when C_Ci coefficients are obtained at each time step with instantaneous betas and thetas due to turbulence, without self excited forces.
    Fad (aerodynamic forces) = Fb (buffeting) + F_sw (static wind). See L.D.Zhu eq. (4-44) and (4-49)
    which_to_get -- 'Fad' or 'Fb'
    """
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)

    U_u, u, v, w = windspeed  # . shape of each: (n_nodes, time)
    U_tilde_Gw = np.array([U_u, v, w])

    time_array_length = len(U_u[0])  # length of U+u at first node
    T_LsGs_3 = T_LsGs_3g_func(g_node_coor, alpha)
    T_GsLs_3 = np.transpose(T_LsGs_3, axes=(0,2,1))
    T_GsLs_6 = M_3x3_to_M_6x6(T_GsLs_3)

    T_LsGw_3 = np.einsum('nij,jk->nik', T_LsGs_3, T_GsGw_func(beta_0,theta_0,dim='3x3'))
    U_tilde_x, U_tilde_y, U_tilde_z = np.einsum('nij,jnt->int', T_LsGw_3, U_tilde_Gw)
    U_tilde_xy = np.sqrt(U_tilde_x**2 + U_tilde_y**2)
    U_tilde = np.sqrt(U_tilde_x**2 + U_tilde_y**2 + U_tilde_z**2)

    g_node_L_3D = g_node_L_3D_func(g_node_coor)
    B_diag = np.diag((CS_width, CS_width, CS_width, CS_width ** 2, CS_width ** 2, CS_width ** 2))

    beta_tilde_flat = np.ndarray.flatten(-np.arccos(U_tilde_y/U_tilde_xy) * np.sign(U_tilde_x))
    theta_tilde_flat = np.ndarray.flatten(np.arcsin(U_tilde_z/U_tilde))

    if skew_approach == '3D':
        C_Ci_tilde_Ls_flat = C_Ci_func(beta_tilde_flat, theta_tilde_flat, aero_coef_method, n_aero_coef, coor_system='Ls')
        C_Ci_tilde_Ls = np.reshape(C_Ci_tilde_Ls_flat, (6, g_node_num, time_array_length))
        f_ad_Ls = 0.5 * rho * np.einsum('nt,ij,jnt->nit', U_tilde**2, B_diag, C_Ci_tilde_Ls, optimize=True)  # in instantaneous local Lw_tilde coordinates

    if '2D' in skew_approach:
        theta_tilde_yz_flat = theta_yz_bar_func(beta_tilde_flat, theta_tilde_flat)
        C_Ci_tilde_beta_0_Ls_flat = C_Ci_func(np.zeros(len(beta_tilde_flat)), theta_tilde_yz_flat, aero_coef_method, n_aero_coef, coor_system='Ls')
        C_Ci_tilde_beta_0_Ls = np.reshape(C_Ci_tilde_beta_0_Ls_flat, (6, g_node_num, time_array_length))
        U_tilde_yz = np.sqrt(U_tilde_y**2 + U_tilde_z**2)
        f_ad_Ls = 0.5 * rho * np.einsum('nt,ij,jnt->nit', U_tilde_yz**2, B_diag, C_Ci_tilde_beta_0_Ls, optimize=True)  # in instantaneous local Lw_tilde coordinates

    if '1D' in skew_approach:
        C_Cx_90 = C_Ci_func(np.array([rad(-90)]), np.array([0]), aero_coef_method, n_aero_coef, coor_system='Ls')[0]
        f_ad_x = 0.5 * rho * U_tilde_x**2 * CS_width * C_Cx_90  # in instantaneous local Lw_tilde coordinates
        f_ad_Ls[:,0,:] += copy.deepcopy(f_ad_x)


    F_ad_Ls = np.einsum('n,nit->nit', g_node_L_3D, f_ad_Ls)  # Local.
    F_ad_Gs = np.einsum('nij,njt->tni', T_GsLs_6, F_ad_Ls, optimize=True)  # Global. Also, reshaping from nit to tni.
    F_ad_Gs = np.reshape(F_ad_Gs, (time_array_length, g_node_num * 6))  # reshaping from 'tnd' (3D) to 't(n*d)' (2D) so it resembles the stiffness matrix shape of (n*d)*(n*d)
    F_ad_Gs = np.concatenate((F_ad_Gs, np.zeros((time_array_length, p_node_num * 6))), axis=1)  # adding Fb = 0 to all remaining dof at the pontoon nodes
    if which_to_get == 'Fad':
        return F_ad_Gs
    elif which_to_get == 'Fb':
        F_sw_Gs = Fsw_func(g_node_coor, p_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef)  # static wind forces
        Fb_Gs = F_ad_Gs - F_sw_Gs[np.newaxis,:]  # buffeting only forces. New time dimension for static wind.
        return Fb_Gs
    else:
        raise ValueError("parameter 'which_to_get' needs to be 'Fad' or 'Fb'")


def Fad_one_t_C_Ci_NL_with_SE(g_node_coor, p_node_coor, alpha, beta_0, theta_0, windspeed_i, v_new,
                              C_C0_func, C_C1_func, C_C2_func, C_C3_func, C_C4_func, C_C5_func):
    """
    windspeed_i: windspeed at time instant i == windspeed[:,:,i]
    v_new: structural velocities, from previous time step
    """

    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)
    g_node_L_3D = g_node_L_3D_func(g_node_coor)
    B_diag = np.diag((CS_width, CS_width, CS_width, CS_width ** 2, CS_width ** 2, CS_width ** 2))

    # Windspeeds without the time dimension!
    U_and_u = windspeed_i[0, :]  # U+u
    windspeed_v = windspeed_i[2, :] # todo: NO NEED FOR THE WHOLE WINDSPEED TIME, ONLY AT TIME i
    windspeed_w = windspeed_i[3, :]

    # Variables, calculated every time step
    T_LsGs_3 = T_LsGs_3g_func(g_node_coor, alpha)
    T_LsGs_6 = T_LsGs_6g_func(g_node_coor, alpha)
    T_GsLs_6 = np.transpose(T_LsGs_6, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_GsGw = T_GsGw_func(beta_0, theta_0)
    T_LrGw = T_LsGs_3 @ T_GsGw
    t11, t12, t13, t21, t22, t23, t31, t32, t33 = T_LrGw[:,0,0], T_LrGw[:,0,1], T_LrGw[:,0,2], \
                                                  T_LrGw[:,1,0], T_LrGw[:,1,1], T_LrGw[:,1,2], \
                                                  T_LrGw[:,2,0], T_LrGw[:,2,1], T_LrGw[:,2,2]
    T_LsGs_full_2D_node_matrix = T_LsGs_full_2D_node_matrix_func(g_node_coor, p_node_coor, alpha)
    # Total relative windspeed vector, in local structural Ls (same as Lr) coordinates. See eq. (4-36) from L.D.Zhu thesis. shape: (3,n_nodes,time)
    v_Ls = T_LsGs_full_2D_node_matrix @ v_new[-1]  # Initial structural speeds
    V_q = t11 * U_and_u + t12 * windspeed_v + t13 * windspeed_w
    V_p = t21 * U_and_u + t22 * windspeed_v + t23 * windspeed_w
    V_h = t31 * U_and_u + t32 * windspeed_v + t33 * windspeed_w
    V_rel_q = V_q - v_Ls[0:g_node_num * 6:6]  # including structural motion. shape:(g_node_num).
    V_rel_p = V_p - v_Ls[1:g_node_num * 6:6]
    V_rel_h = V_h - v_Ls[2:g_node_num * 6:6]
    # Projection of V_Lr in local bridge xy plane (same as qp in L.D.Zhu). See L.D.Zhu eq. (4-44)
    V_rel_qp = np.sqrt(V_rel_q ** 2 + V_rel_p ** 2)  # SRSS of Vq and Vp
    V_rel_tot = np.sqrt(V_rel_q ** 2 + V_rel_p ** 2 + V_rel_h ** 2)
    theta_tilde = np.arccos(V_rel_qp / V_rel_tot)  * np.sign(V_h)  # positive if V_h is positive!
    beta_tilde = np.arccos(V_rel_p / V_rel_qp) * -np.sign(V_q)  # negative if V_q is positive!
    T_LrLwtilde_6 = T_LsLw_func(beta_tilde, theta_tilde, dim='6x6')  # shape:(g,6,6)
    C_Ci_tilde = np.array([C_C0_func.ev(beta_tilde, theta_tilde),  # .ev means "evaluate" the interpolation, at given points
                           C_C1_func.ev(beta_tilde, theta_tilde),
                           C_C2_func.ev(beta_tilde, theta_tilde),
                           C_C3_func.ev(beta_tilde, theta_tilde),
                           C_C4_func.ev(beta_tilde, theta_tilde),
                           C_C5_func.ev(beta_tilde, theta_tilde)])
    F_ad_tilde = 0.5 * rho * np.einsum('n,n,ij,jn->ni', g_node_L_3D, V_rel_tot ** 2, B_diag, C_Ci_tilde, optimize=True)  # in instantaneous local Lw_tilde coordinates
    F_ad_tilde_Ls = np.einsum('nij,nj->ni', T_LrLwtilde_6, F_ad_tilde)  # Local structural
    F_ad_tilde_Gs = np.einsum('nij,nj->ni', T_GsLs_6, F_ad_tilde_Ls)  # Global structural
    F_ad_tilde_Gs = np.reshape(F_ad_tilde_Gs, (g_node_num * 6))  # reshaping from 'nd' (2D) to '(n*d)' (1D) so it resembles the stiffness matrix shape of (n*d)*(n*d)
    F_ad_tilde_Gs = np.concatenate((F_ad_tilde_Gs, np.zeros((p_node_num * 6))), axis=0)  # adding Fb = 0 to all remaining dof at the pontoon g_nodes
    return F_ad_tilde_Gs

########################################################################################################################
# Frequency Domain Buffeting Analysis:
########################################################################################################################
def buffeting_FD_func(include_sw, include_KG, aero_coef_method, n_aero_coef, skew_approach, include_SE, flutter_derivatives_type, n_modes, f_min, f_max, n_freq, g_node_coor, p_node_coor,
                      Ii_simplified, beta_DB, R_loc, D_loc, cospec_type, include_modal_coupling, include_SE_in_modal, f_array_type, make_M_C_freq_dep, dtype_in_response_spectra,
                      Nw_idx, Nw_or_equiv_Hw, generate_spectra_for_discretization=False):
    theta_0 = 0
    print(f'{Nw_or_equiv_Hw} case index: {Nw_idx}') if Nw_idx is not None else 'Running original homogeneous wind only.'
    if Nw_idx is not None:  # Inomogeneous wind:
        with open(fr'intermediate_results\\static_wind\\Nw_dict_{Nw_idx}.json', 'r', encoding='utf-8') as f:
            Nw_1_case = json.load(f)

    print('beta_DB (deg) = '+str(np.round(deg(beta_DB), 1)))
    start_time_1 = time.time()
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    if f_array_type == 'equal_width_bins':
        f_array = np.linspace(f_min, f_max, n_freq)
    elif f_array_type == 'equal_energy_bins':
        print("""When using f_array_type = 'equal_energy_bins' make sure f_array.npy and max_S_delta_local.npy are both representative of the response and are very well discretized""")
        f_array = discretize_S_delta_local_by_equal_energies(f_array=np.load(r"intermediate_results\f_array.npy"), max_S_delta_local=np.load(r"intermediate_results\max_S_delta_local.npy") ,n_freq_desired=n_freq, plot=False)

    n_freq = len(f_array)
    w_array = f_array * 2 * np.pi

    R_loc = copy.deepcopy(R_loc)  # Otherwise, it would increase during repetitive calls of this function
    D_loc = copy.deepcopy(D_loc)  # Otherwise, it would increase during repetitive calls of this function
    girder_N = copy.deepcopy(R_loc[:g_elem_num, 0])  # girder axial forces
    c_N = copy.deepcopy(R_loc[g_elem_num:, 0])  # columns axial forces
    alpha = copy.deepcopy(D_loc[:g_node_num, 3])  # girder nodes torsional rotations

    loop_frequencies = True  # True or False. True = can take slightly more time, but uses less memory. Allows for more discretized simulations
    plot_std_along_girder = False  # 2D plots of std of response in x, y, z, xx, along the bridge girder

    delta_w_array = delta_array_func(w_array)

    if include_sw:  # including static wind
        if Nw_idx is None:  # Homogeneous wind:
            from static_loads import static_wind_func, R_loc_func
            U_bar = U_bar_func(g_node_coor)  # Homogeneous wind only depends on g_node_coor_z...
            # Displacements
            g_node_coor_sw, p_node_coor_sw, D_glob_sw = static_wind_func(g_node_coor, p_node_coor, alpha, U_bar, beta_DB, theta_0, aero_coef_method, n_aero_coef, skew_approach)
            D_loc_sw = mat_Ls_node_Gs_node_all_func(D_glob_sw, g_node_coor, p_node_coor, alpha)
            # Internal forces
            R_loc_sw = R_loc_func(D_glob_sw, g_node_coor, p_node_coor, alpha)  # orig. coord. + displacem. used to calc. R.
        else:  # Inomogeneous wind:
            # Displacements
            g_node_coor_sw, p_node_coor_sw = np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_g_node_coor']), np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_p_node_coor'])
            D_loc_sw = np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_D_loc'])
            # Internal forces
            R_loc_sw = np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_R_loc'])
        # Extracting alpha and axial forces
        alpha_sw = copy.deepcopy(D_loc_sw[:g_node_num, 3])  # local nodal torsional rotation.
        girder_N_sw = copy.deepcopy(R_loc_sw[:g_elem_num, 0])  # local girder element axial force. Positive = compression!
        c_N_sw = copy.deepcopy(R_loc_sw[g_elem_num:, 0])  # local column element axial force Positive = compression!
        # Updating structure.
        g_node_coor, p_node_coor = copy.deepcopy(g_node_coor_sw), copy.deepcopy(p_node_coor_sw)
        R_loc += copy.deepcopy(R_loc_sw)  # element local forces
        D_loc += copy.deepcopy(D_loc_sw)  # nodal global displacements. Includes the alphas.
        girder_N += copy.deepcopy(girder_N_sw)
        c_N += copy.deepcopy(c_N_sw)
        alpha += copy.deepcopy(alpha_sw)

    # Getting the key mean wind features
    if Nw_idx is None:
        U_bar = U_bar_func(g_node_coor)
        beta_0 = beta_0_func(beta_DB)
        beta_bar, theta_bar = beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha)
    else:
        U_bar = np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_U_bar'])
        beta_0, theta_0 = np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_beta_0']), np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_theta_0'])
        beta_bar, theta_bar = np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_beta_bar']), np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_theta_bar'])

    # Transformation matrices.
    T_LsGs_3 = T_LsGs_3g_func(g_node_coor, alpha)  # to be used for the bridge girder elements
    T_LsGs_6 = T_LsGs_6g_func(g_node_coor, alpha)
    T_GsLs_3 = np.transpose(T_LsGs_3, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_GsLs_6 = np.transpose(T_LsGs_6, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))

    T_LrLs = T_LrLs_func(g_node_coor)  # to be used for the bridge girder elements
    T_LrLs_T = np.transpose(T_LrLs, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    # T_LrGw = T_LrLs @ T_LsGs_3 @ T_GsGw  # matrix multiplication
    T_LrLwbar = T_LsLw_func(beta_bar, theta_bar, dim='3x3')
    T_LwbarLr = np.transpose(T_LrLwbar, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LwbarLr_6 = np.zeros((g_node_num, 6, 6))
    T_LwbarLr_6[:, :3, :3] = T_LwbarLr
    T_LwbarLr_6[:, 3:, 3:] = T_LwbarLr
    T_GsLw = T_GsLs_3 @ T_LrLs_T @ T_LrLwbar
    T_GsLw_6 = np.zeros((g_node_num, 6, 6))
    T_GsLw_6[:, :3, :3] = T_GsLw
    T_GsLw_6[:, 3:, 3:] = T_GsLw

    # reduc_coef_sector = reduc_coef_sector_func(beta_DB) # NOT BEING USED

    # ------------------------------------------------------------------------------------------------------------------
    # MODAL PROPERTIES
    # ------------------------------------------------------------------------------------------------------------------
    # Importing mass and stiffness matrices. Units: (N):
    M = mass_matrix_func(g_node_coor, p_node_coor, alpha, w_array=None, make_freq_dep=False)  # frequency-independent, to be able to perform modal analysis
    K = stiff_matrix_func(g_node_coor, p_node_coor, alpha)

    if include_KG:
        KG = geom_stiff_matrix_func(g_node_coor, p_node_coor, girder_N, c_N, alpha)
        K = copy.deepcopy(K) - KG  # geometric stiffness effects are now included in K

    start_time_KseCse = time.time()

    if include_SE_in_modal == True and include_SE == False:
        raise Exception('You cannot have include_SE_in_modal == True and include_SE == False')

    if include_SE:  # Gets matrices of Kse and Cse in physical space, with same size as g_node_num
        Kse, Cse = Kse_Cse_func(g_node_coor, U_bar, beta_bar, theta_bar, alpha, f_array, flutter_derivatives_type, aero_coef_method, n_aero_coef, skew_approach)

    if include_SE_in_modal:  # Calculates full matrices of Kse and Cse in physical space (including pontoon nodes)
        p_node_num = len(p_node_coor)
        if not np.allclose(Kse[0], Kse[-1]) or not np.allclose(Cse[0], Cse[-1]):
            raise Exception('You have frequency-dependent flutter derivatives, but you neglect that in the modal analysis (in the following lines of code, and further below)')  # you can eventually delete this, but conciouslly.

        Kse_0 = Kse[0]  # removing the frequency dimension.
        Cse_0 = Cse[0]  # removing the frequency dimension.
        # Kse_full_mat = np.zeros((n_freq, (g_node_num + p_node_num) * 6, (g_node_num + p_node_num) * 6))
        # Cse_full_mat = np.zeros((n_freq, (g_node_num + p_node_num) * 6, (g_node_num + p_node_num) * 6))
        Kse_full_mat_0 = np.zeros(( (g_node_num + p_node_num) * 6, (g_node_num + p_node_num) * 6))
        Cse_full_mat_0 = np.zeros(( (g_node_num + p_node_num) * 6, (g_node_num + p_node_num) * 6))
        for n in range(g_node_num):
            # Kse_full_mat[:,n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Kse[:,n,:,:]
            # Cse_full_mat[:,n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Cse[:,n,:,:]
            Kse_full_mat_0[n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Kse_0[n,:,:]
            Cse_full_mat_0[n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Cse_0[n,:,:]
        # Modal analysis (Strommen PDF p.263 says Kse can probably be neglected to which_to_get mode shapes, freq, M_tilde & K_tilde)
        K = copy.deepcopy(K) + Kse_full_mat_0

    M_tilde, K_tilde, w_eigen, shapes = modal_analysis_func(M, K)
    M_tilde = M_tilde[:n_modes,:n_modes]  # Trimming.
    K_tilde = K_tilde[:n_modes, :n_modes]  # Trimming.

    # Mode shapes on the bridge girder only (pontoon nodes are disregarded from the modal buffeting forces!).
    g_shapes = np.moveaxis(np.array([shapes[:, i : g_node_num*6 : 6] for i in range(6)]), 0, -1)  # trimming nodes, reshaping.
    w_eigen, g_shapes, shapes = w_eigen[:n_modes], g_shapes[:n_modes], shapes[:n_modes]  # trimming modes.
    damping_alpha, damping_beta = rayleigh_coefficients_func(damping_ratio, damping_Ti, damping_Tj)

    print('First eigen period (s) = ' + str(np.round(2*np.pi/w_eigen[0], 2)))

    if make_M_C_freq_dep:
        M = mass_matrix_func(g_node_coor, p_node_coor, alpha, w_array=w_array, make_freq_dep=make_M_C_freq_dep)  # create new M which is now freq-dependent
        M_tilde = np.einsum('Mm,wmn,Nn->wMN', shapes, M, shapes, optimize=True)
        M_tilde = M_tilde[...,:n_modes, :n_modes]  # Trimming.

    # Hydrodynamic added damping
    C_added = added_damping_global_matrix_func(w_array=w_array, make_freq_dep=make_M_C_freq_dep)  # Shape either: (n_FEM_dof, n_FEM_dof) OR (n_freq, n_FEM_dof, n_FEM_dof)
    C_added_tilde = np.einsum('Mm,...mn,Nn->...MN', shapes, C_added, shapes, optimize=True)  # '...' is for wheather the 'w' dimension is included or not (depending on make_M_C_freq_dep)
    C_added_tilde = C_added_tilde[...,:n_modes,:n_modes]  # trimming. Shape either: (n_modes, n_modes) OR (n_freq, n_modes, n_modes)

    if include_SE:  # include_SE has to do with what goes to the H_tilde function (not into the modal analysis).
        # for speed purposes, Kse_tilde is calculated separatelly, using just g_shapes instead of shapes (full size)
        Kse_tilde = np.einsum('Mmi,wmij,Nmj->wMN', g_shapes, Kse, g_shapes, optimize=True)  # confirmed correct.
        Cse_tilde = np.einsum('Mmi,wmij,Nmj->wMN', g_shapes, Cse, g_shapes, optimize=True)  # confirmed correct.
        assert np.allclose(Kse_tilde[0], Kse_tilde[-1]), "flutter derivatives should be freq-independent for this to work with possible include_SE_in_modal"
        if not include_SE_in_modal: # and if Kse hasn't been added yet
            K_tot_tilde = K_tilde + Kse_tilde  # broadcasting into freq dimension
        else:  # if Kse was already addded
            K_tot_tilde = copy.deepcopy(K_tilde)
        C_tot_tilde = damping_alpha * M_tilde + damping_beta * K_tot_tilde + Cse_tilde + C_added_tilde  # broadcasting into freq dimension if M_tilde and C_added_tilde are not freq-dependent
    else:  # then K_tot_tilde and C_tot_tilde do not have the frequency dimension.
        K_tot_tilde = copy.deepcopy(K_tilde)
        C_tot_tilde = damping_alpha * M_tilde + damping_beta * K_tot_tilde + C_added_tilde  # broadcasting into freq dimension if make_M_C_freq_dep. Shape either: (n_modes, n_modes) OR (n_freq, n_modes, n_modes)
    # At this stage, M_tilde, C_tot_tilde and K_tot_tilde can either be or not frequency-depenent

    # # OLD Confirmation of Kse_tilde and Cse_tilde:
    # g_shapes_vector = np.reshape(g_shapes, (n_modes, g_node_num*6))
    # Kse_full_mat_girder_only = np.zeros((n_freq, g_node_num * 6, g_node_num * 6))
    # Cse_full_mat_girder_only = np.zeros((n_freq, g_node_num * 6, g_node_num * 6))
    # Kse_tilde_confirm = np.zeros((n_freq, n_modes, n_modes))
    # Cse_tilde_confirm = np.zeros((n_freq, n_modes, n_modes))
    # for w in range(n_freq):
    #     for n in range(g_node_num):
    #         Kse_full_mat_girder_only[w, n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Kse[w, n]
    #         Cse_full_mat_girder_only[w, n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Cse[w, n]
    #     Kse_tilde_confirm[w] = g_shapes_vector @ Kse_full_mat_girder_only[w] @ np.transpose(g_shapes_vector)
    #     Cse_tilde_confirm[w] = g_shapes_vector @ Cse_full_mat_girder_only[w] @ np.transpose(g_shapes_vector)
    # print(np.max(abs(Kse_tilde - Kse_tilde_confirm)))
    # print(np.max(abs(Cse_tilde - Cse_tilde_confirm)))
    #
    # # Semi-confirmation of Kse_tilde:
    # Kse_mn = np.zeros((n_freq, g_node_num, g_node_num,6,6))
    # for n in range(g_node_num):
    #     Kse_mn[:,n,n,::] = Kse[:,n,:,:]
    # Kse_tilde_confirm_2 = np.einsum('Mmi,wmnij,Nnj->wMN', g_shapes, Kse_mn, g_shapes, optimize=True)
    # print(np.max(abs(Kse_tilde - Kse_tilde_confirm_2)))
    # ----------------------------------------------------------------------------------------------------------------------
    stop_time_Kse_Cse = np.round((time.time() - start_time_KseCse))
    stop_time_1 = np.round((time.time() - start_time_1))
    start_time_2 = time.time()

    # Frequency Response Transfer function (note: np.reciprocal is NOT the invert of a matrix)
    if include_modal_coupling:  # M, C and K are matrices with off-diagonals
        # No matter the frequency-dependencies, the result will ALWAYS have shape 'wMN', due to the ellipsis: '...'
        H_tilde = np.linalg.inv(np.einsum('...,...MN->...MN', -w_array**2, M_tilde) + np.einsum('...,...MN->...MN', 1j*w_array, C_tot_tilde, optimize=True) + K_tot_tilde)
        # # Confirmation. The following are all equivalent:
        # test1 = np.einsum('w,wMN->wMN', np.arange(8), np.repeat(np.arange(9).reshape(3, 3)[np.newaxis, :, :], 8, axis=0))
        # test2 = np.einsum('w,MN->wMN', np.arange(8), np.arange(9).reshape(3, 3))
        # test3 = np.einsum('...,...MN->...MN', np.arange(8), np.repeat(np.arange(9).reshape(3, 3)[np.newaxis, :, :], 8, axis=0))
        # test4 = np.einsum('...,...MN->...MN', np.arange(8), np.arange(9).reshape(3, 3))
    else:
        # No matter the frequency-dependencies, the result will ALWAYS have shape 'wMN', due to the ellipsis: '...'. The intermediate shape is wM, before np.diag converts again to wMN
        H_tilde = np.linalg.inv(np.array([np.diag((np.einsum('...,M->...M', -w_array**2, np.diagonal(M_tilde,axis1=-2,axis2=-1)) + np.einsum('...,...M->...M', 1j*w_array, np.diagonal(C_tot_tilde,axis1=-2,axis2=-1)) + np.diagonal(K_tot_tilde,axis1=-2,axis2=-1))[f,:]) for f in range(n_freq)]))

    Pb = Pb_func(g_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach)

    # Co-spectrum
    if Nw_idx is None:
        S_aa_hertz = S_aa_func(g_node_coor, beta_DB, f_array, Ii_simplified, cospec_type=2)
    else:  # Inhomogeneous S_aa needs to be calculated in real time!! It is way too large to be stored for all Nw cases...
        if Nw_or_equiv_Hw == 'Nw':
            Nw_Ii = np.array(Nw_1_case[f'{Nw_or_equiv_Hw}_Ii'])
        elif Nw_or_equiv_Hw == 'Hw':
            Nw_Ii = Ii_func(g_node_coor, beta_DB, Ii_simplified)
        S_aa_hertz = Nw_S_aa(g_node_coor, beta_0, theta_0, f_array, U_bar, Nw_Ii, cospec_type=2)

    S_aa_radians = S_aa_hertz / (2*np.pi)  # not intuitive! S(f)*delta_f = S(w)*delta_w. See eq. (2.68) from Strommen.

    # Buffeting cross-spectral loads and displacements. S_aa is collapsed from shape wmncc to wmnc since off-diagonals are 0 (cross-spectrum between turbulence components)
    if loop_frequencies:  # Consumes less memory. Equally fast.
        Sb_FF = np.zeros((n_freq, g_node_num, g_node_num, 6, 6))
        Sb_FF_tilde = np.zeros((n_freq, n_modes, n_modes))
        S_etaeta = np.zeros((n_freq, n_modes, n_modes), dtype=dtype_in_response_spectra)
        S_deltadelta = np.zeros((n_freq, g_node_num, g_node_num, 6, 6), dtype=dtype_in_response_spectra)
        S_deltadelta_local = np.zeros((n_freq, g_node_num, g_node_num, 6, 6), dtype=dtype_in_response_spectra)

        # # # CONFIRMATION OF Sb_FF AND Sb_FF_tilde. GIVES SAME RESULTS
        # # Performing loops and pure matrix multiplications exactly as in Zhu, just for confirmation:
        # Sb_FF_confirm = np.zeros((n_freq, 6*g_node_num, 6*g_node_num))
        # Pb_confirm = np.zeros((6*g_node_num, 3*g_node_num))
        # S_aa_radians_confirm = np.zeros((n_freq, 3*g_node_num, 3*g_node_num))
        # g_shapes_confirm = np.zeros((6*g_node_num, n_modes))
        # Sb_FF_tilde_confirm = np.zeros((n_freq, n_modes, n_modes))
        # for i in range(g_node_num):
        #     for j in range(g_node_num):
        #         for w in range(n_freq):
        #             S_aa_radians_confirm[w,3*i:3*i+3,3*j:3*j+3] = np.diag(S_aa_radians[w,i,j])
        #     Pb_confirm[6*i:6*i+6,3*i:3*i+3] = Pb[i]
        #     g_shapes_confirm[6*i:6*i+6] = g_shapes[:,i,:].T
        # for w in range(n_freq):
        #     Sb_FF_confirm[w,:,:] = np.conjugate(Pb_confirm) @ S_aa_radians_confirm[w] @ np.transpose(Pb_confirm)
        #     Sb_FF_tilde_confirm[w] = g_shapes_confirm.T @ Sb_FF_confirm[w] @ g_shapes_confirm
        # Sb_FF_confirm_same_shape = np.moveaxis(np.moveaxis(np.reshape(Sb_FF_confirm, (n_freq, g_node_num, 6, g_node_num, 6)), 2, -1), 3, -1)
        # ----------------------------------------------------------------------------------------------------------------------

        for w in range(n_freq):
            Sb_FF[w] = np.einsum('muc,mnc,nvc-> mnuv', np.conjugate(Pb), S_aa_radians[w], Pb, optimize=True)
            Sb_FF_tilde[w] = np.einsum('Mmu,mnuv,Nnv->MN', g_shapes, Sb_FF[w], g_shapes, optimize=True)
            # Modal response spectrum
            if 'complex' in dtype_in_response_spectra:
                S_etaeta[w] = np.einsum('MN,NO,OP->MP', np.conjugate(H_tilde[w]), Sb_FF_tilde[w], np.transpose(H_tilde[w]), optimize=True)
            elif 'float' in dtype_in_response_spectra:
                S_etaeta[w] = np.real(np.einsum('MN,NO,OP->MP', np.conjugate(H_tilde[w]), Sb_FF_tilde[w], np.transpose(H_tilde[w]), optimize=True))
            # # Nodal response spectrum
            S_deltadelta[w] = np.einsum('Mmu,MN,Nnv->mnuv', g_shapes, S_etaeta[w], g_shapes, optimize=True)  # 'Mmv,Mf,Mmv->mvf'
            # Response in local coordinate system x,y,z:
            S_deltadelta_local[w] = np.einsum('miu, mnuv, nvj->mnij', T_LsGs_6, S_deltadelta[w], T_GsLs_6, optimize=True)  # optimization accelerates process 4.5 times
    else: # This was confirmed to be equivalent.
        raise NotImplementedError("Forget this method, not useful")
        # Sb_FF = np.einsum('muc,wmnc,nvc-> wmnuv', np.conjugate(Pb), S_aa_radians, Pb, optimize=True)
        # Sb_FF_tilde = np.einsum('Mmu,wmnuv,Nnv->wMN', g_shapes, Sb_FF, g_shapes, optimize=True)
        # # Modal response spectrum
        # S_etaeta = np.einsum('wMN,wNO,wOP->wMP', np.conjugate(H_tilde), Sb_FF_tilde, np.moveaxis(H_tilde, 1, 2), optimize=True)
        # # # Nodal response spectrum
        # S_deltadelta = np.einsum('Mmu,wMN,Nnv->wmnuv', g_shapes, S_etaeta, g_shapes, optimize=True)  # 'Mmv,Mf,Mmv->mvf'
        # # Response in local coordinate system x,y,z:
        # S_deltadelta_local = np.einsum('miu, wmnuv, nvj->wmnij', T_LsGs_6, S_deltadelta, T_GsLs_6, optimize=True)  # optimization accelerates process 4.5 times

    # # Response in global coordinates.
    # S_delta = np.diagonal(S_deltadelta, axis1=3, axis2=4)  # extracting the diagonal only (no interaction between dof??)
    # S_delta = np.moveaxis(np.diagonal(S_delta, axis1=1, axis2=2), 1, -1)   # extracting the diagonal only (no interaction between g_nodes??). Apparently np.diagonal changes order of the dimensions so np.moveaxis is needed.
    # # Root of sum of squares (RSS) of the 3 first components (displacement components, not rotations) of S_delta. Extracting the diagonal terms only
    # S_delta_norm = np.sqrt(np.sum(np.square(S_delta[:,:,:3]), axis=-1))
    # # std_delta: formulation with just diagonal of S_deltadelta
    # std_delta = np.sqrt(np.einsum('wmv,w->vm', np.real(S_delta), delta_w_array))

    # Response in local coordinates.
    S_delta_local = np.diagonal(S_deltadelta_local, axis1=3, axis2=4)   # extracting the diagonal only (no interaction between dof??)
    S_delta_local = np.moveaxis(np.diagonal(S_delta_local, axis1=1, axis2=2), 1, -1)   # extracting the diagonal only (no interaction between g_nodes??). Apparently np.diagonal changes order of the dimensions so np.moveaxis is needed.
    # Root of sum of squares (RSS) of the 3 first components (displacement components, not rotations) of S_delta. Extracting the diagonal terms only
    S_delta_norm_local = np.sqrt(np.sum(np.square(S_delta_local[:,:,:3]), axis=-1))  #
    # std_delta: formulation with just diagonal of S_deltadelta

    std_delta_local = np.sqrt(np.einsum('wmv,w->vm', np.real(S_delta_local), delta_w_array))  # np.real(S_delta) is confirmed to be good since imag part is virtually 0!

    if generate_spectra_for_discretization:
        max_S_delta_local = np.max(np.real(S_delta_local), axis=1)  # Maximum along bridge girder. Shape (n_freq, 6)
        np.save(r"intermediate_results\max_S_delta_local.npy", max_S_delta_local)
        np.save(r"intermediate_results\f_array.npy", f_array)

    # CONFIRMING THAT THE GLOBAL CAN BE AGAIN RETRIEVED FROM THE LOCAL. (backG - back to Global):
    # S_deltadelta_backG = np.einsum('miu, wmnuv, nvj->wmnij', T_GsLs_6, S_deltadelta_local, T_LsGs_6)
    # S_delta_backG = np.diagonal(S_deltadelta_backG, axis1=3, axis2=4)  # extracting the diagonal only (no interaction between g_nodes??). Apparently np.diagonal changes order of the dimensions so np.moveaxis is needed.
    # S_delta_backG = np.moveaxis(np.diagonal(S_delta_backG, axis1=1, axis2=2), 1, -1)   # extracting the diagonal only (no interaction between g_nodes??). Apparently np.diagonal changes order of the dimensions so np.moveaxis is needed.
    # # Root of sum of squares (RSS) of the 3 first components (displacement components, not rotations) of S_delta. Extracting the diagonal terms only
    # S_delta_norm_backG = np.sqrt(np.sum(np.square(S_delta_backG[:,:,:3]), axis=-1))  #
    # # std_delta: formulation with just diagonal of S_deltadelta
    # std_delta_backG = np.sqrt(np.einsum('wmv,w->vm', np.real(S_delta_backG), delta_w_array))

    stop_time_2 = np.round((time.time() - start_time_2))
    start_time_3 = time.time()

    ########################################################################################################################
    ### PREVIOUSLY USED FORMULATION, WHICH WAS SUPER FAST, SIMPLIFIED, AND CONFIRMED. But it was not possible to obtain local axes values. Nor retrieve off-diagonal information.
    # Warning: M,M ->M. APPARENTLY CORRECT AS IT GIVES SAME RESULTS AS THE LAST ONE. MUCH FASTER!!! Does it take advantage that the Sb_FF is symmetric and H_tilde is diagonal?
    # Sb_FF_tilde = np.einsum('Mmv,wmnv,Mnv->wM', g_shapes, Sb_FF, g_shapes, optimize='optimal')
    # S_etaeta = np.einsum('wM,wM,wM->wM', np.conjugate(H_tilde), Sb_FF_tilde, H_tilde)
    # S_deltadelta_0 = np.einsum('Mmv,wM,Mmv->wmv', g_shapes, S_etaeta, g_shapes)  # 'Mmv,wM,Mmv->wmv'
    # # Forcing the dimensions of S_deltadelta from wmv to wmnv, where mn is diagonal matrix:
    # S_deltadelta = np.zeros((n_freq, g_node_num, g_node_num, 6), dtype='complex_')  #
    # for n in range(g_node_num):  # Or instead (but uses too much memory): S_deltadelta = np.moveaxis([[np.diag(S_deltadelta[w,:,v]) for v in range(6)] for w in range(n_freq)], 1, -1)
    #     S_deltadelta[:,n,n,:] = S_deltadelta_0[:,n,:]
    # S_delta = np.moveaxis(np.diagonal(S_deltadelta, axis1=1, axis2=2), 1, -1)  # extracting the diagonal only (no interaction between g_nodes??). Apparently np.diagonal changes order of the dimensions so np.moveaxis is needed.
    # # Root of sum of squares (RSS) of the 3 first components (displacement components, not rotations) of S_delta. Extracting the diagonal terms only
    # S_delta_norm = np.sqrt(np.sum(np.square(S_delta[:,:,:3]), axis=-1))  #
    # # std_delta: formulation with just diagonal of S_deltadelta
    # std_delta = np.sqrt(np.einsum('wmv,w->vm', np.real(S_delta), delta_w_array))
    ########################################################################################################################
    ### TRASH. WRONG AT END ONLY.
    ### Sb_FF_tilde = np.einsum('Mmv,wmnv,Nnv->wMN', g_shapes, Sb_FF, g_shapes, optimize='optimal')  # probably right? WHY ARE OFF-DIAGONALS DIFFERENT FROM 0 ?? MODAL COUPLING?
    ### S_etaeta = np.einsum('wM,wMN,wN->wMN', np.conjugate(H_tilde), Sb_FF_tilde, H_tilde)
    ### S_deltadelta = np.einsum('Mmv,wMN,Nnv->wmn', g_shapes, S_etaeta, g_shapes)  # 'Mmv,Mf,Mmv->mvf'  # probably wrong? summing the different DOF is not correct!
    ###
    ### WRONG AS MODAL FOCES FOR EACH DOF ARE DEPENDENT, NOT INDEPENDENT. KEEPING (MODES X MODES) and keeping vector DOF all the way.
    ### Sb_FF_tilde = np.einsum('Mmv,wmnv,Nnv->wMNv', g_shapes, Sb_FF, g_shapes, optimize='optimal')  # perhaps wrong since modal forces are independent for each DOF, when they are indeed dependent?
    ### S_etaeta = np.einsum('wM,wMNv,wN->wMNv', np.conjugate(H_tilde), Sb_FF_tilde, H_tilde)
    ### S_deltadelta = np.einsum('Mmv,wMNv,Nnv->wmnv', g_shapes, S_etaeta, g_shapes)  # 'Mmv,Mf,Mmv->mvf'
    ########################################################################################################################
    # # # RIGHT ONE BUT SLOWER? KEEPING (MODES X MODES) and vector. BEST ONE??? the DOF are only retrieved back in the end.
    # Sb_FF_tilde = np.einsum('Mmv,wmnv,Nnv->wMN', g_shapes, Sb_FF, g_shapes, optimize='optimal')
    # S_etaeta = np.einsum('wM,wMN,wN->wMN', np.conjugate(H_tilde), Sb_FF_tilde, H_tilde)  # H_tilde needs to be 'wMN' when off-diagonals exist. This may happen when Kse and Cse are included (self-excited forces)
    # S_deltadelta = np.einsum('Mmv,wMN,Nnv->wmnv', g_shapes, S_etaeta, g_shapes)  # 'Mmv,Mf,Mmv->mvf'
    ########################################################################################################################
    # # Double-checking Sb_FF
    # Sb_FF_2 = np.zeros((n_freq, g_node_num, g_node_num, 6))
    # for w in range(n_freq):
    #     for m in range(g_node_num):
    #         for n in range(g_node_num):
    #             for i in range(6):
    #                     Sb_FF_2[w,m,n,i] = Pb[m,i] @ np.diag(S_aa[w,m,n]) @ Pb[n,i]
    # np.max(np.absolute(Sb_FF_2 - Sb_FF))
    #
    # # Double-checking Sb_FF_tilde (when wrongly keeping the 6 DOF independent)
    # Sb_FF_tilde_2 = np.zeros((n_freq, n_modes, n_modes, 6))
    # for w in range(n_freq):
    #     for v in range(6):
    #         Sb_FF_tilde_2[w,:,:, v] = g_shapes[:,:,v] @ Sb_FF[w,:,:,v] @ np.transpose(g_shapes[:,:,v])
    # # np.max(np.absolute(Sb_FF_tilde - Sb_FF_tilde_2))
    # Sb_FF_tilde_3 = np.zeros((n_freq, n_modes, n_modes))
    # Sb_FF_tilde_3[:,:,:] = np.sum(Sb_FF_tilde_2, axis=-1)
    # np.max(np.absolute(Sb_FF_tilde - Sb_FF_tilde_3)) / np.max(np.absolute(Sb_FF_tilde + Sb_FF_tilde_3))
    ########################################################################################################################

    # Plotting colormaps
    #
    # plt.figure()
    # plt.title(r'$S_{\Delta\Delta} (x, \omega)$')
    # plt.contourf(x,y, np.real(S_delta_norm), 256, cmap='jet')
    # plt.xlabel('Node x-coordinate [m]')
    # plt.ylabel('Frequency [rad/s]')
    # plt.colorbar()
    # plt.grid()
    #
    # # plt.figure()
    # # plt.title(r'$\sigma_{\Delta} (f,x)$')
    # # plt.contourf(x,y, np.sqrt(np.einsum('w,wn->wn', delta_w_array, S_delta_norm)), 256, cmap='rainbow')
    # # plt.xlabel('Node x-coordinate [m]')
    # # plt.ylabel('Frequency [rad/s]')
    # # plt.colorbar()
    # # plt.grid()


    ####################################################################################################################################################################################
    ## THE MAIN CONTOURF PLOT IS HERE @@@@@@@@@@@@@@@@@@@@@@@@@@@ ######################################################################################################################
    # from buffeting_plots import plot_contourf_spectral_response
    # plot_contourf_spectral_response(f_array, S_delta_local, g_node_coor, S_by_freq_unit='rad', zlims_bool=False, cbar_extend='min', filename='Contour_FD_', idx_plot=[1,2,3])
    ## THE MAIN CONTOURF PLOT IS HERE @@@@@@@@@@@@@@@@@@@@@@@@@@@ ######################################################################################################################
    ####################################################################################################################################################################################

    #
    # plt.figure()
    # plt.title(r'Vertical displacements only. $log(S_{\Delta\Delta} (x, \omega))$')
    # plt.contourf(x,y, np.log(np.real(S_delta[:,:,2])), 256, cmap='jet', vmin=-20, vmax=-2)
    # plt.ylabel('Frequency [rad/s]')
    # plt.xlabel('Node x-coordinate [m]')
    # plt.colorbar()
    # plt.grid()
    #
    # plt.figure()
    # plt.title(r'Torsional displacements only. $log(S_{\Delta\Delta} (x, \omega))$')
    # plt.contourf(x,y, np.log(np.real(S_delta_local[:,:,3])), 256, cmap='jet', vmin=-20, vmax=-10)
    # plt.ylabel('Frequency [rad/s]')
    # plt.xlabel('Node x-coordinate [m]')
    # plt.colorbar()
    # plt.axhline(w_eigen[8], ls='--', color='black', label='1st vertical mode')
    # plt.axhline(w_eigen[37], ls='--', color='orange', label='1st torsional mode')
    # plt.legend()
    # plt.grid()
    #
    # plt.figure()
    # plt.title(r'Vertical displacements only. $log(S_{\Delta\Delta} (x, \omega))$')
    # plt.contourf(x,y, np.log(np.real(S_delta_local[:,:,2])), 256, cmap='jet', vmin=-20, vmax=-2)
    # plt.ylabel('Frequency [rad/s]')
    # plt.xlabel('Node x-coordinate [m]')
    # plt.colorbar()
    # plt.axhline(w_eigen[8], ls='--', color='black', label='1st vertical mode')
    # plt.axhline(w_eigen[37], ls='--', color='orange', label='1st torsional mode')
    # plt.legend()
    # plt.grid()
    #
    #
    # plt.figure()
    # plt.title(r'Vertical displacements only. $S_{\Delta\Delta} (x, \omega)$')
    # plt.contourf(x,y, np.real(S_delta[:,:,2]), 256, cmap='jet')
    # plt.ylabel('Frequency [rad/s]')
    # plt.xlabel('Node x-coordinate [m]')
    # plt.axhline(w_array[0])
    # plt.grid()
    # plt.colorbar()
    #
    # plt.figure()
    # plt.title(r' Any-direction displacements. $log(S_{\Delta\Delta} (x, \omega))$')
    # plt.contourf(x,y, np.log(np.real(S_delta_norm)), 256, cmap='jet', vmin=-10, vmax=8)
    # plt.ylabel('Frequency [rad/s]')
    # plt.xlabel('Node x-coordinate [m]')
    # plt.colorbar()
    # plt.grid()
    # plt.show()

    # Plotting local VS global std. of the response along the bridge girder.
    from simple_5km_bridge_geometry import g_s_3D_func
    g_s_3D = g_s_3D_func(g_node_coor)
    if plot_std_along_girder:
        plt.figure(figsize=(3.65, 2.65), dpi=400)
        plt.title(r'$\sigma_{i}=\sqrt{\int S_{\Delta_{i}} (\omega) d\omega }$')
        # plt.plot(g_s_3D, std_delta[0], label='X - FD', color='b')
        # plt.plot(g_s_3D, std_delta[1], label='Y - FD', color='orange')
        # plt.plot(g_s_3D, std_delta[2] * 10, label='Z - FD * 10', color='g')
        # plt.plot(g_s_3D, std_delta[3] * 1000, label='XX - FD * 1000', color='m')
        plt.plot(g_s_3D, std_delta_local[0], label=r'$\sigma_{x}\/[m]$', linestyle="--", color='b', alpha=0.8)
        plt.plot(g_s_3D, std_delta_local[1], label=r'$\sigma_{y}\/[m]$', linestyle="-", color='orange', alpha=0.8)
        plt.plot(g_s_3D, std_delta_local[2] * 10, label=r'$\sigma_{z}\times 10\/[m]$', linestyle="-.", color='g', alpha=0.8)
        plt.plot(g_s_3D, std_delta_local[3] * 10 *180/np.pi, label=r'$\sigma_{rx}\times10\/[\degree]$', linestyle=":", color='m', alpha=0.8)
        plt.ylim(0, 4)
        plt.xlim(0, 5000)
        plt.xticks([0, 2500, 5000])
        # plt.legend(loc='upper right', ncol=4)
        plt.xlabel('Along arc length [m]')
        # plt.ylabel(r'$[m]$ or $[m\times10]$ or $[rad\times1000]$')
        plt.grid()
        plt.tight_layout()
        handles,labels = plt.gca().get_legend_handles_labels()
        plt.savefig(
            'results\FD_BetaDB-' + str('%.0f' % round(deg(beta_DB))) + '_Spec-' + str(cospec_type) + \
            '_zeta-' + str(damping_ratio) + '_Ti-' + str(damping_Ti) + '_Tj-' + str(damping_Tj) + '_Nodes-' + \
            str(g_node_num) + '_Modes-' + str(n_modes) + '_f-' + str(f_array[0]) + '-' + str(f_array[-1]) + \
            '-' + str(n_freq) + '_' + str(aero_coef_method)[:6] + '_Ca-' + str(n_aero_coef)[:1] + '_SE-' + \
            str(include_SE)[:1] + '_FD-' + '.png')
        plt.close()
        # Legend
        plt.figure(figsize=(3, 1), dpi=400)
        plt.axis("off")
        plt.legend(handles,labels, ncol=2)
        plt.tight_layout
        plt.savefig('results\lengend_FD_along_arc_STD.png')
        plt.close()

    # Execution info
    stop_time_3 = np.round((time.time() - start_time_3))
    print('Kse & Cse, Pre-processing, Buffeting, and Post-processing times (s): ' + str(stop_time_Kse_Cse) + ', ' + str(stop_time_1) + ', ' + str(stop_time_2) + \
          ', ' + str(stop_time_3) + '. Total time (s): ' + str(stop_time_Kse_Cse+stop_time_1+stop_time_2+stop_time_3))

    return {'std_delta_local': std_delta_local,
            'cospec_type':cospec_type,
            'damping_ratio':damping_ratio,
            'damping_Ti':damping_Ti,
            'damping_Tj':damping_Tj}

def list_of_cases_FD_func(n_aero_coef_cases, include_SE_cases, aero_coef_method_cases, beta_DB_cases, flutter_derivatives_type_cases, n_freq_cases, n_modes_cases,
                          n_nodes_cases, f_min_cases, f_max_cases, include_sw_cases, include_KG_cases, skew_approach_cases, f_array_type_cases, make_M_C_freq_dep_cases,
                          dtype_in_response_spectra_cases, Nw_idxs, Nw_or_equiv_Hw_cases):
    # List of cases (parameter combinations) to be run:
    list_of_cases = [(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,z) for a in aero_coef_method_cases for b in n_aero_coef_cases for c in include_SE_cases
                     for d in flutter_derivatives_type_cases for e in n_modes_cases for f in n_freq_cases
                     for g in n_nodes_cases for h in f_min_cases for i in f_max_cases for j in include_sw_cases for k in include_KG_cases for l in skew_approach_cases
                     for m in f_array_type_cases for n in make_M_C_freq_dep_cases for o in dtype_in_response_spectra_cases for p in Nw_idxs for q in Nw_or_equiv_Hw_cases for z in beta_DB_cases] # Note: new parameters should be added before beta_DB
    list_of_cases = [list(case) for case in list_of_cases
                     if not (('3D' in case[3] and '2D' in case[11]) or ('2D' in case[3] and '3D' in case[11]) or (case[2]==False and case[3] in ['3D_Scanlan', '3D_Scanlan_confirm', '3D_Zhu', '3D_Zhu_bad_P5', '2D_in_plane']))]
    # if skew_approach is '3D' only '3D' flutter_derivatives accepted. If SE=False, only one dummy FD case is accepted: '3D_full' or '2D_full'
    return list_of_cases

def parametric_buffeting_FD_func(list_of_cases, g_node_coor, p_node_coor, Ii_simplified, R_loc, D_loc, cospec_type=2, include_modal_coupling=True, include_SE_in_modal=False):
    # Empty Dataframe to store results
    results_df = pd.DataFrame(list_of_cases)
    results_df.columns = ['Method', 'n_aero_coef', 'SE', 'FD_type', 'n_modes', 'n_freq', 'g_node_num', 'f_min', 'f_max', 'SWind', 'KG', 'skew_approach', 'f_array_type', 'make_M_C_freq_dep',
                          'dtype_in_response_spectra', 'Nw_idx', 'Nw_or_equiv_Hw', 'beta_DB']
    for i in range(0, 6):
        results_df['std_max_dof_' + str(i)] = None

    case_idx = -1  # index of the case
    for aero_coef_method, n_aero_coef, include_SE, flutter_derivatives_type, n_modes, n_freq, g_node_num, f_min, f_max, include_sw, include_KG, skew_approach, f_array_type, make_M_C_freq_dep, \
        dtype_in_response_spectra, Nw_idx, Nw_or_equiv_Hw, beta_DB in list_of_cases:
        case_idx += 1  # starts at 0.
        buffeting_results = buffeting_FD_func(include_sw, include_KG, aero_coef_method, n_aero_coef, skew_approach, include_SE, flutter_derivatives_type, n_modes, f_min, f_max, n_freq, g_node_coor,
                                              p_node_coor, Ii_simplified, beta_DB, R_loc, D_loc, cospec_type, include_modal_coupling, include_SE_in_modal, f_array_type, make_M_C_freq_dep,
                                              dtype_in_response_spectra, Nw_idx, Nw_or_equiv_Hw)
        # Reading results
        std_delta_local = buffeting_results['std_delta_local']
        cospec_type = buffeting_results['cospec_type']
        damping_ratio = buffeting_results['damping_ratio']
        damping_Ti = buffeting_results['damping_Ti']
        damping_Tj = buffeting_results['damping_Tj']
        # Writing results
        results_df.at[case_idx, 'cospec_type'] = cospec_type
        results_df.at[case_idx, 'damping_ratio'] = damping_ratio
        results_df.at[case_idx, 'damping_Ti'] = damping_Ti
        results_df.at[case_idx, 'damping_Tj'] = damping_Tj
        for i in range(0,6):
            results_df.at[case_idx, 'std_max_dof_'+str(i)] = np.max(std_delta_local[i])

    # Exporting the results to a table
    from time import gmtime, strftime
    results_df.to_csv(r'results\FD_std_delta_max_'+strftime("%Y-%m-%d_%H-%M-%S", gmtime())+'.csv')
    return None

########################################################################################################################
# Time Domain Buffeting Analysis:
########################################################################################################################
def wind_field_3D_all_blocks_func(g_node_coor, beta_DB, dt, wind_block_T, wind_overlap_T, wind_T, ramp_T, cospec_type,
                                  Ii_simplified, plots=False):
    """
    Generates wind speed time series [U+u, u, v, w] for each g_node, in global wind Gw coordinates (XuYvZw),
    with shape: (4,g,t) by concatenating wind blocks with duration wind_block_T each, overlapped by wind_overlap_T,
    with total duration of wind_T, including transient_T. Wind speeds are smoothened during wind_overlap_T, from one wind_block_T, to the next.
    """
    beta_0 = beta_0_func(beta_DB)
    g_node_num = len(g_node_coor)

    T_GsGw = T_GsGw_func(beta_0, theta_0)

    U_bar = U_bar_func(g_node_coor)
    Ai = Ai_func(cond_rand_A=False)
    Cij = Cij_func(cond_rand_C=False)
    iLj = iLj_func(g_node_coor)
    Ii = Ii_func(g_node_coor, beta_DB, Ii_simplified)
    # ----------------------------------------------------------------------------------------------------------------------
    # WIND FIELD
    # ----------------------------------------------------------------------------------------------------------------------
    # Smaller wind files (blocks) are generated and then concatenated together into a full wind time series, with overlapping and a linear smooth function in the transition between blocks.
    from wind_field.wind_field_3D import wind_field_3D_func

    wind_freq = 1 / dt  # (Hz). Frequency at which time-domain simulation is calculated

    # Other variables
    wind_overlap_size = int(wind_overlap_T * wind_freq)
    wind_block_T_raw_first = wind_block_T + wind_overlap_T / 2
    wind_block_T_raw = wind_block_T + wind_overlap_T
    wind_block_T_raw_last = wind_block_T + wind_overlap_T / 2
    n_wind_blocks = int(wind_T / wind_block_T)  # total number of wind blocks, copied from one original and concatenated together
    n_nodes_wind = copy.deepcopy(g_node_num)  # g_node_num. To be deleted eventually.
    node_coor_wind = np.einsum('ni,ij->nj', g_node_coor, T_GsGw)  # Nodes in wind coordinates. X along, Y across, Z vertical
    iLj_reshape = np.reshape(np.moveaxis(iLj, 0, -1), (9, g_node_num))  # to adapt to the wind_field_3D_function

    # Alert for possible mistakes:
    assert (wind_T / wind_block_T).is_integer(), 'Error: wind_T should be multiple of wind_block_T!!'
    assert (wind_T / dt).is_integer(), 'Error: wind_T should be multiple of dt!!'
    assert (ramp_T / dt).is_integer(), 'Error: ramp_T should be multiple of dt!!'
    assert (wind_overlap_T / dt).is_integer(), 'Error: smooth_transition_T should be multiple of dt!!'
    assert (wind_overlap_size / 2).is_integer(), 'Error: smooth_transition_size must be even!!'

    # GENERATING WIND:
    import time
    start_time = time.time()

    # First block
    wind_field_data = [wind_field_3D_func(node_coor_wind[:n_nodes_wind], U_bar[:n_nodes_wind], Ai, Cij.flatten(),
                                          Ii[:n_nodes_wind], iLj_reshape[:, :n_nodes_wind], wind_block_T_raw_first,
                                          wind_freq, spectrum_type=cospec_type)]
    print("--- %s seconds. Generated wind block number 1" % np.round_(time.time() - start_time))

    # Next blocks
    for n in range(1, n_wind_blocks - 1):
        wind_field_data.append(wind_field_3D_func(node_coor_wind[:n_nodes_wind], U_bar[:n_nodes_wind], Ai, Cij.flatten(),
                                                  Ii[:n_nodes_wind], iLj_reshape[:, :n_nodes_wind], wind_block_T_raw,
                                                  wind_freq,
                                                  spectrum_type=cospec_type))
        print("--- %s seconds. Generated wind block number " % np.round_(time.time() - start_time) + str(n + 1))

    # Last block
    wind_field_data.append(wind_field_3D_func(node_coor_wind[:n_nodes_wind], U_bar[:n_nodes_wind], Ai, Cij.flatten(),
                                              Ii[:n_nodes_wind], iLj_reshape[:, :n_nodes_wind], wind_block_T_raw_last,
                                              wind_freq,
                                              spectrum_type=cospec_type))
    print("--- %s seconds in total to generate wind ---" % np.round_(time.time() - start_time))

    # Retrieving wind speeds (discarding first value of all blocks except the first one)
    windspeed_raw = [wind_field_data[0]["windspeed"]]
    time_array_raw = [wind_field_data[0]["timepoints"]]
    for n in range(1, n_wind_blocks):
        windspeed_raw.append(wind_field_data[n]["windspeed"][:, :, 1:])
        time_array_raw.append(
            np.array(wind_field_data[n]["timepoints"][1:]) + time_array_raw[n - 1][-(wind_overlap_size + 1)])

    def smooth_func(v1, v2):
        """
        Returns an array which is a function of v1 and v2, going from similar to v1 to increasingly more similar to v2.
        For n-D arrays, only last dimension is used for smoothing.
        :param v1: array
        :param v2: array
        :return: smooth overlapping array
        """
        if len(v1.shape) == 1 and v2.shape == v1.shape:
            size = len(v1)
        elif len(v1.shape) > 1 and v2.shape == v1.shape:
            size = v1.shape[-1]
        linear = np.linspace(1 / (size + 1), 1 - 1 / (size + 1), size)  # choose other function here as desired
        return v1 * (1 - linear) + v2 * linear


    # OVERLAPPING
    # First block
    windspeed = np.array(windspeed_raw[0][:, :, :-wind_overlap_size])  # first clean part
    windspeed_overlap = smooth_func(windspeed_raw[0][:, :, -wind_overlap_size:],
                                    windspeed_raw[1][:, :, :wind_overlap_size])  # first overlapped part
    windspeed = np.concatenate((windspeed, windspeed_overlap), axis=-1)  # concatenating

    # Following blocks
    for n in range(1, n_wind_blocks - 1):
        windspeed = np.concatenate((windspeed, windspeed_raw[n][:, :, wind_overlap_size:-wind_overlap_size]),
                                   axis=-1)  # concatenating next clean part
        windspeed_overlap = smooth_func(windspeed_raw[n][:, :, -wind_overlap_size:],
                                        windspeed_raw[n + 1][:, :, :wind_overlap_size])  # next overlapped part
        windspeed = np.concatenate((windspeed, windspeed_overlap), axis=-1)  # concatenation

    # Last block
    windspeed = np.concatenate((windspeed, windspeed_raw[-1][:, :, wind_overlap_size:]), axis=-1)  # concatenating next clean part

    # Ramping up windspeeds during the first half of the transient_T, instead of starting full speed
    if ramp_T != 0:
        ramp_T_size = int(ramp_T * wind_freq)
        windspeed[:,:,:ramp_T_size] = np.einsum('igt,t->igt', windspeed[:,:,:ramp_T_size], np.linspace(0.1,1,ramp_T_size))  # cannot start at 0, or NaN occurs.

    # Final time array
    time_array = np.linspace(0, wind_T, int(wind_T * wind_freq + 1))

    # Plotting the windspeed VS windspeed_raw for comparison and understanding
    if plots:
        plt.figure()
        plt.title(
            'Time series of the wind speed. Comparison between individual wind blocks, and the continuous smoothened function')
        for n in range(n_wind_blocks):
            plt.plot(time_array_raw[n], windspeed_raw[n][0, 0])
        plt.plot(time_array, windspeed[0, 0], label='smoothed continuous wind', linewidth=3)
        plt.legend()
        plt.ylabel('Wind speed [m/s]')
        plt.xlabel('Time [s]')
        plt.show()

    return windspeed

def MDOF_TD_NL_wind_solver(g_node_coor, p_node_coor, beta_0, theta_0, aero_coef_method, n_aero_coef, include_KG, geometric_linearity, R_loc, D_loc, M, C, K, windspeed, u0, v0, T, dt, gamma=1 / 2, beta=1 / 4):
    """Multi-degree of freedom time-domain solver, for self-excited + wind forces. The instantaneous angles are calculated \n
    at each time step, due to both turbulence and structure motions, and then instantanous coefficients and forces are calculated. \n
    Reference: Page 19 (of 55) of "Time Domain Methods - ETHZ course material.pdf" (or of https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femII/presentation_05_dynamics_v3.pdf) \n
    M -- Mass matrix \n
    C -- Damping matrix = damp_ratio* 2*np.sqrt(K*M) \n
    K -- Stiffness matrix (already including KG and/or Kse if necessary) \n
    windspeed -- Array with wind speeds with components: [V,u,v,w]. shape:(4,g,t). Time length = len(np.arange(0,T+dt,dt)) \n
    u0 -- Initial displacement (m) \n
    v0 -- Initial velocity (m/s) \n
    T -- Simulation duration (s) \n
    dt -- Time step (s). Should be smaller then eigen period / 40. \n
    gamma -- Newmark parameter. \n
    beta -- Newmark parameter. \n
    [1] https://en.wikipedia.org/wiki/Newmark-beta_method \n
    """
    from scipy import interpolate

    time = np.arange(0, T + dt, dt)
    dt_array = np.array([time[i + 1] - time[i] for i in range(len(time[:-1]))])
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    p_node_num = len(p_node_coor)

    R_loc = copy.deepcopy(R_loc)  # Otherwise it would increase during repetitive calls of this function?
    D_loc = copy.deepcopy(D_loc)  # Otherwise it would increase during repetitive calls of this function?
    girder_N = copy.deepcopy(R_loc[:g_elem_num, 0])
    c_N = copy.deepcopy(R_loc[g_elem_num:, 0])
    alpha = copy.deepcopy(D_loc[:g_node_num, 3])

    # Variables, calculated once
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)

    # Creating linear interpolation functions of aerodynamic coefficients, from an extensive grid of possible angles:
    grid_inc = rad(0.05)  # grid angle increments
    beta_grid = np.arange(-np.pi, np.pi + grid_inc*0.9, grid_inc)  # change the grid discretization here, as desired! IT NEEDS TO INCLUDE THE END POINT, therefore + grid_inc !! /100 is to simply have something bigger
    theta_grid = np.arange(-np.pi / 4, np.pi / 4 + grid_inc*0.9, grid_inc)  # change the grid interval and discretization here, as desired!  # todo: 0.1 INSTEAD OF 0.5. BETTER?

    # If a C_Ci_grid file already exists containing all coefficients in an extensive grid
    if os.path.isfile(os.getcwd()+r'\aerodynamic_coefficients\C_Ci_grid.npy'):
        C_Ci_grid = np.load(os.getcwd()+r'\aerodynamic_coefficients\C_Ci_grid.npy')
        print('Using previously existing grid of aerodynamic coefficients for interpolation')
    else:
        xx, yy = np.meshgrid(beta_grid, theta_grid)
        C_Ci_grid_flat = C_Ci_func(xx.flatten(), yy.flatten(), aero_coef_method, n_aero_coef, coor_system='Lw')
        C_Ci_grid = C_Ci_grid_flat.reshape((6, len(theta_grid), len(beta_grid)))
        np.save(os.getcwd()+r'\aerodynamic_coefficients\C_Ci_grid.npy', C_Ci_grid)
        print('Created extensive grid of aerodynamic coefficients for interpolation')
    C_C0_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1, 2)[0], kx=1, ky=1)
    C_C1_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1, 2)[1], kx=1, ky=1)
    C_C2_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1, 2)[2], kx=1, ky=1)
    C_C3_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1, 2)[3], kx=1, ky=1)
    C_C4_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1, 2)[4], kx=1, ky=1)
    C_C5_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1, 2)[5], kx=1, ky=1)
    print('Created interpolation functions for the aerodynamic coefficients, from an existing grid')

    # Start Newmark
    # =========================================================================
    # Constant-Average-Acceleration Method. (see Newmark-Beta method)
    # =========================================================================
    i = 0  # counter.
    dt = dt_array[0]
    u_new = [u0]  # Initial displacement
    v_new = [v0]  # Initial velocity

    F = Fad_one_t_C_Ci_NL_with_SE(g_node_coor, p_node_coor, alpha, beta_0, theta_0, windspeed[:,:,i], v_new,
                                  C_C0_func, C_C1_func, C_C2_func, C_C3_func, C_C4_func, C_C5_func)

    if include_KG:
        K_tot = copy.deepcopy(K - geom_stiff_matrix_func(g_node_coor, p_node_coor, girder_N, c_N, alpha))

    # Initial acceleration
    a_new = [ np.linalg.inv(M + C * gamma * dt + K_tot * beta * dt ** 2) @ (F - K_tot @ u_new[-1] - C @ v_new[-1])]

    for _ in time[1:]:
        i += 1
        dt = dt_array[i - 1]
        u = u_new[-1]  # u & u_new are position      at time step i-1 & i respectively.
        v = v_new[-1]  # v & v_new are velocities    at time step i-1 & i respectively.
        a = a_new[-1]  # a & a_new are accelerations at time step i-1 & i respectively.
        # Calculating the predictors:
        u_new.append(u + v * dt + a * (1 / 2 - beta) * dt ** 2)
        u_new_reshape = copy.deepcopy(np.reshape(u_new[-1], (g_node_num + p_node_num, 6)))
        v_new.append(v + a * (1 - gamma) * dt)
        alpha_new = np.einsum('nij,nj->ni', T_LsGs, u_new_reshape[:g_node_num, 3:])[:, 0]  # final shape is (n). Using alpha original to calculate T_LsGs!
        g_node_coor_new = copy.deepcopy(g_node_coor + u_new_reshape[:g_node_num, :3])  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
        p_node_coor_new = copy.deepcopy(p_node_coor + u_new_reshape[g_node_num:, :3])  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw

        # 1: Updating the motion-dependent axial forces, based on new 'u' and on 'previous' (or 'original' if linear geometry) alpha and coordinates
        if include_KG and geometric_linearity == 'L':
            from static_loads import static_wind_func, R_loc_func
            # Attention: these node_coor and alpha need to be the ones before the displacements for R_loc to be correct.
            R_loc = copy.deepcopy(R_loc_func(u_new_reshape, g_node_coor, p_node_coor, alpha))  # orig. coord. + displacem. used to calc. R! coord are updated next.
            girder_N = copy.deepcopy(R_loc[:g_elem_num, 0])  # No girder axial forces
            c_N = copy.deepcopy(R_loc[g_elem_num:, 0])  # No columns axial forces
            K_tot = copy.deepcopy(K) - geom_stiff_matrix_func(g_node_coor, p_node_coor, girder_N, c_N, alpha)  # K is not taken at new displaced position when geometric_linearity == 'L'

        if geometric_linearity == 'NL':
            raise ValueError("geometric_linearity == 'NL' not implemented")
            # 2: Updating alpha and the coordinates:
            alpha = np.einsum('nij,nj->ni', T_LsGs, u_new_reshape[:g_node_num, 3:])[:, 0]  # final shape is (n). Using alpha original to calculate T_LsGs!
            F_prev = copy.deepcopy(F)
            M_new = copy.deepcopy(mass_matrix_func(g_node_coor_new, p_node_coor_new, alpha))
            K_new = copy.deepcopy(stiff_matrix_func(g_node_coor_new, p_node_coor_new, alpha))
            # todo: geometric_linearity could be called "large_displacements" or "K_and_M_updating"?
            # todo: if NLGeom, T_LsGs needs to be calculated at each step
            # todo: Keep a F_new and F_prev, so that F_prev is the restoring force accumulated, after building new structure with new K and M on new coordinates.

        # Aerodynamic forces, based on instantaneous bridge displacements and velocities
        F = Fad_one_t_C_Ci_NL_with_SE(g_node_coor_new, p_node_coor_new, alpha_new, beta_0, theta_0, windspeed[:, :, i], v_new,
                                      C_C0_func, C_C1_func, C_C2_func, C_C3_func, C_C4_func, C_C5_func)

        # Solution of the linear problem:
        if geometric_linearity == 'L':
            a_new.append(np.linalg.inv(M + C * gamma * dt + K_tot * beta * dt ** 2) @ (F - K_tot @ u_new[-1] - C @ v_new[-1]))

        elif geometric_linearity == 'NL':
            raise ValueError("geometric_linearity == 'NL' not implemented")
            a_new.append(np.linalg.inv(M_new + C * gamma * dt + K_new * beta * dt ** 2) @ (F - K_new @ (u_new[-1] - u_new[2]) - C @ v_new[-1]))

        # Correcting the predictors:
        u_new[-1] = u_new[-1] + a_new[-1] * beta * dt ** 2  # this was the missing term in "Calculating the predictors" step.
        v_new[-1] = v_new[-1] + a_new[-1] * gamma * dt  # this was the missing term in "Calculating the predictors" step.

        # # One way of checking instabilities?
        # if np.max(abs(deg(alpha_new))) > 5:
        #     print(str(np.round(np.max(abs(deg(alpha))), 1)) + '  at time:' + str(_))


        if i % 1000 == 0:
            print('time step: ' + str(i))

    return {'u': np.array(u_new),
            'v': np.array(v_new),
            'a': np.array(a_new)}

def buffeting_TD_func(aero_coef_method, skew_approach, n_aero_coef, include_SE, flutter_derivatives_type, include_sw, include_KG, g_node_coor, p_node_coor, Ii_simplified, R_loc, D_loc, n_seeds, dt, wind_block_T,
                      wind_overlap_T, wind_T, transient_T, ramp_T, beta_DB, aero_coef_linearity, SE_linearity, geometric_linearity, cospec_type=2, plots=False, save_txt=False):

    g_node_num = len(g_node_coor)
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0
    g_elem_num = g_node_num - 1
    p_node_num = len(p_node_coor)
    beta_0 = beta_0_func(beta_DB)

    R_loc = copy.deepcopy(R_loc)  # Otherwise it would increase during repetitive calls of this function
    D_loc = copy.deepcopy(D_loc)  # Otherwise it would increase during repetitive calls of this function
    girder_N = copy.deepcopy(R_loc[:g_elem_num, 0])  # No girder axial forces
    c_N = copy.deepcopy(R_loc[g_elem_num:, 0])  # No columns axial forces
    alpha = copy.deepcopy(D_loc[:g_node_num, 3])  # No girder nodes torsional rotations

    # assert include_sw, 'include_sw must be always True in Time Domain'
    assert not (include_SE==False and SE_linearity=='NL'), "include_SE = False can only be used together with SE_linearity='L'"

    # # delete??
    # if include_sw and :  # including static wind
    #     # Displacements
    #     g_node_coor_sw, p_node_coor_sw, D_glob_sw = static_wind_func(g_node_coor, p_node_coor, alpha, beta_DB,
    #                                                             theta_0, aero_coef_method, n_aero_coef)
    #     D_loc_sw = mat_Ls_node_Gs_node_all_func(D_glob_sw, g_node_coor, p_node_coor)
    #     alpha_sw = copy.deepcopy(D_loc_sw[:g_node_num, 3])  # local nodal torsional rotation.
    #     # Internal forces
    #     R_loc_sw = R_loc_func(D_glob_sw, g_node_coor, p_node_coor)  # orig. coord. + displacem. used to calc. R.
    #     girder_N_sw = copy.deepcopy(R_loc_sw[:g_elem_num, 0])  # local girder element axial force. Positive = compression!
    #     c_N_sw = copy.deepcopy(R_loc_sw[g_elem_num:, 0])  # local column element axial force Positive = compression!
    #     # Updating structure.
    #     g_node_coor, p_node_coor = copy.deepcopy(g_node_coor_sw), copy.deepcopy(p_node_coor_sw)
    #     R_loc += copy.deepcopy(R_loc_sw)  # element local forces
    #     D_loc += copy.deepcopy(D_loc_sw)  # nodal global displacements. Includes the alphas.
    #     girder_N += copy.deepcopy(girder_N_sw)
    #     c_N += copy.deepcopy(c_N_sw)
    #     alpha += copy.deepcopy(alpha_sw)
    # # delete??

    beta_bar, theta_bar = beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha)

    # Transformation from all nodes in Global to Local
    T_LsGs_full_2D_node_matrix = T_LsGs_full_2D_node_matrix_func(g_node_coor, p_node_coor, alpha)

    # ----------------------------------------------------------------------------------------------------------------------
    # STRUCTURAL PROPERTIES
    # ----------------------------------------------------------------------------------------------------------------------
    # Importing mass and stiffness matrices. Units: (N):
    M = mass_matrix_func(g_node_coor, p_node_coor, alpha, w_array=None, make_freq_dep=False)
    K = stiff_matrix_func(g_node_coor, p_node_coor, alpha)
    C_added = added_damping_global_matrix_func(w_array=None, make_freq_dep=False)
    if include_KG:
        KG = geom_stiff_matrix_func(g_node_coor, p_node_coor, girder_N, c_N, alpha)
        K = copy.deepcopy(K) - KG

    if include_SE:
        f_array = np.array([0.001])  # fictitious. It will be cancelled out since the QS FD are not frequency dependent. This was confirmed.
        Kse, Cse = Kse_Cse_func(g_node_coor, U_bar, beta_bar, theta_bar, alpha, f_array, flutter_derivatives_type, aero_coef_method, n_aero_coef, skew_approach)
        Kse_0 = Kse[0]  # removing the frequency dimension
        Cse_0 = Cse[0]  # removing the frequency dimension
        Kse_full_mat_0 = np.zeros(((g_node_num + p_node_num) * 6, (g_node_num + p_node_num) * 6))
        Cse_full_mat_0 = np.zeros(((g_node_num + p_node_num) * 6, (g_node_num + p_node_num) * 6))
        for n in range(g_node_num):
            Kse_full_mat_0[n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Kse_0[n]
            Cse_full_mat_0[n * 6:n * 6 + 6, n * 6:n * 6 + 6] = Cse_0[n]
        K_tot = K + Kse_full_mat_0
        C_tot = rayleigh_damping_matrix_func(M, K_tot, damping_ratio, damping_Ti, damping_Tj) + C_added + Cse_full_mat_0
    else:
        K_tot = copy.deepcopy(K)
        C_tot = rayleigh_damping_matrix_func(M, K_tot, damping_ratio, damping_Ti, damping_Tj) + C_added

    # Modal analysis:
    _, _, w_eigen, _ = simplified_modal_analysis_func(M, K_tot)

    std_delta_local_all_seeds = []

    transient_resp_error = np.e ** (-damping_ratio * w_eigen[0] * transient_T)  # (amplitude of transient response / total response), at the end of the transient (ramp-up) period. In a damped free vibration (decaying response), the response amplitude = e**(-zeta*omega*t)
    print('Expected residual response at the end of the transient time is: %s percent' % str(np.round_(transient_resp_error * 100, 2)))

    wind_freq = 1/dt
    time_array = np.linspace(0, wind_T, int(wind_T * wind_freq + 1))
    # Number of simulations
    for seed in range(0, n_seeds):
        # ------------------------------------------
        # WIND FIELD
        # ------------------------------------------
        # Possible error:
        assert (transient_T / wind_block_T).is_integer(), 'Error: transient_T should be multiple of wind_block_T'

        windspeed = wind_field_3D_all_blocks_func(g_node_coor, beta_DB, dt, wind_block_T, wind_overlap_T, wind_T, ramp_T, cospec_type, Ii_simplified, plots=False)


        # # TESTING WITH STATIC WIND @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # windspeed = np.zeros((4,g_node_num,len(time_array)))
        # windspeed[0,:,:] = 30
        # # TESTING WITH STATIC WIND @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # ------------------------------------------
        # NEWMARK METHOD. PHYSICAL-SPACE FORMULATION
        # ------------------------------------------
        from newmark_method import MDOF_TD_solver

        start_time = time.time()

        # # STATIC TEST VALIDATION (Use F instead of Fb in the MDOF_TD_solver)
        # F = np.zeros((len(time_array), n_dof))
        # C_Drag = C_Dp * CS_width / CS_height
        # dof_excited = np.array(range(g_node_num))*6 + 1  # dof 2, at every girder node
        # for d,i in zip(dof_excited, range(g_node_num)):
        #     F[:,d] = np.array([1/2 * rho * U_bar[i]**2 * L_3D_nodes[i] * CS_height * C_Drag /1000]*len(time_array))  #  (kN)

        # Initial conditions
        u0 = np.array([0] * len(M))  # 0 initial displacement at every dof
        v0 = np.array([0] * len(M))  # 0 initial velocity at every dof

        if SE_linearity == 'L':  # also valid if include_SE = False, because K_tot and C_tot will not include SE
            if aero_coef_linearity == 'L':  # 'L' -> Linear aero_coef and derivates used, with Taylor's formula.
                if include_sw:
                    F = Fad_or_Fb_all_t_Taylor_hyp_func(g_node_coor, p_node_coor, alpha, beta_0, theta_0, beta_bar, theta_bar, windspeed, aero_coef_method, n_aero_coef, skew_approach, which_to_get='Fad')
                else:
                    F = Fad_or_Fb_all_t_Taylor_hyp_func(g_node_coor, p_node_coor, alpha, beta_0, theta_0, beta_bar, theta_bar, windspeed, aero_coef_method, n_aero_coef, skew_approach, which_to_get='Fb')
                read_dict = MDOF_TD_solver(M=M, C=C_tot, K=K_tot, F=F, u0=u0, v0=v0, T=wind_T, dt=dt)
            elif aero_coef_linearity == 'NL':  # aero_coefficients used as functions of beta_tilde and theta_tilde
                if include_sw:
                    F = Fad_or_Fb_all_t_C_Ci_NL_no_SE_func(g_node_coor, p_node_coor, alpha, beta_0, theta_0, beta_bar, theta_bar, windspeed, aero_coef_method, n_aero_coef, skew_approach, which_to_get='Fad')
                else:
                    F = Fad_or_Fb_all_t_C_Ci_NL_no_SE_func(g_node_coor, p_node_coor, alpha, beta_0, theta_0, beta_bar, theta_bar, windspeed, aero_coef_method, n_aero_coef, skew_approach, which_to_get='Fb')
                read_dict = MDOF_TD_solver(M=M, C=C_tot, K=K_tot, F=F, u0=u0, v0=v0, T=wind_T, dt=dt)
        elif SE_linearity == 'NL' and include_SE:
            if include_KG:
                C_tot = rayleigh_damping_matrix_func(M, K - KG + Kse_full_mat_0, damping_ratio, damping_Ti, damping_Tj)  # KG effects on C_tot are pre-determined here, but are not included in K. todo: INCLUDE OR EXCLUDE SE??
            else:
                C_tot = rayleigh_damping_matrix_func(M, K + Kse_full_mat_0, damping_ratio, damping_Ti, damping_Tj) #todo: INCLUDE OR EXCLUDE SE??
            u0 = np.array([0] * len(M))  # 0 initial displacement at every dof
            v0 = np.array([0] * len(M))  # 0 initial velocity at every dof
            read_dict = MDOF_TD_NL_wind_solver(g_node_coor, p_node_coor, beta_0, theta_0, aero_coef_method, n_aero_coef, include_KG, geometric_linearity, R_loc, D_loc, M, C_tot, K, windspeed, u0=u0, v0=v0, T=wind_T, dt=dt)

        # Results in Global coordinates XYZ.
        u_glob, v_glob, a_glob = read_dict['u'], read_dict['v'], read_dict['a']

        u_loc = np.einsum('mn,tn->tm', T_LsGs_full_2D_node_matrix, u_glob)
        v_loc = np.einsum('mn,tn->tm', T_LsGs_full_2D_node_matrix, v_glob)
        a_loc = np.einsum('mn,tn->tm', T_LsGs_full_2D_node_matrix, a_glob)

        # # Converting results to Local coordinates xyz
        # def mat_Ls_elem_Gs_elem_all_func(D_elem_glob, g_node_coor, p_node_coor):
        #     """
        #     Converts a global element displacement matrix D, to the local one. Shape (total num elem, 12).
        #     """
        #     g_node_num = len(g_node_coor)
        #     g_elem_num = g_node_num - 1
        #     n_columns = len(p_node_coor)
        #
        #     rot_mat_12b = T_LsGs_12b_func(g_node_coor)
        #     rot_mat_12c = T_LsGs_12c_func(g_node_coor, p_node_coor)
        #
        #     T_all_elem_glob_to_all_elem_loc = np.zeros((g_elem_num + n_columns, 12, 12))
        #     for el in range(g_elem_num):
        #         T_all_elem_glob_to_all_elem_loc[el] = rot_mat_12b[el]  # first nodes of the 12x12 beams
        #     for el, c in zip(range(g_elem_num, g_elem_num + n_columns), range(n_columns)):
        #         T_all_elem_glob_to_all_elem_loc[el] = rot_mat_12c[c]  # last node is the second node of the 12x12.
        #     D_elem_loc = np.einsum('nij,nj->ni', T_all_elem_glob_to_all_elem_loc, D_elem_glob)
        #     return D_elem_locu

        print("--- %s seconds for the Time Domain simulation ---" % np.round_((time.time() - start_time)))

        # Plotting
        if plots:
            plot_dof = 100*6+1  # dof to be plotted as a time history and corresponding PSD
            plt.figure()
            plt.plot(time_array, u_loc[:,plot_dof])
            plt.show()
            from scipy import signal
            f, Pxx_den = signal.welch(u_loc[:,plot_dof], fs=1/dt)
            plt.figure()
            plt.plot(f, Pxx_den)
            plt.show()
            np.savetxt(r'results\u_loc_B-' + str(np.round(deg(beta_DB))) + '_T-' + str(wind_T - transient_T) + '_dt-' + str(dt) + '_Spec-' + str(cospec_type) + \
                       '_zeta-' + str(damping_ratio) + '_Ti-' + str(damping_Ti) + '_Tj-' + str(damping_Tj) + '_Nodes-' + str(g_node_num) + '_iter-' + str(seed+1) + '.txt', u_loc)

        g_dof_X = np.array([n * 6 + 0 for n in g_nodes])
        g_dof_Y = np.array([n * 6 + 1 for n in g_nodes])
        g_dof_Z = np.array([n * 6 + 2 for n in g_nodes])
        g_dof_XX = np.array([n * 6 + 3 for n in g_nodes])
        g_dof_YY = np.array([n * 6 + 4 for n in g_nodes])
        g_dof_ZZ = np.array([n * 6 + 5 for n in g_nodes])

        # std of displacements
        time_cutoff_idx = (np.abs(time_array - transient_T)).argmin()  # finding the closest value
        std_delta_local = np.array([[np.std(u_loc[time_cutoff_idx:,d]) for d in g_dof_X],
                                    [np.std(u_loc[time_cutoff_idx:,d]) for d in g_dof_Y],
                                    [np.std(u_loc[time_cutoff_idx:,d]) for d in g_dof_Z],
                                    [np.std(u_loc[time_cutoff_idx:,d]) for d in g_dof_XX],
                                    [np.std(u_loc[time_cutoff_idx:,d]) for d in g_dof_YY],
                                    [np.std(u_loc[time_cutoff_idx:,d]) for d in g_dof_ZZ]])

        mean_delta_local = np.array([[np.mean(u_loc[time_cutoff_idx:,d]) for d in g_dof_X],
                                     [np.mean(u_loc[time_cutoff_idx:,d]) for d in g_dof_Y],
                                     [np.mean(u_loc[time_cutoff_idx:,d]) for d in g_dof_Z],
                                     [np.mean(u_loc[time_cutoff_idx:,d]) for d in g_dof_XX],
                                     [np.mean(u_loc[time_cutoff_idx:,d]) for d in g_dof_YY],
                                     [np.mean(u_loc[time_cutoff_idx:,d]) for d in g_dof_ZZ]])

        # # Plotting std of displacements
        # g_elem_nodes = g_elem_nodes_func(g_nodes)
        # g_elem_L_3D = np.array([np.linalg.norm(g_node_coor[g_elem_nodes[i, 1]] - g_node_coor[g_elem_nodes[i, 0]]) for i in range(g_elem_num)])
        # g_s_3D = np.array([0] + list(np.cumsum(g_elem_L_3D)))  # arc length along arch for each node

        # plt.figure(figsize=(10,5), dpi=200)
        # plt.title('Benchmarking: TD vs FD')
        # plt.plot(g_s_3D, std_delta_local[0], label='X - TD', color='b')
        # # plt.plot(g_s_3D, std_delta[0],    label='X - FD', linestyle="--", color='b')
        # plt.plot(g_s_3D, std_delta_local[1], label='Y - TD', color='orange')
        # # plt.plot(g_s_3D, std_delta[1],    label='Y - FD', linestyle="--", color='orange')
        # plt.plot(g_s_3D, std_delta_local[2], label='Z - TD', color='g')
        # # plt.plot(g_s_3D, std_delta[2],    label='Z - FD', linestyle="--", color='g')
        # plt.plot(g_s_3D, std_delta_local[3]*1000, label='XX - TD * 1000', color='m')
        # # plt.plot(g_s_3D, std_delta[3]*1000,    label='XX - FD * 1000', linestyle="--", color='m')
        # plt.legend(bbox_to_anchor=(1.25,1))
        # plt.xlabel('Along arc length [m]')
        # plt.ylabel('$\sigma$ of displacements [m] or rotations [rad*1000]')
        # plt.tight_layout()
        # plt.savefig(r'results\TD-vs-FD_BetaDB-' + str(deg(beta_DB)) +'_T-' + str(wind_T - transient_T) + '_dt-' + str(dt) + '_Spec-' + str(cospec_type) + \
        #             '_zeta-' + str(damping_ratio) +'_Ti-' + str(damping_Ti) +'_Tj-' + str(damping_Tj) +'_Nodes-' + str(g_node_num) + \
        #             '_FD-f-' + str(f_array[0]) +'-' + str(f_array[-1]) + '_iter-' + str(seed+1) +'.png')
        #
        # Plotting time history of displacements
        # g_node_half_idx = (np.abs(g_s_3D-arc_length/2)).argmin()  # finding the closest value
        # g_node_third_idx = (np.abs(g_s_3D-arc_length/3)).argmin()  # finding the closest value
        # g_node_fourth_idx = (np.abs(g_s_3D-arc_length/4)).argmin()  # finding the closest value
        #
        # plt.figure(figsize=(10,5), dpi=200)
        # plt.title('Time histories of displacements')
        # plt.plot(time_array, u_loc[:,g_dof_Y[g_node_half_idx]], label='L/2')
        # plt.plot(time_array, u_loc[:,g_dof_Y[g_node_third_idx]], label='L/3')
        # plt.plot(time_array, u_loc[:,g_dof_Y[g_node_fourth_idx]], label='L/4')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Global Y displacements [m]')
        # plt.xlim([0,wind_T])
        # plt.axvline(x=transient_T, label='Ramp-up time', color='r', linestyle="--")
        # plt.legend(title='Position along the arc:',bbox_to_anchor=(1.02,1))
        # plt.tight_layout()
        # plt.savefig(r'results\TD-hist_BetaDB-' + str(deg(beta_DB)) +'_T-' + str(wind_T - transient_T) + '_dt-' + str(dt) + '_Spec-' + str(cospec_type) + \
        #             '_zeta-' + str(damping_ratio) +'_Ti-' + str(damping_Ti) +'_Tj-' + str(damping_Tj) +'_Nodes-' + str(g_node_num) + \
        #             '_FD-f-' + str(f_array[0]) +'-' + str(f_array[-1]) + '_iter-' + str(seed+1) + '.png')

        std_delta_local_all_seeds.append(std_delta_local)
        # Saving std of displacements
        if save_txt:
            np.savetxt(r'results\std_delta_local_TD_all_seeds_B-' + str(np.round(deg(beta_DB))) + '_T-' + str(wind_T - transient_T) + '_dt-' + str(dt) + '_Spec-' + str(cospec_type) + \
                       '_zeta-' + str(damping_ratio) + '_Ti-' + str(damping_Ti) + '_Tj-' + str(damping_Tj) + '_Nodes-' + str(g_node_num) + '_iter-' + str(seed+1) + '.txt', std_delta_local)
            np.savetxt(r'results\mean_delta_local_TD_all_seeds_B-' + str(np.round(deg(beta_DB))) + '_T-' + str(wind_T - transient_T) + '_dt-' + str(dt) + '_Spec-' + str(cospec_type) + \
                       '_zeta-' + str(damping_ratio) + '_Ti-' + str(damping_Ti) + '_Tj-' + str(damping_Tj) + '_Nodes-' + str(g_node_num) + '_iter-' + str(seed+1) + '.txt', mean_delta_local)

    std_delta_local_mean = np.mean(std_delta_local_all_seeds, axis=0)
    std_delta_local_std = np.std(std_delta_local_all_seeds, axis=0)

    return {'std_delta_local_mean': std_delta_local_mean,
            'std_delta_local_std': std_delta_local_std,
            'cospec_type':cospec_type,
            'damping_ratio':damping_ratio,
            'damping_Ti':damping_Ti,
            'damping_Tj':damping_Tj}

def list_of_cases_TD_func(aero_coef_method_cases, n_aero_coef_cases, include_SE_cases, flutter_derivatives_type_cases, n_nodes_cases, include_sw_cases, include_KG_cases, n_seeds_cases, dt_cases,
                          aero_coef_linearity_cases, SE_linearity_cases, geometric_linearity_cases, skew_approach_cases, beta_DB_cases):
    # List of cases (parameter combinations) to be run:
    list_of_cases = [(a,b,c,d,e,f,g,h,i,j,k,l,m,z) for a in aero_coef_method_cases for b in n_aero_coef_cases for c in include_SE_cases
                     for d in flutter_derivatives_type_cases for e in n_nodes_cases for f in include_sw_cases for g in include_KG_cases
                     for h in n_seeds_cases for i in dt_cases for j in aero_coef_linearity_cases for k in SE_linearity_cases
                     for l in geometric_linearity_cases for m in skew_approach_cases for z in beta_DB_cases] # Note: new parameters should be added before beta_DB
    list_of_cases = [list(case) for case in list_of_cases
                     if not (('3D' in case[3] and '2D' in case[11]) or ('2D' in case[3] and '3D' in case[11]) or (case[2]==False and case[3] in ['3D_Scanlan', '3D_Scanlan_confirm', '3D_Zhu', '3D_Zhu_bad_P5', '2D_in_plane']))]
    # if skew_approach is '3D' only '3D' flutter_derivatives accepted. If SE=False, only one dummy FD case is accepted: '3D_full' or '2D_full'
    return list_of_cases

def parametric_buffeting_TD_func(list_of_cases, g_node_coor, p_node_coor, Ii_simplified, wind_block_T, wind_overlap_T,
                      wind_T, transient_T, ramp_T, R_loc, D_loc, cospec_type=2, plots=False, save_txt=False):
    # Empty Dataframe to store results
    results_df = pd.DataFrame(list_of_cases)
    results_df.columns = ['Method', 'n_aero_coef', 'SE', 'FD_type', 'g_node_num', 'SWind', 'KG', 'N_seeds', 'dt', 'C_Ci_linearity', 'SE_linearity', 'geometric_linearity', 'skew_approach', 'beta_DB']
    for i in range(0, 6):
        results_df['std_max_dof_' + str(i)] = None

    case_idx = -1  # index of the case
    for aero_coef_method, n_aero_coef, include_SE, flutter_derivatives_type, g_node_num, include_sw, include_KG, n_seeds, dt, aero_coef_linearity,\
        SE_linearity, geometric_linearity, skew_approach, beta_DB in list_of_cases:
        case_idx += 1  # starts at 0.
        buffeting_results = buffeting_TD_func(aero_coef_method, skew_approach, n_aero_coef, include_SE, flutter_derivatives_type, include_sw, include_KG, g_node_coor, p_node_coor, Ii_simplified, R_loc, D_loc,
                                              n_seeds, dt, wind_block_T, wind_overlap_T, wind_T, transient_T, ramp_T, beta_DB, aero_coef_linearity, SE_linearity, geometric_linearity,
                                              cospec_type, plots, save_txt)
        # Reading results
        std_delta_local_mean = buffeting_results['std_delta_local_mean']
        std_delta_local_std = buffeting_results['std_delta_local_std']
        cospec_type = buffeting_results['cospec_type']
        damping_ratio = buffeting_results['damping_ratio']
        damping_Ti = buffeting_results['damping_Ti']
        damping_Tj = buffeting_results['damping_Tj']
        # Writing results
        results_df.at[case_idx, 'cospec_type'] = cospec_type
        results_df.at[case_idx, 'damping_ratio'] = damping_ratio
        results_df.at[case_idx, 'damping_Ti'] = damping_Ti
        results_df.at[case_idx, 'damping_Tj'] = damping_Tj
        for i in range(0,6):
            results_df.at[case_idx, 'std_max_dof_'+str(i)] = np.max(std_delta_local_mean[i])
            results_df.at[case_idx, 'std_std_max_dof_'+str(i)] = np.max(std_delta_local_std[i])
        print('beta_DB = ' + str(round(deg(beta_DB))))

    # Exporting the results to a table
    from time import gmtime, strftime
    results_df.to_csv(r'results\TD_std_delta_max_'+strftime("%Y-%m-%d_%H-%M-%S", gmtime())+'.csv')
    return None
