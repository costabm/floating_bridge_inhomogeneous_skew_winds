# -*- coding: utf-8 -*-
"""
Updated: 04/2020
author: Bernardo Costa
email: bernamdc@gmail.com

Calculates the effects from static loads, such as mean wind
"""

import numpy as np
from simple_5km_bridge_geometry import pontoons_s
from mass_and_stiffness_matrix import stiff_matrix_func, stiff_matrix_12b_local_func, stiff_matrix_12c_local_func, linmass, SDL
from transformations import g_node_L_3D_func, mat_Gs_elem_Gs_node_all_func, \
    mat_Ls_elem_Gs_elem_all_func
from buffeting import U_bar_func, beta_and_theta_bar_func, Pb_func, beta_0_func
import copy


def R_loc_func(D_node_glob, g_node_coor, p_node_coor, alpha):
    """
    Internal local resultant forces and moments for each element of all the girder and columns. Shape (total num elem, 12).
    """
    stiff_matrix_12b_loc = stiff_matrix_12b_local_func(g_node_coor)
    stiff_matrix_12c_loc = stiff_matrix_12c_local_func(p_node_coor)
    stiff_matrix_all_loc = np.concatenate((stiff_matrix_12b_loc, stiff_matrix_12c_loc), axis=0)
    D_elem_glob = mat_Gs_elem_Gs_node_all_func(D_node_glob, g_node_coor, p_node_coor)
    D_elem_loc = mat_Ls_elem_Gs_elem_all_func(D_elem_glob, g_node_coor, p_node_coor, alpha)
    R_all_elem_loc = np.einsum('nij,nj->ni', stiff_matrix_all_loc, D_elem_loc)
    return R_all_elem_loc


def static_wind_from_U_beta_theta_bar(g_node_coor, p_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach):
    """
    :return: New girder and gontoon node coordinates, as well as the displacements that led to them.
    """
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)
    stiff_matrix = stiff_matrix_func(g_node_coor, p_node_coor, alpha)  # Units: (N)
    Pb = Pb_func(g_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach, Chi_Ci='ones')
    sw_vector = np.array([U_bar, np.zeros(len(U_bar)), np.zeros(len(U_bar))])  # instead of a=(u,v,w) a vector (U,0,0) is used.
    F_sw = np.einsum('ndi,in->nd', Pb, sw_vector) / 2  # Global buffeting force vector. See Paper from LD Zhu, eq. (24). Units: (N)
    F_sw_flat = np.ndarray.flatten(F_sw)  # flattening
    F_sw_flat = np.array(list(F_sw_flat) + [0]*len(p_node_coor)*6)  # adding 0 force to all the remaining pontoon DOFs
    # Global nodal Displacement matrix
    D_sw_flat = np.linalg.inv(stiff_matrix) @ F_sw_flat
    D_glob_sw = np.reshape(D_sw_flat, (g_node_num + p_node_num, 6))
    g_node_coor_sw = g_node_coor + D_glob_sw[:g_node_num,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
    p_node_coor_sw = p_node_coor + D_glob_sw[g_node_num:,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
    return g_node_coor_sw, p_node_coor_sw, D_glob_sw


def static_wind_func(g_node_coor, p_node_coor, alpha, U_bar, beta_DB, theta_0, aero_coef_method, n_aero_coef, skew_approach):
    """
    :return: New girder and gontoon node coordinates, as well as the displacements that led to them.
    """
    beta_0 = beta_0_func(beta_DB)
    beta_bar, theta_bar = beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha)
    g_node_coor_sw, p_node_coor_sw, D_glob_sw = static_wind_from_U_beta_theta_bar(g_node_coor, p_node_coor, alpha, U_bar, beta_bar, theta_bar, aero_coef_method, n_aero_coef, skew_approach)
    return g_node_coor_sw, p_node_coor_sw, D_glob_sw


def static_dead_loads_func(g_node_coor, p_node_coor, alpha):
    """
    :return: New girder and gontoon node coordinates, as well as the displacements that led to them.
    """
    spans = copy.deepcopy(pontoons_s)
    spans[1:] -= copy.deepcopy(spans[:-1])  # x[1:]-= x[:-1] is inverse of np.cumsum. Retrieves spans from pontoons_s
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)
    # Dead loads, at each girder node
    DL = (linmass + SDL)  # N/m.
    g_elem_L_3D = g_node_L_3D_func(g_node_coor)
    F_DL = np.zeros((g_node_num, 6))
    F_DL[:,2] = -DL*g_elem_L_3D  # downwards
    # Buoyancy forces, at each pontoon
    buoyancy = np.zeros((p_node_num, 6))
    buoyancy[:,2] = DL * spans  # upwards
    F_DL_and_buoy = np.concatenate((F_DL, buoyancy), axis=0)
    F_DL_and_buoy_flat = np.ndarray.flatten(F_DL_and_buoy)
    # Stiffness matrix
    stiff_matrix = stiff_matrix_func(g_node_coor, p_node_coor, alpha)  # Units: (N).
    # Global nodal displacement matrix
    D_DL_and_buoy_flat = np.linalg.inv(stiff_matrix) @ F_DL_and_buoy_flat
    D_glob_DL_and_buoy = np.reshape(D_DL_and_buoy_flat, (g_node_num + p_node_num, 6))
    g_node_coor_sw = g_node_coor + D_glob_DL_and_buoy[:g_node_num,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
    p_node_coor_sw = p_node_coor + D_glob_DL_and_buoy[g_node_num:,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
    return g_node_coor_sw, p_node_coor_sw, D_glob_DL_and_buoy


