"""
created: 2021
author: Bernardo Costa
email: bernamdc@gmail.com

"Nw" is short for "Nonhomogeneous wind" and is extensively used in this script
"""

import numpy as np
from scipy import interpolate
from buffeting import RP, U_bar_func
from simple_5km_bridge_geometry import g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func
from simple_5km_bridge_geometry import g_node_coor, p_node_coor, g_node_coor_func  # todo: delete?
import matplotlib.pyplot as plt

n_WRF_nodes = 11

def interpolate_from_WRF_nodes_to_g_nodes(g_node_coor, WRF_node_coor, WRF_node_func, plot=False):
    """
    Linear interpolation of a function known at the WRF_nodes, estimated at the g_nodes, assuming all nodes follow the same arc, along which the 1D interpolation dist is calculated
    This interpolation is made in 1D, along the along-arc distance s, otherwise the convex hull of WRF_nodes would not encompass the g_nodes and 2D extrapolations are not efficient / readily available
    """
    # Make sure the first and last g_nodes and WRF_nodes are positioned in the same place
    assert np.allclose(g_node_coor[0], WRF_node_coor[0])
    assert np.allclose(g_node_coor[-1], WRF_node_coor[-1])
    n_WRF_nodes = len(WRF_node_coor)
    n_g_nodes   = len(  g_node_coor)
    WRF_node_s = np.linspace(0, arc_length, n_WRF_nodes)
    g_node_s   = np.linspace(0, arc_length,   n_g_nodes)
    func = interpolate.interp1d(x=WRF_node_s, y=WRF_node_func, kind='linear')
    if plot:
        plt.scatter(g_node_coor[:, 0], g_node_coor[:, 1])
        plt.scatter(WRF_node_coor[:, 0], WRF_node_coor[:, 1], alpha=0.5, s=100)
        plt.axis('equal')
        plt.show()
    return func(g_node_s)


def Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])):
    """ 10min mean wind """  #
    n_g_nodes = len(g_node_coor)
    U_bar_10min = U_bar_func(g_node_coor)


    assert R==5000 and arc_length==5000, "Make sure that the R and arc length are consistent in: 1) the bridge model and 2) WRF nodes (the arc along which WRF data is collected)"
    WRF_node_coor = g_node_coor_func(R=R, arc_length=arc_length, pontoons_s=[], zbridge=zbridge, FEM_max_length=arc_length/(n_WRF_nodes-1), bridge_shape=bridge_shape)
    Nw_shape_norm  # normalized to have average = 1



    return


def static_wind_func(g_node_coor, p_node_coor, alpha, beta_DB, theta_0, aero_coef_method, n_aero_coef, skew_approach):
    """
    :return: New girder and gontoon node coordinates, as well as the displacements that led to them.
    """
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)
    beta_0 = beta_0_func(beta_DB)
    U_bar = U_bar_func(g_node_coor)
    beta_bar, theta_bar = beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha)
    stiff_matrix = stiff_matrix_func(g_node_coor, p_node_coor, alpha)  # Units: (N)
    Pb = Pb_func(g_node_coor, beta_bar, theta_bar, alpha, aero_coef_method, n_aero_coef, skew_approach, Chi_Ci='ones')
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

def nonhomogeneous_wind_func():
    Nw_beta_bar, Nw_theta_bar = Nw_beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha)
