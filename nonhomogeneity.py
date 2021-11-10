"""
created: 2021
author: Bernardo Costa
email: bernamdc@gmail.com

"Nw" is short for "Nonhomogeneous wind" and is extensively used in this script
"""

import numpy as np
from scipy import interpolate
from buffeting import U_bar_func, beta_0_func, RP, Pb_func
from mass_and_stiffness_matrix import stiff_matrix_func, stiff_matrix_12b_local_func, stiff_matrix_12c_local_func, linmass, SDL
from simple_5km_bridge_geometry import g_node_coor, p_node_coor, g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func
from transformations import T_LsGs_3g_func, T_GsGw_func, from_cos_sin_to_0_2pi
from WRF_500_interpolated.create_minigrid_data_from_raw_WRF_500_data import n_bridge_WRF_nodes, bridge_WRF_nodes_coor_func, earth_R
import matplotlib.pyplot as plt


n_WRF_nodes = n_bridge_WRF_nodes
WRF_node_coor = g_node_coor_func(R=R, arc_length=arc_length, pontoons_s=[], zbridge=zbridge, FEM_max_length=arc_length/(n_WRF_nodes-1), bridge_shape=bridge_shape)  # needs to be calculated


# # Testing consistency between WRF nodes in bridge coordinates and in (lats,lons)
# test_WRF_node_consistency = True
# if test_WRF_node_consistency: # Make sure that the R and arc length are consistent in: 1) the bridge model and 2) WRF nodes (the arc along which WRF data is collected)
#     assert (R==5000 and arc_length==5000)
#     WRF_node_coor_2 = np.deg2rad(bridge_WRF_nodes_coor_func()) * earth_R
#     WRF_node_coor_2[:, 1] = -WRF_node_coor_2[:, 1]  # attention! bridge_WRF_nodes_coor_func() gives coor in (lats,lons) which is a left-hand system! This converts to right-hand (lats,-lons).
#     WRF_node_coor_2 = (WRF_node_coor_2 - WRF_node_coor_2[0]) @ np.array([[np.cos(np.deg2rad(-10)), -np.sin(np.deg2rad(-10))], [np.sin(np.deg2rad(-10)), np.cos(np.deg2rad(-10))]])
#     assert np.allclose(WRF_node_coor[:, :2], WRF_node_coor_2)


def interpolate_from_WRF_nodes_to_g_nodes(WRF_node_func, g_node_coor, WRF_node_coor, plot=False):
    """
    Linear interpolation of a function known at the WRF_nodes, estimated at the g_nodes, assuming all nodes follow the same arc, along which the 1D interpolation dist is calculated
    This interpolation is made in 1D, along the along-arc distance s, otherwise the convex hull of WRF_nodes would not encompass the g_nodes and 2D extrapolations are not efficient / readily available
    """
    # Make sure the first and last g_nodes and WRF_nodes are positioned in the same place
    assert np.allclose(g_node_coor[0], WRF_node_coor[0])
    assert np.allclose(g_node_coor[-1], WRF_node_coor[-1])
    if plot:
        plt.scatter(g_node_coor[:, 0], g_node_coor[:, 1])
        plt.scatter(WRF_node_coor[:, 0], WRF_node_coor[:, 1], alpha=0.5, s=100)
        plt.axis('equal')
        plt.show()
    n_WRF_nodes = len(WRF_node_coor)
    n_g_nodes   = len(  g_node_coor)
    WRF_node_s = np.linspace(0, arc_length, n_WRF_nodes)
    g_node_s   = np.linspace(0, arc_length,   n_g_nodes)
    func = interpolate.interp1d(x=WRF_node_s, y=WRF_node_func, kind='linear')
    return func(g_node_s)


# # todo: delete below
# import copy
# from create_WRF_data_at_bridge_nodes_from_minigrid_data import wd_to_plot, ws_to_plot
# Nw_beta_DB_cos = interpolate_from_WRF_nodes_to_g_nodes(np.cos(wd_to_plot, dtype=float))
# Nw_beta_DB_sin = interpolate_from_WRF_nodes_to_g_nodes(np.sin(wd_to_plot, dtype=float))
# Nw_beta_DB = from_cos_sin_to_0_2pi(Nw_beta_DB_cos, Nw_beta_DB_sin, out_units='rad')
# Nw_beta_0 = np.array([beta_0_func(i) for i in Nw_beta_DB])
# print(np.rad2deg(Nw_beta_0))
# Nw_theta_0 = (copy.deepcopy(Nw_beta_0) * 0 + 1) * np.deg2rad(0)
# alpha = (copy.deepcopy(Nw_beta_0) * 0 + 1) *  np.deg2rad(0)
# # todo: delete above


def Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes, force_Nw_U_and_N400_U_to_have_same=None):
    """
    Returns a vector of Nonhomogeneous mean wind at each of the g_nodes
    force_Nw_and_U_bar_to_have_same_avg : None, 'mean', 'energy'. force the Nw_U_bar_at_WRF_nodes to have the same e.g. mean 1, and thus when multiplied with U_bar, the result will have the same mean (of all nodes) wind
    """
    assert Nw_U_bar_at_WRF_nodes.shape[-1] == n_WRF_nodes
    U_bar_10min = U_bar_func(g_node_coor)
    interp_fun = interpolate_from_WRF_nodes_to_g_nodes(Nw_U_bar_at_WRF_nodes, g_node_coor, WRF_node_coor)
    if force_Nw_U_and_N400_U_to_have_same == 'mean':
        Nw_U_bar = U_bar_10min *        ( interp_fun / np.mean(interp_fun) )
        assert np.isclose(np.mean(Nw_U_bar), np.mean(U_bar_10min))        # same mean(U)
    elif force_Nw_U_and_N400_U_to_have_same == 'energy':
        Nw_U_bar = U_bar_10min * np.sqrt( interp_fun / np.mean(interp_fun) )
        assert np.isclose(np.mean(Nw_U_bar**2), np.mean(U_bar_10min**2))  # same energy = same mean(U**2)
    else:
        Nw_U_bar = interp_fun
    return Nw_U_bar

# Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes=ws_to_plot, force_Nw_U_bar_and_U_bar_to_have_same=None)

def U_bar_equivalent_to_Nw_U_bar(g_node_coor, Nw_U_bar, force_Nw_U_bar_and_U_bar_to_have_same='energy'):
    """
    Returns a homogeneous wind velocity field, equivalent to the input Nw_U_bar in terms of force_Nw_U_bar_and_U_bar_to_have_same
    force_Nw_U_bar_and_U_bar_to_have_same: None, 'mean', 'energy'. force the U_bar_equivalent to have the same mean or energy 1 as Nw_U_bar
    """
    if force_Nw_U_bar_and_U_bar_to_have_same is None:
        U_bar_equivalent = U_bar_func(g_node_coor)
    elif force_Nw_U_bar_and_U_bar_to_have_same == 'mean':
        U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.mean(Nw_U_bar)
        assert np.isclose(np.mean(Nw_U_bar), np.mean(U_bar_equivalent))
    elif force_Nw_U_bar_and_U_bar_to_have_same == 'energy':
        U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.sqrt(np.mean(Nw_U_bar**2))
        assert np.isclose(np.mean(Nw_U_bar ** 2), np.mean(U_bar_equivalent ** 2))  # same energy = same mean(U**2)
    return U_bar_equivalent


def Nw_beta_and_theta_bar_func(g_node_coor, Nw_beta_0, Nw_theta_0, alpha):
    """Returns the Nonhomogeneous beta_bar and theta_bar at each node, relative to the mean of the axes of the adjacent elements.
    Note: the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
    n_g_nodes = len(g_node_coor)
    assert len(Nw_beta_0) == len(Nw_theta_0) == n_g_nodes
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
    T_GsNw = np.array([T_GsGw_func(Nw_beta_0[i], Nw_theta_0[i]) for i in range(n_g_nodes)])
    T_LsNw = np.einsum('nij,njk->nik', T_LsGs, T_GsNw)
    U_Gw_norm = np.array([1, 0, 0])  # U_Gw = (U, 0, 0), so the normalized U_Gw_norm is (1, 0, 0)
    U_Ls = np.einsum('nij,j->ni', T_LsNw, U_Gw_norm)
    Ux = U_Ls[:, 0]
    Uy = U_Ls[:, 1]
    Uz = U_Ls[:, 2]
    Uxy = np.sqrt(Ux ** 2 + Uy ** 2)
    Nw_beta_bar = np.array([-np.arccos(Uy[i] / Uxy[i]) * np.sign(Ux[i]) for i in range(len(g_node_coor))])
    Nw_theta_bar = np.array([np.arcsin(Uz[i] / 1) for i in range(len(g_node_coor))])
    return Nw_beta_bar, Nw_theta_bar


def Nw_static_wind_func(g_node_coor, p_node_coor, alpha, Nw_U_bar, Nw_beta_0, Nw_theta_0, aero_coef_method='2D_fit_cons', n_aero_coef=6, skew_approach='3D'):
    """
    :return: New girder and gontoon node coordinates, as well as the displacements that led to them.
    """
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)
    Nw_beta_bar, Nw_theta_bar = Nw_beta_and_theta_bar_func(g_node_coor, Nw_beta_0, Nw_theta_0, alpha)
    stiff_matrix = stiff_matrix_func(g_node_coor, p_node_coor, alpha)  # Units: (N)
    Pb = Pb_func(g_node_coor, Nw_beta_bar, Nw_theta_bar, alpha, aero_coef_method, n_aero_coef, skew_approach, Chi_Ci='ones')
    sw_vector = np.array([Nw_U_bar, np.zeros(len(Nw_U_bar)), np.zeros(len(Nw_U_bar))])  # instead of a=(u,v,w) a vector (U,0,0) is used.
    F_sw = np.einsum('ndi,in->nd', Pb, sw_vector) / 2  # Global buffeting force vector. See Paper from LD Zhu, eq. (24). Units: (N)
    F_sw_flat = np.ndarray.flatten(F_sw)  # flattening
    F_sw_flat = np.array(list(F_sw_flat) + [0]*len(p_node_coor)*6)  # adding 0 force to all the remaining pontoon DOFs
    # Global nodal Displacement matrix
    D_sw_flat = np.linalg.inv(stiff_matrix) @ F_sw_flat
    D_glob_sw = np.reshape(D_sw_flat, (g_node_num + p_node_num, 6))
    g_node_coor_sw = g_node_coor + D_glob_sw[:g_node_num,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
    p_node_coor_sw = p_node_coor + D_glob_sw[g_node_num:,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
    return g_node_coor_sw, p_node_coor_sw, D_glob_sw


