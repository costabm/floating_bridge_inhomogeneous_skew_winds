"""
created: 2021
author: Bernardo Costa
email: bernamdc@gmail.com

"Nw" is short for "Nonhomogeneous wind" and is extensively used in this script
"""

import numpy as np
from scipy import interpolate
from buffeting import U_bar_func, beta_0_func
from simple_5km_bridge_geometry import g_node_coor, g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func
from transformations import T_LsGs_3g_func, T_GsGw_func, from_cos_sin_to_0_2pi
from WRF_500_interpolated.create_minigrid_data_from_raw_WRF_500_data import n_bridge_WRF_nodes, bridge_WRF_nodes_coor_func, earth_R
import matplotlib.pyplot as plt


n_WRF_nodes = n_bridge_WRF_nodes
WRF_node_coor = g_node_coor_func(R=R, arc_length=arc_length, pontoons_s=[], zbridge=zbridge, FEM_max_length=arc_length/(n_WRF_nodes-1), bridge_shape=bridge_shape)  # needs to be calculated


# Testing consistency between WRF nodes in bridge coordinates and in (lats,lons)
test_WRF_node_consistency = True
if test_WRF_node_consistency: # Make sure that the R and arc length are consistent in: 1) the bridge model and 2) WRF nodes (the arc along which WRF data is collected)
    assert (R==5000 and arc_length==5000)
    WRF_node_coor_2 = np.deg2rad(bridge_WRF_nodes_coor_func()) * earth_R
    WRF_node_coor_2[:, 1] = -WRF_node_coor_2[:, 1]  # attention! bridge_WRF_nodes_coor_func() gives coor in (lats,lons) which is a left-hand system! This converts to right-hand (lats,-lons).
    WRF_node_coor_2 = (WRF_node_coor_2 - WRF_node_coor_2[0]) @ np.array([[np.cos(np.deg2rad(-10)), -np.sin(np.deg2rad(-10))], [np.sin(np.deg2rad(-10)), np.cos(np.deg2rad(-10))]])
    assert np.allclose(WRF_node_coor[:, :2], WRF_node_coor_2)


def interpolate_from_WRF_nodes_to_g_nodes(WRF_node_func, g_node_coor=g_node_coor, WRF_node_coor=WRF_node_coor, plot=False):
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


# todo: delete below
from create_WRF_data_at_bridge_nodes_from_minigrid_data import wd_to_plot, ws_to_plot
Nw_beta_DB_cos = interpolate_from_WRF_nodes_to_g_nodes(np.cos(wd_to_plot, dtype=float))
Nw_beta_DB_sin = interpolate_from_WRF_nodes_to_g_nodes(np.sin(wd_to_plot, dtype=float))
Nw_beta_DB = from_cos_sin_to_0_2pi(Nw_beta_DB_cos, Nw_beta_DB_sin, out_units='rad')
Nw_beta_0 = np.array([beta_0_func(i) for i in Nw_beta_DB])
print(np.rad2deg(Nw_beta_0))
Nw_theta_0 = Nw_beta_0 * 0
alpha = Nw_theta_0
# todo: delete above


def Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes, force_Nw_U_bar_and_U_bar_to_have_same='energy'):
    """
    Returns a vector of Nonhomogeneous mean wind at each of the g_nodes
    force_Nw_and_U_bar_to_have_same_avg : None or '', 'mean', 'energy'. force the Nw_U_bar_at_WRF_nodes to have the same e.g. mean 1, and thus when multiplied with U_bar, the result will have the same mean (of all nodes) wind
    """
    assert len(Nw_U_bar_at_WRF_nodes) == n_WRF_nodes
    U_bar_10min = U_bar_func(g_node_coor)
    interp_fun = interpolate_from_WRF_nodes_to_g_nodes(Nw_U_bar_at_WRF_nodes)
    if force_Nw_U_bar_and_U_bar_to_have_same =='mean':
        Nw_U_bar = U_bar_10min *        ( interp_fun / np.mean(interp_fun) )
        assert np.isclose(np.mean(Nw_U_bar), np.mean(U_bar_10min))        # same mean(U)
    elif force_Nw_U_bar_and_U_bar_to_have_same =='energy':
        Nw_U_bar = U_bar_10min * np.sqrt( interp_fun / np.mean(interp_fun) )
        assert np.isclose(np.mean(Nw_U_bar**2), np.mean(U_bar_10min**2))  # same energy = same mean(U**2)
    else:
        Nw_U_bar = interp_fun
    return Nw_U_bar


# Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes=ws_to_plot, force_Nw_U_bar_and_U_bar_to_have_same=None)


def Nw_beta_and_theta_bar_func(g_node_coor, Nw_beta_0, Nw_theta_0, alpha):
    """Returns the Nonhomogeneous beta_bar and theta_bar at each node, relative to the mean of the axes of the adjacent elements.
    Note: the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
    n_g_nodes = len(g_node_coor)
    assert len(Nw_beta_0) == len(Nw_theta_0) == n_g_nodes
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
    T_GsNw = np.array([T_GsGw_func(Nw_beta_0[i], Nw_theta_0[i]) for i in range(n_g_nodes)])
    T_LsNw = np.einsum('nij,njk->nik', T_LsGs, T_GsNw)
    U_Gw_norm = np.array([1, 0, 0])  # U_Gw = (U, 0, 0), so normalized is (1, 0, 0)
    U_Ls = np.einsum('nij,j->ni', T_LsNw, U_Gw_norm)
    # todo: I hope you had a nice weekend bernardo, continue here :)


    ######## GET INSPIRATION FROM BELOW
    def beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha):
        """Returns the beta_bar and theta_bar at each node, as a mean of adjacent elements.
        Note that the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
        T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
        T_GsGw = T_GsGw_func(beta_0, theta_0)
        T_LsGw = np.einsum('nij,jk->nik', T_LsGs, T_GsGw)
        U_bar = U_bar_func(g_node_coor, RP=RP)
        U_Gw = np.array([U_bar, np.zeros(U_bar.shape), np.zeros(U_bar.shape)]).T  # todo: delete line above and make this line np.array([1,0,0])? should give same results...
        U_Ls = np.einsum('nij,nj->ni', T_LsGw, U_Gw)
        Ux = U_Ls[:, 0]
        Uy = U_Ls[:, 1]
        Uz = U_Ls[:, 2]
        Uxy = np.sqrt(Ux ** 2 + Uy ** 2)
        beta_bar = np.array([-np.arccos(Uy[i] / Uxy[i]) * np.sign(Ux[i]) for i in range(len(g_node_coor))])
        theta_bar = np.array([np.arcsin(Uz[i] / U_bar[i]) for i in range(len(g_node_coor))])
        return beta_bar, theta_bar
    ###################

def nonhomogeneous_wind_func():  # todo: get inspiration in the static_wind_func to do this
    Nw_beta_bar, Nw_theta_bar = Nw_beta_and_theta_bar_func(g_node_coor, beta_0, theta_0, alpha)
