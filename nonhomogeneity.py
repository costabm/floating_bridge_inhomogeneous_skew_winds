"""
created: 2021
author: Bernardo Costa
email: bernamdc@gmail.com

"Nw" is short for "Nonhomogeneous wind"
"""

import os
import copy
import datetime
import json
import netCDF4
import warnings
import numpy as np
import pandas as pd
from windrose import WindroseAxes
from scipy import interpolate
from buffeting import U_bar_func, beta_0_func, RP, Pb_func, Ai_func, iLj_func, Cij_func, beta_and_theta_bar_func
from mass_and_stiffness_matrix import stiff_matrix_func, stiff_matrix_12b_local_func, stiff_matrix_12c_local_func, linmass, SDL
from simple_5km_bridge_geometry import g_node_coor, p_node_coor, g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func
from transformations import T_LsGs_3g_func, T_GsNw_func, from_cos_sin_to_0_2pi, T_xyzXYZ_ndarray, mat_6_Ls_node_12_Ls_elem_girder_func, mat_Ls_node_Gs_node_all_func, beta_within_minus_Pi_and_Pi_func
from static_loads import static_wind_from_U_beta_theta_bar, R_loc_func
from WRF_500_interpolated.create_minigrid_data_from_raw_WRF_500_data import n_bridge_WRF_nodes, bridge_WRF_nodes_coor_func, earth_R, lat_lon_aspect_ratio
from other.orography import get_all_geotiffs_merged
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib


n_WRF_nodes = n_bridge_WRF_nodes
WRF_node_coor = g_node_coor_func(R=R, arc_length=arc_length, pontoons_s=[], zbridge=zbridge, FEM_max_length=arc_length/(n_WRF_nodes-1), bridge_shape=bridge_shape)  # needs to be calculated

def interpolate_from_WRF_nodes_to_g_nodes(WRF_node_func, g_node_coor, WRF_node_coor, plot=False):
    """
    input:
    WRF_node_func.shape == (n_cases, n_WRF_nodes)
    output: shape (n_cases, n_g_nodes)
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

# # Testing consistency between WRF nodes in bridge coordinates and in (lats,lons). The circunference to calculate lat. distances (earth_R) is larger than the circunference to calculate lon. distances
# test_WRF_node_consistency = True
# if test_WRF_node_consistency: # Make sure that the R and arc length are consistent in: 1) the bridge model and 2) WRF nodes (the arc along which WRF data is collected)
#     assert (R==5000 and arc_length==5000)
#     WRF_node_coor_2 = np.deg2rad(bridge_WRF_nodes_coor_func()) * np.array([earth_R, 1/lat_lon_aspect_ratio*earth_R])[None,:]
#     WRF_node_coor_2[:, 1] = -WRF_node_coor_2[:, 1]  # attention! bridge_WRF_nodes_coor_func() gives coor in (lats,lons) which is a left-hand system! This converts to right-hand (lats,-lons).
#     WRF_node_coor_2 = (WRF_node_coor_2 - WRF_node_coor_2[0]) @ np.array([[np.cos(np.deg2rad(-10)), -np.sin(np.deg2rad(-10))], [np.sin(np.deg2rad(-10)), np.cos(np.deg2rad(-10))]])
#     assert np.allclose(WRF_node_coor[:, :2], WRF_node_coor_2)


# todo: see if we can replace the following 3 functions with the NwOneCase:
# def Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes, force_Nw_U_and_N400_U_to_have_same=None):
#     """
#     Returns a vector of Nonhomogeneous mean wind at each of the g_nodes
#     force_Nw_and_U_bar_to_have_same_avg : None, 'mean', 'energy'. force the Nw_U_bar_at_WRF_nodes to have the same e.g. mean 1, and thus when multiplied with U_bar, the result will have the same mean (of all nodes) wind
#     """
#     assert Nw_U_bar_at_WRF_nodes.shape[-1] == n_WRF_nodes
#     U_bar_10min = U_bar_func(g_node_coor)  # N400
#     interp_fun = interpolate_from_WRF_nodes_to_g_nodes(Nw_U_bar_at_WRF_nodes, g_node_coor, WRF_node_coor)
#     if force_Nw_U_and_N400_U_to_have_same == 'mean':
#         Nw_U_bar = U_bar_10min *        ( interp_fun / np.mean(interp_fun) )
#         assert np.isclose(np.mean(Nw_U_bar), np.mean(U_bar_10min))        # same mean(U)
#     elif force_Nw_U_and_N400_U_to_have_same == 'energy':
#         Nw_U_bar = U_bar_10min * np.sqrt( interp_fun / np.mean(interp_fun) )
#         assert np.isclose(np.mean(Nw_U_bar**2), np.mean(U_bar_10min**2))  # same energy = same mean(U**2)
#     else:
#         Nw_U_bar = interp_fun
#     return Nw_U_bar
# # Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes=ws_to_plot, force_Nw_U_bar_and_U_bar_to_have_same=None)
#
# def U_bar_equivalent_to_Nw_U_bar(g_node_coor, Nw_U_bar, force_Nw_U_bar_and_U_bar_to_have_same='energy'):
#     """
#     Nw_U_bar shape: (n_cases, n_nodes)
#     Returns a homogeneous wind velocity field, equivalent to the input Nw_U_bar in terms of force_Nw_U_bar_and_U_bar_to_have_same
#     force_Nw_U_bar_and_U_bar_to_have_same: None, 'mean', 'energy'. force the U_bar_equivalent to have the same mean or energy 1 as Nw_U_bar
#     """
#     if force_Nw_U_bar_and_U_bar_to_have_same is None:
#         U_bar_equivalent = U_bar_func(g_node_coor)
#     elif force_Nw_U_bar_and_U_bar_to_have_same == 'mean':
#         U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.mean(Nw_U_bar, axis=1)[:,None]
#         assert all(np.isclose(np.mean(Nw_U_bar, axis=1)[:,None], np.mean(U_bar_equivalent, axis=1)[:,None]))
#     elif force_Nw_U_bar_and_U_bar_to_have_same == 'energy':
#         U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.sqrt(np.mean(Nw_U_bar**2, axis=1)[:,None])
#         assert all(np.isclose(np.mean(Nw_U_bar ** 2, axis=1)[:,None], np.mean(U_bar_equivalent ** 2, axis=1)[:,None]))  # same energy = same mean(U**2))
#     return U_bar_equivalent

# # DELETE THIS FUNCTION
# def Nw_static_wind_one(g_node_coor, p_node_coor, alpha, Nw_U_bar, Nw_beta_bar, Nw_theta_bar, aero_coef_method='2D_fit_cons', n_aero_coef=6, skew_approach='3D'):
#     """
#     One Nw case only
#     :return: New girder and pontoon node coordinates, as well as the displacements that led to them.
#     """
#     g_node_num = len(g_node_coor)
#     p_node_num = len(p_node_coor)
#     stiff_matrix = stiff_matrix_func(g_node_coor, p_node_coor, alpha)  # Units: (N)
#     Pb = Pb_func(g_node_coor, Nw_beta_bar, Nw_theta_bar, alpha, aero_coef_method, n_aero_coef, skew_approach, Chi_Ci='ones')
#     sw_vector = np.array([Nw_U_bar, np.zeros(len(Nw_U_bar)), np.zeros(len(Nw_U_bar))])  # instead of a=(u,v,w) a vector (U,0,0) is used.
#     F_sw = np.einsum('ndi,in->nd', Pb, sw_vector) / 2  # Global buffeting force vector. See Paper from LD Zhu, eq. (24). Units: (N)
#     F_sw_flat = np.ndarray.flatten(F_sw)  # flattening
#     F_sw_flat = np.array(list(F_sw_flat) + [0]*len(p_node_coor)*6)  # adding 0 force to all the remaining pontoon DOFs
#     # Global nodal Displacement matrix
#     D_sw_flat = np.linalg.inv(stiff_matrix) @ F_sw_flat
#     D_glob_sw = np.reshape(D_sw_flat, (g_node_num + p_node_num, 6))
#     g_node_coor_sw = g_node_coor + D_glob_sw[:g_node_num,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
#     p_node_coor_sw = p_node_coor + D_glob_sw[g_node_num:,:3]  # Only the first 3 DOF are added as displacements. The 4th is alpha_sw
#     return g_node_coor_sw, p_node_coor_sw, D_glob_sw


def Nw_static_wind_all(g_node_coor, p_node_coor, alpha, Nw_U_bar_all, Nw_beta_bar_all, Nw_theta_bar_all, aero_coef_method='2D_fit_cons', n_aero_coef=6, skew_approach='3D'):
    """
    Perform static wind analyses, for all available Nw cases
    From given U_bar, beta_0 and theta_0, and a static wind analysis, obtain:
     1 - displacements at all girder nodes.
     2 - absolute value of the 6 forces & moments at all girder nodes
    All girder nodes ==  all 1st nodes of each element + last node of last element
    Inputs shape(n_cases, g_node_num)
    Outputs shape(n_cases, g_node_num, 6)
    """
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    n_cases = Nw_U_bar_all.shape[0]
    Nw_g_node_coor_all, Nw_p_node_coor_all, Nw_D_glob_all = [], [], []
    Nw_D_loc_all = []  # displacements in local coordinates
    Nw_R12_all = []  # internal element forces 12 DOF
    for i, (Nw_U_bar, Nw_beta_bar, Nw_theta_bar) in enumerate(zip(Nw_U_bar_all, Nw_beta_bar_all, Nw_theta_bar_all)):
        print(f'Calculating sw... Case: {i}')
        Nw_g_node_coor, Nw_p_node_coor, Nw_D_glob = static_wind_from_U_beta_theta_bar(g_node_coor, p_node_coor, alpha, Nw_U_bar, Nw_beta_bar, Nw_theta_bar,
                                                                                      aero_coef_method=aero_coef_method, n_aero_coef=n_aero_coef, skew_approach=skew_approach)
        Nw_D_loc = mat_Ls_node_Gs_node_all_func(Nw_D_glob, g_node_coor, p_node_coor, alpha)  # Displacements in local coordinates
        Nw_R_loc = R_loc_func(Nw_D_glob, g_node_coor, p_node_coor, alpha)  # # Internal forces. orig. coord. + displacem. used to calc. R.
        Nw_g_node_coor_all.append(Nw_g_node_coor)  # Storing
        Nw_p_node_coor_all.append(Nw_p_node_coor)  # Storing
        Nw_D_glob_all.append(Nw_D_glob)  # Storing
        Nw_D_loc_all.append(Nw_D_loc)  # Storing
        Nw_R12_all.append(Nw_R_loc)  # Storing
    # Converting to arrays:
    Nw_g_node_coor_all = np.array(Nw_g_node_coor_all)
    Nw_p_node_coor_all = np.array(Nw_p_node_coor_all)
    Nw_D_glob_all = np.array(Nw_D_glob_all)
    Nw_D_loc_all = np.array(Nw_D_loc_all)
    Nw_R12_all = np.array(Nw_R12_all)  # R12 -> Internal forces represented as the 12 DOF of each element
    Nw_R12g_all = Nw_R12_all[:, :g_elem_num]  # g -> girder elements only
    Nw_R6g_all = np.array([mat_6_Ls_node_12_Ls_elem_girder_func(Nw_R12g_all[i]) for i in range(n_cases)])  # From 12DOF to 6DOF (first 6 DOF of each 12DOF element + last 6 DOF of the last element. See description of the function for details)
    return Nw_g_node_coor_all, Nw_p_node_coor_all, Nw_D_glob_all, Nw_D_loc_all, Nw_R12_all, Nw_R6g_all


def get_Iu_ANN_Z2_preds(ANN_Z1_preds, EN_Z1_preds, EN_Z2_preds):
    """
    inputs with special format: dict of (points) dicts of ('sector' & 'Iu') lists of floats

    Get the Artificial Neural Network predictions of Iu at a new height above sea level Z2, using a transfer function from different EN-1991-1-4 predictions at both Z1 and Z2.
    The transfer function is just a number for each mean wind direction (it varies with wind direction between e.g. 1.14 and 1.30, for Z1=48m to Z2=14.5m)

    Details:
    Converting ANN preds from Z1=48m, to Z2=14.5m, requires log(Z1/z0)/log(Z2/z0), but z0 depends on terrain roughness which is inhomogeneous and varies with wind direction. Solution: Predict Iu
    using the EN1991 at both Z1 and Z2 (using the "binary-geneous" terrain roughnesses), and find the transfer function between Iu(Z2) and Iu(Z1), for each wind direction, and apply to ANN preds.

    Try:
    from sympy import Symbol, simplify, ln
    z1 = Symbol('z1', real=True, positive=True)
    z2 = Symbol('z2', real=True, positive=True)
    c = Symbol('c', real=True, positive=True)
    z0 = Symbol('z0', real=True, positive=True)
    Iv1 = c / ln(z1 / z0)  # c is just a constant. It assumes Iu(Z) = sigma_u / Vm(Z), where sigma_u is independent of Z, and where Vm depends only on cr(Z), which depends on ln(Z / z0)
    Iv2 = c / ln(z2 / z0)
    simplify(Iv2 / Iv1)
    """
    ANN_Z2_preds = {}
    for point in list(ANN_Z1_preds.keys()):
        assert ANN_Z1_preds[point]['sector'] == EN_Z1_preds[point]['sector'] == EN_Z2_preds[point]['sector'] == np.arange(360).tolist(), 'all inputs must have all 360 directions!'
        Iu_ANN_Z2 = np.array(ANN_Z1_preds[point]['Iu']) * (np.array(EN_Z2_preds[point]['Iu']) / np.array(EN_Z1_preds[point]['Iu']))
        ANN_Z2_preds[point] = {'sector':ANN_Z1_preds[point]['sector'], 'Iu':Iu_ANN_Z2.tolist()}
    return ANN_Z2_preds


def Nw_Iu_all_dirs_database(g_node_coor, model='ANN', use_existing_file=True):
    """
    This function is simple but got a bit confusing in the process with too much copy paste...
    model: 'ANN' or 'EN'
    use_existing_file: False should be used when we have new g_node_num!!
    Returns an array of Iu with shape (n_g_nodes, n_dirs==360)
    """
    assert zbridge == 14.5, "ERROR: zbridge!=14.5m. You must produce new Iu_EN_preds at the correct Z. Go to MetOcean project and replace all '14m' by desired Z. Copy the new json files to this project "

    if model == 'ANN':
        if not use_existing_file:
            # Then there must exist 3 other necessary files (at each WRF node) that will be used to create and store the desired file (at each girder node)
            with open(r"intermediate_results\\Nw_Iu\\Iu_48m_ANN_preds.json") as f:
                dict_Iu_48m_ANN_preds = json.loads(f.read())
            with open(r"intermediate_results\\Nw_Iu\\Iu_48m_EN_preds.json") as f:
                dict_Iu_48m_EN_preds = json.loads(f.read())
            with open(r"intermediate_results\\Nw_Iu\\Iu_14m_EN_preds.json") as f:
                dict_Iu_14m_EN_preds = json.loads(f.read())
            dict_Iu_14m_ANN_preds = get_Iu_ANN_Z2_preds(ANN_Z1_preds=dict_Iu_48m_ANN_preds, EN_Z1_preds=dict_Iu_48m_EN_preds, EN_Z2_preds=dict_Iu_14m_EN_preds)
            Iu_14m_ANN_preds_WRF = np.array([dict_Iu_14m_ANN_preds[k]['Iu'] for k in dict_Iu_14m_ANN_preds.keys()]).T  # calculated at 11 WRF nodes
            Iu_14m_ANN_preds = interpolate_from_WRF_nodes_to_g_nodes(Iu_14m_ANN_preds_WRF, g_node_coor, WRF_node_coor, plot=False)  # calculated at the girder nodes
            Iu_14m_ANN_preds = Iu_14m_ANN_preds.T
            # Storing
            with open(r'intermediate_results\\Nw_Iu\\Iu_14m_ANN_preds_g_nodes.json', 'w', encoding='utf-8') as f:
                json.dump(Iu_14m_ANN_preds.tolist(), f, ensure_ascii=False, indent=4)
        else:
            with open(r'intermediate_results\\Nw_Iu\\Iu_14m_ANN_preds_g_nodes.json') as f:
                Iu_14m_ANN_preds = np.array(json.loads(f.read()))
        return Iu_14m_ANN_preds
    elif model == 'EN':
        if not use_existing_file:
            with open(r"intermediate_results\\Nw_Iu\\Iu_14m_EN_preds.json") as f:
                dict_Iu_14m_EN_preds = json.loads(f.read())
            Iu_14m_EN_preds_WRF = np.array([dict_Iu_14m_EN_preds[k]['Iu'] for k in dict_Iu_14m_EN_preds.keys()]).T  # calculated at 11 WRF nodes
            Iu_14m_EN_preds = interpolate_from_WRF_nodes_to_g_nodes(Iu_14m_EN_preds_WRF, g_node_coor, WRF_node_coor, plot=False)  # calculated at the girder nodes
            Iu_14m_EN_preds = Iu_14m_EN_preds.T
            # Storing
            with open(r'intermediate_results\\Nw_Iu\\Iu_14m_EN_preds_g_nodes.json', 'w', encoding='utf-8') as f:
                json.dump(Iu_14m_EN_preds.tolist(), f, ensure_ascii=False, indent=4)
        else:
            with open(r'intermediate_results\\Nw_Iu\\Iu_14m_EN_preds_g_nodes.json') as f:
                Iu_14m_EN_preds = np.array(json.loads(f.read()))
        return Iu_14m_EN_preds


def Nw_beta_and_theta_bar_func(g_node_coor, Nw_beta_0, Nw_theta_0, alpha):
    """Returns the Nonhomogeneous beta_bar and theta_bar at each node, relative to the mean of the axes of the adjacent elements.
    Note: the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
    n_g_nodes = len(g_node_coor)
    assert Nw_beta_0.shape[-1] == Nw_theta_0.shape[-1] == n_g_nodes
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
    T_GsNw = T_GsNw_func(Nw_beta_0, Nw_theta_0)
    T_LsNw = np.einsum('...ij,...jk->...ik', T_LsGs, T_GsNw)  # todo: if problems, replace "..." with "n"
    U_Gw_norm = np.array([1, 0, 0])  # U_Gw = (U, 0, 0), so the normalized U_Gw_norm is (1, 0, 0)
    U_Ls = np.einsum('...ij,j->...i', T_LsNw, U_Gw_norm)   # todo: if problems, replace "..." with "n"
    Ux = U_Ls[..., 0]  # todo: if problems, replace "..." with ":"
    Uy = U_Ls[..., 1]  # todo: if problems, replace "..." with ":"
    Uz = U_Ls[..., 2]  # todo: if problems, replace "..." with ":"
    Uxy = np.sqrt(Ux ** 2 + Uy ** 2)
    # Nw_beta_bar = np.array([-np.arccos(Uy[i] / Uxy[i]) * np.sign(Ux[i]) for i in range(len(g_node_coor))])
    # Nw_theta_bar = np.array([np.arcsin(Uz[i] / 1) for i in range(len(g_node_coor))])
    Nw_beta_bar = -np.arccos(Uy / Uxy) * np.sign(Ux)
    Nw_theta_bar = np.arcsin(Uz / 1)
    return Nw_beta_bar, Nw_theta_bar


def equivalent_Hw_beta_0_all(Nw_U_bar, Nw_beta_0, g_node_num, eqv_Hw_beta_method):
    """this is an auxiliary function that gets Hw_beta_0_all"""
    if eqv_Hw_beta_method == 'mean':
        Hw_cos_beta_0_all = np.repeat(np.mean(np.cos(Nw_beta_0), axis=1)[:, None], repeats=g_node_num, axis=1)  # making the average of all betas along the bridge girder
        Hw_sin_beta_0_all = np.repeat(np.mean(np.sin(Nw_beta_0), axis=1)[:, None], repeats=g_node_num, axis=1)  # making the average of all betas along the bridge girder
    elif eqv_Hw_beta_method == 'U_weighted_mean':
        Hw_cos_beta_0_all = np.repeat(np.average(np.cos(Nw_beta_0), axis=1, weights=Nw_U_bar)[:, None], repeats=g_node_num, axis=1)  # the weighted averages were carefully tested
        Hw_sin_beta_0_all = np.repeat(np.average(np.sin(Nw_beta_0), axis=1, weights=Nw_U_bar)[:, None], repeats=g_node_num, axis=1)  # the weighted averages were carefully tested
    elif eqv_Hw_beta_method == 'U2_weighted_mean':
        Hw_cos_beta_0_all = np.repeat(np.average(np.cos(Nw_beta_0), axis=1, weights=Nw_U_bar ** 2)[:, None], repeats=g_node_num, axis=1)  # the weighted averages were carefully tested
        Hw_sin_beta_0_all = np.repeat(np.average(np.sin(Nw_beta_0), axis=1, weights=Nw_U_bar ** 2)[:, None], repeats=g_node_num, axis=1)  # the weighted averages were carefully tested
    Hw_beta_0_all = beta_within_minus_Pi_and_Pi_func(from_cos_sin_to_0_2pi(Hw_cos_beta_0_all, Hw_sin_beta_0_all, out_units='rad'))  # making the average of all betas along the bridge girder
    return Hw_beta_0_all


class NwOneCase:
    """
    Non-homogeneous wind class. Gets the necessary information of one WRF case.
    """
    def __init__(self, reset_structure=True, reset_WRF_database=True, reset_wind=True):
        if reset_structure:
            # Structure
            self.g_node_coor = None
            self.p_node_coor = None
            self.alpha = None
        if reset_WRF_database:
            # WRF Dataframe:
            self.df_WRF = None  # Dataframe with WRF speeds and directions at each of the 11 WRF-bridge nodes
            self.aux_WRF = {}  # Auxiliary variables are stored here
            self.props_WRF = {}  # Properties of the WRF data
        if reset_wind:
            # Non-homogeneous wind
            self.df_WRF_idx = None  # Index used from the df_WRF to generate the Nw wind
            self.U_bar = None  # Array of non-homogeneous mean wind speeds at all the girder nodes.
            self.beta_DB = None
            self.beta_0 = None
            self.theta_0 = None
            self.beta_bar = None
            self.theta_bar = None
            self.Ii = None  # Turbulence intensities
            self.f_array = None
            self.Ai = None
            self.iLj = None
            self.S_a = None
            self.S_aa = None
            # Other
            self.create_new_Iu_all_dirs_database = True  # This needs to be run once

    def set_df_WRF(self, U_tresh=12, tresh_requirement_type='any', sort_by='time'):
        """
        Set (establish) a dataframe with the WRF data, according to the argument filters.
        U_tresh: e.g. 12  # m/s. threshold. Data rows with datapoints below threshold, are removed.
        tresh_requirement_type: 'any' to keep all cases where at least 1 U is above U_tresh; 'all' to keep only cases where all U >= U_tresh
        sort_by: 'time', 'ws_var', 'ws_max', 'wd_var'
        """
        WRF_dataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', r'WRF_at_bridge_nodes.nc'), 'r', format='NETCDF4')
        with warnings.catch_warnings():  # ignore a np.bool deprecation warning inside the netCDF4 module
            warnings.simplefilter("ignore")
            ws_orig = WRF_dataset['ws'][:].data  # original data
            wd_orig = WRF_dataset['wd'][:].data
            time_orig = WRF_dataset['time'][:].data
            self.aux_WRF['lats_bridge'] = WRF_dataset['latitudes'][:].data
            self.aux_WRF['lons_bridge'] = WRF_dataset['longitudes'][:].data
        n_WRF_bridge_nodes = np.shape(ws_orig)[0]
        ws_cols = [f'ws_{n:02}' for n in range(n_WRF_bridge_nodes)]
        wd_cols = [f'wd_{n:02}' for n in range(n_WRF_bridge_nodes)]
        self.aux_WRF['ws_cols'] = ws_cols
        self.aux_WRF['wd_cols'] = wd_cols
        df_WRF = pd.DataFrame(ws_orig.T, columns=ws_cols)
        df_WRF = df_WRF.join(pd.DataFrame(wd_orig.T, columns=wd_cols))
        df_WRF['hour'] = time_orig
        df_WRF['datetime'] = [datetime.datetime.min + datetime.timedelta(hours=int(time_orig[i])) - datetime.timedelta(days=2) for i in range(len(time_orig))]
        # Filtering
        bools_to_keep = df_WRF[ws_cols] >= U_tresh
        if tresh_requirement_type == 'any':
            bools_to_keep = bools_to_keep.any(axis='columns')  # choose .any() to keep rows with at least one value above treshold, or .all() to keep only rows where all values are above treshold
        elif tresh_requirement_type == 'all':
            bools_to_keep = bools_to_keep.all(axis='columns')  # choose .any() to keep rows with at least one value above treshold, or .all() to keep only rows where all values are above treshold
        else:
            raise ValueError
        df_WRF = df_WRF.loc[bools_to_keep].reset_index(drop=True)
        # Sorting
        ws = df_WRF[ws_cols]
        wd = np.deg2rad(df_WRF[wd_cols])
        wd_cos = np.cos(wd)
        wd_sin = np.sin(wd)
        wd_cos_var = np.var(wd_cos, axis=1)
        wd_sin_var = np.var(wd_sin, axis=1)
        idxs_sorted_by = {'time':   np.arange(df_WRF.shape[0]),
                          'ws_var': np.array(np.argsort(np.var(ws, axis=1))),
                          'ws_max': np.array(np.argsort(np.max(ws, axis=1))),
                          'wd_var': np.array(np.argsort(pd.concat([wd_cos_var, wd_sin_var], axis=1).max(axis=1)))}
        self.df_WRF = df_WRF.loc[idxs_sorted_by[sort_by]].reset_index(drop=True)
        self.props_WRF['df_WRF'] = {'U_tresh':U_tresh, 'tresh_requirement_type':tresh_requirement_type, 'sorted_by':sort_by}

    def set_structure(self, g_node_coor, p_node_coor, alpha):
        self.g_node_coor = g_node_coor
        self.p_node_coor = p_node_coor
        self.alpha = alpha
        # Reseting wind. Making sure we have no structural-dependent Nw data yet (e.g. U_bar depends on n_g_nodes).
        self.__init__(reset_structure=False, reset_WRF_database=False, reset_wind=True)

    def set_Nw_wind(self, df_WRF_idx, force_Nw_U_and_N400_U_to_have_same=None, model='ANN', cospec_type=2,  f_array='static_wind_only'):
        self._set_U_bar_beta_DB_beta_0_theta_0(df_WRF_idx=df_WRF_idx, force_Nw_U_and_N400_U_to_have_same=force_Nw_U_and_N400_U_to_have_same)
        self._set_beta_and_theta_bar()
        self._set_Ii(model=model)
        if not f_array is 'static_wind_only':
            self._set_S_a(f_array=f_array)
            self._set_S_aa(cospec_type=cospec_type)

    def _set_U_bar_beta_DB_beta_0_theta_0(self, df_WRF_idx, force_Nw_U_and_N400_U_to_have_same=None):
        """
        Returns a vector of Nonhomogeneous mean wind at each of the g_nodes
        force_Nw_and_U_bar_to_have_same_avg : None, 'mean', 'energy'. force the Nw_U_bar_at_WRF_nodes to have the same e.g. mean 1, and thus when multiplied with U_bar, the result will have the same mean (of all nodes) wind
        """
        # Setting Nw U_bar:
        assert self.df_WRF is not None
        assert self.g_node_coor is not None

        g_node_coor = self.g_node_coor
        ws_cols = self.aux_WRF['ws_cols']
        wd_cols = self.aux_WRF['wd_cols']
        Nw_U_bar_at_WRF_nodes = self.df_WRF[ws_cols].iloc[df_WRF_idx].to_numpy()
        n_WRF_cases = Nw_U_bar_at_WRF_nodes.shape[0]
        assert Nw_U_bar_at_WRF_nodes.shape[-1] == n_WRF_nodes
        interp_fun = interpolate_from_WRF_nodes_to_g_nodes(Nw_U_bar_at_WRF_nodes, g_node_coor, WRF_node_coor)
        if force_Nw_U_and_N400_U_to_have_same is None:
            Nw_U_bar = interp_fun
        else:
            U_bar_10min = U_bar_func(g_node_coor)  # N400
            if force_Nw_U_and_N400_U_to_have_same == 'mean':
                Nw_U_bar = U_bar_10min * (interp_fun / np.mean(interp_fun))
                assert np.isclose(np.mean(Nw_U_bar), np.mean(U_bar_10min))  # same mean(U)
            elif force_Nw_U_and_N400_U_to_have_same == 'energy':
                Nw_U_bar = U_bar_10min * np.sqrt(interp_fun / np.mean(interp_fun))
                assert np.isclose(np.mean(Nw_U_bar ** 2), np.mean(U_bar_10min ** 2))  # same energy = same mean(U**2)
        # Setting Nw beta_DB, beta_0 and theta_0:
        wd_at_WRF_nodes = np.deg2rad(self.df_WRF[wd_cols].iloc[df_WRF_idx].to_numpy())
        Nw_beta_DB_cos = interpolate_from_WRF_nodes_to_g_nodes(np.cos(wd_at_WRF_nodes, dtype=float), g_node_coor, WRF_node_coor)
        Nw_beta_DB_sin = interpolate_from_WRF_nodes_to_g_nodes(np.sin(wd_at_WRF_nodes, dtype=float), g_node_coor, WRF_node_coor)
        Nw_beta_DB = from_cos_sin_to_0_2pi(Nw_beta_DB_cos, Nw_beta_DB_sin, out_units='rad')
        Nw_beta_0 = beta_0_func(Nw_beta_DB)
        Nw_theta_0 = np.zeros(Nw_beta_0.shape)
        self.beta_DB = Nw_beta_DB
        self.beta_0 = Nw_beta_0
        self.theta_0 = Nw_theta_0
        self.U_bar = Nw_U_bar
        self.df_WRF_idx = df_WRF_idx

    def _set_beta_and_theta_bar(self):
        """Returns the Nonhomogeneous beta_bar and theta_bar at each node, relative to the mean of the axes of the adjacent elements.
        Note: the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
        assert self.beta_DB is not None
        assert self.g_node_coor is not None
        g_node_coor = self.g_node_coor
        alpha = self.alpha
        Nw_beta_0 = self.beta_0
        Nw_theta_0 = self.theta_0
        Nw_beta_bar, Nw_theta_bar = Nw_beta_and_theta_bar_func(g_node_coor, Nw_beta_0, Nw_theta_0, alpha)
        self.beta_bar = Nw_beta_bar
        self.theta_bar = Nw_theta_bar

    def _set_Ii(self, model='ANN'):
        """
        For computer efficiency, nearest neighbour is used (instead of linear inerpolation), assuming 360 directions in the database
        Nw_beta_DB: len == n_g_nodes
        Returns: array that describes Iu, Iv, Iw at each g node, with shape (n_nodes, 3)
        """
        assert self.beta_DB is not None
        assert self.g_node_coor is not None
        Nw_beta_DB = self.beta_DB
        g_node_coor = self.g_node_coor

        if self.create_new_Iu_all_dirs_database:
            # Creating a database when importing this nonhomogeneity.py file! This will be run only once, when this file is imported, so that the correct g_node_num is used to create the database!
            Nw_Iu_all_dirs_database(g_node_coor, model='ANN', use_existing_file=False)
            Nw_Iu_all_dirs_database(g_node_coor, model='EN', use_existing_file=False)

        Iu = Nw_Iu_all_dirs_database(g_node_coor, model=model, use_existing_file=True)
        assert Iu.shape[-1] == 360, "360 directions assumed in the database. If not, the code must change substantially"
        dir_idxs = np.rint(np.rad2deg(Nw_beta_DB)).astype(int)
        dir_idxs[dir_idxs == 360] = 0  # in case a direction is assumed to be 360, convert it to 0
        Iu = np.array([Iu[n, d] for n, d in enumerate(dir_idxs)])
        Iv = 0.84 * Iu  # Design basis rev 2C, 2021, Chapter 3.6.1
        Iw = 0.60 * Iu  # Design basis rev 2C, 2021, Chapter 3.6.1
        self.Ii = np.array([Iu, Iv, Iw]).T

    def _set_S_a(self, f_array):
        """
        f_array and n_hat need to be in Hertz, not radians!
        """
        Nw_U_bar = self.U_bar
        Nw_Ii = self.Ii
        g_node_coor = self.g_node_coor
        Ai = Ai_func(cond_rand_A=False)
        iLj = iLj_func(g_node_coor)
        sigma_n = np.einsum('na,n->na', Nw_Ii, Nw_U_bar)  # standard deviation of the turbulence, for each node and each component.
        # Autospectrum
        n_hat = np.einsum('f,na,n->fna', f_array, iLj[:, :, 0], 1 / Nw_U_bar)
        S_a = np.einsum('f,na,a,fna,fna->fna', 1/f_array, sigma_n ** 2, Ai, n_hat, 1 / (1 + 1.5 * np.einsum('a,fna->fna', Ai, n_hat)) ** (5 / 3))
        self.f_array = f_array
        self.Ai = Ai
        self.iLj = iLj
        self.S_a = S_a

    def _set_S_aa(self, cospec_type=2):
        """
        In Hertz. The input coordinates are in Global Structural Gs (even though Gw is calculated and used in this function)
        """
        g_node_coor = self.g_node_coor  # shape (g_node_num,3)
        U_bar = self.U_bar
        f_array = self.f_array
        S_a = self.S_a
        beta_0 = self.beta_0
        theta_0 = self.theta_0
        Cij = Cij_func(cond_rand_C=False)
        n_g_nodes = len(g_node_coor)

        # Difficult part. We need a cross-transformation matrix T_GsNw_avg, which is an array with shape (n_g_nodes, n_g_nodes, 3) where each (i,j) entry is the T_GsNw_avg, where Nw_avg is the avg. between Nw_i (at node i) and Nw_j (at node j)
        T_GsNw = T_GsNw_func(beta_0, theta_0)  # shape (n_g_nodes,3,3)
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

        U_bar_avg = (U_bar[:, None] + U_bar) / 2  # from shape (n_g_nodes) to shape (n_g_nodes,n_g_nodes)

        if cospec_type == 1:  # Alternative 1: LD Zhu coherence and cross-spectrum. Developed in radians? So it is converted to Hertz in the end.
            raise NotImplementedError
        if cospec_type == 2:  # Coherence and cross-spectrum (adapted Davenport for 3D). Developed in Hertz!
            f_hat_aa = np.einsum('f,mna->fmna', f_array,
                                 np.divide(np.sqrt((Cij[:, 0] * delta_xyz_Nw[:, :, 0, None]) ** 2 + (Cij[:, 1] * delta_xyz_Nw[:, :, 1, None]) ** 2 + (Cij[:, 2] * delta_xyz_Nw[:, :, 2, None]) ** 2),
                                           U_bar_avg[:, :, None]))  # This is actually in omega units, not f_array, according to eq.(10.35b)! So: rad/s
            f_hat = f_hat_aa  # this was confirmed to be correct with a separate 4 loop "f_hat_aa_confirm" and one Cij at the time
            R_aa = np.e ** (-f_hat)  # phase spectrum is not included because usually there is no info. See book eq.(10.34)
            S_aa = np.einsum('fmna,fmna->fmna', np.sqrt(np.einsum('fma,fna->fmna', S_a, S_a)), R_aa)  # S_a is only (3,) because assumed no cross-correlation between components
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
        self.S_aa = S_aa


class NwAllCases:
    """
    Whereas NwOneCase stores the Nw data of one WRF case, this NwAllCases stores the Nw data of multiple or all WRF cases
    n_Nw_cases: 'all' or an integer. An integer, e.g. 10, will store the Nw data of the last 10 cases (df_WRF has ascending order), e.g. those with highest ws_max
    """
    def __init__(self, reset_structure=True, reset_WRF_database=True, reset_wind=True):
        if reset_structure:
            # Structure - Only 1
            self.g_node_coor = None
            self.p_node_coor = None
            self.alpha = None
            # Structure - 1 for each Nw case, updated after each static wind analysis
            self.g_node_coor_sw = []
            self.p_node_coor_sw = []
            self.alpha_sw = []
        if reset_WRF_database:
            # WRF Dataframe:
            self.df_WRF = None  # Dataframe with WRF speeds and directions at each of the 11 WRF-bridge nodes
            self.aux_WRF = {}  # Auxiliary variables are stored here
            self.props_WRF = {}  # Properties of the WRF data
        if reset_wind:
            # Non-homogeneous wind properties. Those with "_sw" are an update after a static analysis.
            self.n_Nw_cases = None
            self.U_bar = []  # Array of non-homogeneous mean wind speeds at all the girder nodes.
            self.beta_DB = []
            self.beta_0 = []
            self.theta_0 = []
            self.beta_bar = []
            self.theta_bar = []
            self.beta_bar_sw = []  #  updated after a static analysis
            self.theta_bar_sw = []  #  updated after a static analysis. This takes into account alpha_sw
            self.Ii = []  # Turbulence intensities
            self.f_array = []
            self.Ai = []
            self.iLj = []
            self.S_a = []
            self.S_aa = []
            # Other
            self.equiv_Hw_U_bar = []  # Equivalent Homogeneous mean wind speeds at all the girder nodes
            self.equiv_Hw_beta_0 = []  # Equivalent Homogeneous beta 0
            self.equiv_Hw_theta_0 = []  # Equivalent Homogeneous theta 0
            self.equiv_Hw_beta_bar = []  # Equivalent Homogeneous beta bar
            self.equiv_Hw_theta_bar = []  # Equivalent Homogeneous theta bar

    def set_df_WRF(self, U_tresh=12, tresh_requirement_type='any', sort_by='time'):
        """
        Set (establish) a dataframe with the WRF data, according to the argument filters.
        U_tresh: e.g. 12  # m/s. threshold. Data rows with datapoints below threshold, are removed.
        tresh_requirement_type: 'any' to keep all cases where at least 1 U is above U_tresh; 'all' to keep only cases where all U >= U_tresh
        sort_by: 'time', 'ws_var', 'ws_max', 'wd_var'
        """
        Nw_temp = NwOneCase()
        Nw_temp.set_df_WRF(U_tresh=U_tresh, tresh_requirement_type=tresh_requirement_type, sort_by=sort_by)
        self.df_WRF = Nw_temp.df_WRF
        self.aux_WRF = Nw_temp.aux_WRF
        self.props_WRF = Nw_temp.props_WRF

    def set_structure(self, g_node_coor, p_node_coor, alpha):
        self.g_node_coor = g_node_coor
        self.p_node_coor = p_node_coor
        self.alpha = alpha
        # Reseting wind. Making sure we have no structural-dependent Nw data yet (e.g. U_bar depends on n_g_nodes and beta_bar on g_node_coor, so wind needs to be reset).
        self.__init__(reset_structure=False, reset_WRF_database=False, reset_wind=True)

    def set_Nw_wind(self, n_Nw_cases='all', force_Nw_U_and_N400_U_to_have_same=None, Iu_model='ANN', cospec_type=2, f_array='static_wind_only'):
        assert self.df_WRF is not None, "df_WRF is not set. Please run the method set_df_WRF() on the current instance"
        assert self.g_node_coor is not None, "Structure is not set. Please run the method set_structure() on the current instance"
        if n_Nw_cases == 'all':
            n_Nw_cases = len(self.df_WRF)
        self.n_Nw_cases = n_Nw_cases
        # Copying the current WRF and structural data into a temporary instance of NwOneCase, which will be used to retrieve the Nw wind data for all WRF cases
        Nw_temp = NwOneCase()
        Nw_temp.set_structure(self.g_node_coor, self.p_node_coor, self.alpha)
        Nw_temp.df_WRF = self.df_WRF
        Nw_temp.aux_WRF = self.aux_WRF
        Nw_temp.props_WRF = self.props_WRF
        for i in range(n_Nw_cases):
            Nw_temp.set_Nw_wind(df_WRF_idx=-i-1, force_Nw_U_and_N400_U_to_have_same=force_Nw_U_and_N400_U_to_have_same, model=Iu_model, cospec_type=cospec_type, f_array=f_array)
            self.U_bar.append(Nw_temp.U_bar)
            self.beta_DB.append(Nw_temp.beta_DB)
            self.beta_0.append(Nw_temp.beta_0)
            self.theta_0.append(Nw_temp.theta_0)
            self.beta_bar.append(Nw_temp.beta_bar)
            self.theta_bar.append(Nw_temp.theta_bar)
            self.Ii.append(Nw_temp.Ii)
            self.f_array.append(Nw_temp.f_array)
            self.Ai.append(Nw_temp.Ai)
            self.iLj.append(Nw_temp.iLj)
            self.S_a.append(Nw_temp.S_a)
            self.S_aa.append(Nw_temp.S_aa)
        self._convert_attributes_from_lists_to_arrs()

    def set_equivalent_Hw_U_bar(self, force_Nw_U_bar_and_U_bar_to_have_same='energy'):
        """
        Nw_U_bar shape: (n_cases, n_nodes)
        Returns a homogeneous wind velocity field, equivalent to the input Nw_U_bar in terms of force_Nw_U_bar_and_U_bar_to_have_same
        force_Nw_U_bar_and_U_bar_to_have_same: None, 'mean', 'energy'. force the U_bar_equivalent to have the same mean or energy 1 as Nw_U_bar
        """
        assert self.U_bar is not None
        assert self.g_node_coor is not None
        g_node_coor = self.g_node_coor
        Nw_U_bar = self.U_bar
        if force_Nw_U_bar_and_U_bar_to_have_same is None:
            U_bar_equivalent = U_bar_func(g_node_coor)
        elif force_Nw_U_bar_and_U_bar_to_have_same == 'mean':
            U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.mean(Nw_U_bar, axis=1)[:, None]
            assert all(np.isclose(np.mean(Nw_U_bar, axis=1)[:, None], np.mean(U_bar_equivalent, axis=1)[:, None]))
        elif force_Nw_U_bar_and_U_bar_to_have_same == 'energy':
            U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.sqrt(np.mean(Nw_U_bar ** 2, axis=1)[:, None])
            assert all(np.isclose(np.mean(Nw_U_bar ** 2, axis=1)[:, None], np.mean(U_bar_equivalent ** 2, axis=1)[:, None]))  # same energy = same mean(U**2))
        self.equiv_Hw_U_bar = U_bar_equivalent

    def set_equivalent_Hw_beta(self, eqv_Hw_beta_method = 'U2_weighted_mean'):
        """eqv_Hw_beta_method: 'mean' or 'U_weighted_mean' or 'U2_weighted_mean'"""
        g_node_coor = self.g_node_coor
        g_node_num = g_node_coor.shape[0]
        n_Nw_cases = self.n_Nw_cases
        Hw_beta_0_all = equivalent_Hw_beta_0_all(Nw_U_bar=self.U_bar, Nw_beta_0=self.beta_0, g_node_num=g_node_num, eqv_Hw_beta_method=eqv_Hw_beta_method)
        Hw_theta_0_all = np.zeros((n_Nw_cases, g_node_num))
        Hw_beta_bar_all, Hw_theta_bar_all = Nw_beta_and_theta_bar_func(g_node_coor, Hw_beta_0_all, Hw_theta_0_all, self.alpha)
        self.equiv_Hw_beta_0 = Hw_beta_0_all
        self.equiv_Hw_theta_0 = Hw_theta_0_all
        self.equiv_Hw_beta_bar = Hw_beta_bar_all
        self.equiv_Hw_theta_bar = Hw_theta_bar_all

    def _convert_attributes_from_lists_to_arrs(self):
        """Converts the instance attributes from a list of lists, to numpy arrays"""
        self.U_bar = np.array(self.U_bar)
        self.beta_DB = np.array(self.beta_DB)
        self.beta_0 = np.array(self.beta_0)
        self.theta_0 = np.array(self.theta_0)
        self.beta_bar = np.array(self.beta_bar)
        self.theta_bar = np.array(self.theta_bar)
        self.Ii = np.array(self.Ii)
        self.f_array = np.array(self.f_array)
        self.Ai = np.array(self.Ai)
        self.iLj = np.array(self.iLj)
        self.S_a = np.array(self.S_a)
        self.S_aa = np.array(self.S_aa)

    def plot_U(self, df_WRF_idx):
        # def colorbar(mappable):
        #     ax = mappable.axes
        #     fig = ax.figure
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     return fig.colorbar(mappable, cax=cax)
        ws_cols = self.aux_WRF['ws_cols']
        wd_cols = self.aux_WRF['wd_cols']
        ws_to_plot = self.df_WRF[ws_cols].iloc[df_WRF_idx].to_numpy()
        wd_to_plot = np.deg2rad(self.df_WRF[wd_cols].iloc[df_WRF_idx].to_numpy())
        cm = matplotlib.cm.cividis
        norm = matplotlib.colors.Normalize()
        sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
        ws_colors = cm(norm(ws_to_plot))
        plt.figure(figsize=(4, 6), dpi=300)
        lats_bridge = self.aux_WRF['lats_bridge']
        lons_bridge = self.aux_WRF['lons_bridge']
        plt.scatter(*np.array([lons_bridge, lats_bridge]), color='black', s=5)
        plt.gca().set_aspect(lat_lon_aspect_ratio, adjustable='box')
        plt.quiver(*np.array([lons_bridge, lats_bridge]), -ws_to_plot * np.sin(wd_to_plot), -ws_to_plot * np.cos(wd_to_plot), color=ws_colors, angles='uv', scale=80, width=0.015, headlength=3,
                   headaxislength=3)
        cbar = plt.colorbar(sm, fraction=0.078, pad=0.076)  # play with these values until the colorbar has good size and the entire plot and axis labels is visible
        cbar.set_label('U [m/s]')
        plt.title(f'Inhomogeneous wind')
        plt.xlim(5.35, 5.41)
        plt.ylim(60.080, 60.135)
        # plt.xlim(5.362, 5.395)
        # plt.ylim(60.084, 60.13)
        plt.xlabel('Longitude [$\degree$]')
        plt.ylabel('Latitude [$\degree$]')
        plt.tight_layout()
        plt.savefig('plots/U_case.png')
        plt.show()
        plt.close()

    def plot_Ii_at_WRF_points(self):
        g_node_coor = self.g_node_coor  # shape (g_node_num,3)
        n_g_nodes = len(g_node_coor)
        lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged()
        with open(r"intermediate_results\\Nw_Iu\\Iu_48m_ANN_preds.json") as f:
            dict_Iu_48m_ANN_preds = json.loads(f.read())
        with open(r"intermediate_results\\Nw_Iu\\Iu_48m_EN_preds.json") as f:
            dict_Iu_48m_EN_preds = json.loads(f.read())

        # bj_coors_WRONG_OLD_METHOD = np.array([[-34449.260, 6699999.046],
        #                                       [-34244.818, 6700380.872],
        #                                       [-34057.265, 6700792.767],
        #                                       [-33888.469, 6701230.609],
        #                                       [-33740.109, 6701690.024],
        #                                       [-33613.662, 6702166.417],
        #                                       [-33510.378, 6702655.026],
        #                                       [-33431.282, 6703150.969],
        #                                       [-33377.153, 6703649.290],
        #                                       [-33348.522, 6704145.006],
        #                                       [-33345.665, 6704633.167]])
        bj_coors = np.array([[-34449.260, 6699999.046],
                             [-34098.712, 6700359.394],
                             [-33786.051, 6700752.909],
                             [-33514.390, 6701175.648],
                             [-33286.431, 6701623.380],
                             [-33104.435, 6702091.622],
                             [-32970.204, 6702575.689],
                             [-32885.057, 6703070.741],
                             [-32849.826, 6703571.830],
                             [-32864.842, 6704073.945],
                             [-32929.936, 6704572.075]])
        bj_pt_strs = ['bj' + f'{i + 1:02}' for i in range(n_WRF_nodes)]

        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap

        # FIGURE 1, ZOOMED OUT
        lon_lims = [-45000, -20000]
        lat_lims = [6.685E6, 6.715E6]
        lon_lim_idxs = [np.where(lon_mosaic[0, :] == lon_lims[0])[0][0], np.where(lon_mosaic[0, :] == lon_lims[1])[0][0]]
        lat_lim_idxs = [np.where(lat_mosaic[:, 0] == lat_lims[0])[0][0], np.where(lat_mosaic[:, 0] == lat_lims[1])[0][0]]
        lon_mosaic_crop = lon_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
        lat_mosaic_crop = lat_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
        imgs_mosaic_crop = imgs_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
        # cmap = copy.copy(plt.get_cmap('magma_r'))
        cmap = copy.copy(plt.get_cmap('binary'))
        imgs_mosaic_crop = np.ma.masked_where(imgs_mosaic_crop == 0, imgs_mosaic_crop)  # set mask where height is 0, to be converted to another color
        cmap.set_bad(color='skyblue')  # color where height == 0
        plt.figure(dpi=400)
        plt.title('Topography')
        bbox = ((lon_mosaic_crop.min(), lon_mosaic_crop.max(),
                 lat_mosaic_crop.min(), lat_mosaic_crop.max()))
        imshow = plt.imshow(imgs_mosaic_crop, extent=bbox, zorder=0, cmap=cmap, vmin=0, vmax=750)
        main_ax = plt.gca()
        wrax = {}
        for pt_idx, pt in enumerate(bj_pt_strs):
            plt.scatter(bj_coors[pt_idx][0], bj_coors[pt_idx][1], marker='o', facecolors='black', edgecolors='black', s=10, label='Measurement location' if pt == 0 else None)
        cb = plt.colorbar(imshow)
        plt.xlabel('Easting [m]')
        plt.ylabel('Northing [m]')
        cb.set_label('Height [m]')
        # plt.legend()  # [handles[idx] for idx in order], [labels[idx] for idx in order], handletextpad=0.1)
        plt.tight_layout(pad=0.05)

        Iu_min_all_pts = np.min(np.array([dict_Iu_48m_ANN_preds[pt]['Iu'] for pt in bj_pt_strs]))
        Iu_max_all_pts = np.max(np.array([dict_Iu_48m_ANN_preds[pt]['Iu'] for pt in bj_pt_strs]))
        # for pt_idx, pt in enumerate(bj_pt_strs):
        #     ########### WIND ROSES
        #     wd = np.array(dict_Iu_48m_ANN_preds[pt]['sector'])
        #     Iu = np.array(dict_Iu_48m_ANN_preds[pt]['Iu'])
        #     Iu_min = np.min(Iu)
        #     Iu_max = np.max(Iu)
        #     wrax[pt] = inset_axes(main_ax,
        #                           width=0.1,  # size in inches
        #                           height=0.1,  # size in inches
        #                           loc='center',  # center bbox at given position
        #                           bbox_to_anchor=(bj_coors[pt_idx][0], bj_coors[pt_idx][1]),  # position of the axe
        #                           bbox_transform=main_ax.transData,  # use data coordinate (not axe coordinate)
        #                           axes_class=WindroseAxes)  # specify the class of the axe
        #     # print(f'Min: {(Iu_min-Iu_min_all_pts)/(Iu_max_all_pts-Iu_min_all_pts)}')
        #     # print(f'Max; {(Iu_max-Iu_min_all_pts)/(Iu_max_all_pts-Iu_min_all_pts)}')
        #     wrax[pt].bar(wd, Iu, opening=1.0, nsector=360, cmap=truncate_colormap(matplotlib.pyplot.cm.Reds, (Iu_min - Iu_min_all_pts) / (Iu_max_all_pts - Iu_min_all_pts),
        #                                                                           (Iu_max - Iu_min_all_pts) / (Iu_max_all_pts - Iu_min_all_pts)))
        #     wrax[pt].tick_params(labelleft=False, labelbottom=False)
        #     wrax[pt].patch.set_alpha(0)
        #     wrax[pt].axis('off')
        plt.savefig('plots/ANN_preds_zoomout.png')
        plt.show()
        plt.close()

        # FIGURE 2, ZOOMED IN
        lon_lims = [-35000, -32000]
        lat_lims = [6.6996E6, 6.705E6]
        lon_lim_idxs = [np.where(lon_mosaic[0, :] == lon_lims[0])[0][0], np.where(lon_mosaic[0, :] == lon_lims[1])[0][0]]
        lat_lim_idxs = [np.where(lat_mosaic[:, 0] == lat_lims[0])[0][0], np.where(lat_mosaic[:, 0] == lat_lims[1])[0][0]]
        lon_mosaic_crop = lon_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
        lat_mosaic_crop = lat_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
        imgs_mosaic_crop = imgs_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
        # cmap = copy.copy(plt.get_cmap('magma_r'))
        cmap = copy.copy(plt.get_cmap('binary'))
        imgs_mosaic_crop = np.ma.masked_where(imgs_mosaic_crop == 0, imgs_mosaic_crop)  # set mask where height is 0, to be converted to another color
        cmap.set_bad(color='skyblue')  # color where height == 0
        plt.figure(dpi=400)
        plt.title('ANN predictions of $I_u$')
        bbox = ((lon_mosaic_crop.min(), lon_mosaic_crop.max(),
                 lat_mosaic_crop.min(), lat_mosaic_crop.max()))
        imshow = plt.imshow(imgs_mosaic_crop, extent=bbox, zorder=0, cmap=cmap, vmin=0, vmax=750)
        main_ax = plt.gca()
        wrax = {}
        # for pt_idx, pt in enumerate(bj_pt_strs):
        #     plt.scatter(bj_coors[pt_idx][0], bj_coors[pt_idx][1], marker='o', facecolors='black', edgecolors='black', s=10, label='Measurement location' if pt==0 else None)

        plt.xlabel('Easting [m]')
        plt.ylabel('Northing [m]')

        # plt.legend() #[handles[idx] for idx in order], [labels[idx] for idx in order], handletextpad=0.1)
        # plt.tight_layout(pad=0.05)

        Iu_min_all_pts = np.min(np.array([dict_Iu_48m_ANN_preds[pt]['Iu'] for pt in bj_pt_strs]))
        Iu_max_all_pts = np.max(np.array([dict_Iu_48m_ANN_preds[pt]['Iu'] for pt in bj_pt_strs]))
        rose_radius = np.ones(11) * 0.4
        for pt_idx, pt in enumerate(bj_pt_strs):
            if True:  # pt_idx%10==0: # if point index is even:
                ########### WIND ROSES
                wd = np.array(dict_Iu_48m_ANN_preds[pt]['sector'])
                Iu = np.array(dict_Iu_48m_ANN_preds[pt]['Iu'])
                Iu_min = np.min(Iu)
                Iu_max = np.max(Iu)
                wrax[pt] = inset_axes(main_ax,
                                      width=rose_radius[pt_idx],  # size in inches
                                      height=rose_radius[pt_idx],  # size in inches
                                      loc='center',  # center bbox at given position
                                      bbox_to_anchor=(bj_coors[pt_idx][0], bj_coors[pt_idx][1]),  # position of the axe
                                      bbox_transform=main_ax.transData,  # use data coordinate (not axe coordinate)
                                      axes_class=WindroseAxes)  # specify the class of the axe
                # print(f'Min: {(Iu_min-Iu_min_all_pts)/(Iu_max_all_pts-Iu_min_all_pts)}')
                # print(f'Max; {(Iu_max-Iu_min_all_pts)/(Iu_max_all_pts-Iu_min_all_pts)}')
                wrax[pt].bar(wd, Iu, opening=1.0, nsector=360, cmap=truncate_colormap(matplotlib.pyplot.cm.Reds, (Iu_min - Iu_min_all_pts) / (Iu_max_all_pts - Iu_min_all_pts),
                                                                                      (Iu_max - Iu_min_all_pts) / (Iu_max_all_pts - Iu_min_all_pts)))
                wrax[pt].tick_params(labelleft=False, labelbottom=False)
                wrax[pt].patch.set_alpha(0)
                wrax[pt].axis('off')

        cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=Iu_min_all_pts, vmax=Iu_max_all_pts), cmap=matplotlib.pyplot.cm.Reds), ax=main_ax)
        cb.set_label('$I_u$')
        main_ax.axis('off')
        plt.savefig('plots/ANN_preds_zoomin.png')
        plt.show()
        plt.close()

# alpha = np.zeros(g_node_coor.shape[0])
# f_min = 0.002
# f_max = 0.5
# n_freq = 128
# f_array = np.linspace(f_min, f_max, n_freq)


# Nw = NwOneCase()
# Nw.set_df_WRF(sort_by='wd_var')
# Nw.set_structure(g_node_coor, p_node_coor, alpha)
# Nw.set_Nw_wind(df_WRF_idx=-20, f_array=f_array)
# Nw.plot_U(df_WRF_idx=Nw.df_WRF_idx)
# # Nw.plot_Ii_at_WRF_points()

# Nw_all_cases = NwAllCases()
# Nw_all_cases.set_df_WRF(sort_by='wd_var')
# Nw_all_cases.set_structure(g_node_coor, p_node_coor, alpha)
# Nw_all_cases.set_Nw_wind(n_Nw_cases='all', f_array=f_array)


