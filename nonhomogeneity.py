"""
created: 2021
author: Bernardo Costa
email: bernamdc@gmail.com

"Nw" is short for "Nonhomogeneous wind" and is extensively used in this script
"""

import os
import copy
import datetime
import json
import netCDF4
import warnings
import numpy as np
import pandas as pd
from scipy import interpolate
from buffeting import U_bar_func, beta_0_func, RP, Pb_func, Ai_func, iLj_func, Cij_func
from mass_and_stiffness_matrix import stiff_matrix_func, stiff_matrix_12b_local_func, stiff_matrix_12c_local_func, linmass, SDL
from simple_5km_bridge_geometry import g_node_coor, p_node_coor, g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func
from transformations import T_LsGs_3g_func, T_GsNw_func, from_cos_sin_to_0_2pi
from WRF_500_interpolated.create_minigrid_data_from_raw_WRF_500_data import n_bridge_WRF_nodes, bridge_WRF_nodes_coor_func, earth_R
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


# # Testing consistency between WRF nodes in bridge coordinates and in (lats,lons)
# test_WRF_node_consistency = True
# if test_WRF_node_consistency: # Make sure that the R and arc length are consistent in: 1) the bridge model and 2) WRF nodes (the arc along which WRF data is collected)
#     assert (R==5000 and arc_length==5000)
#     WRF_node_coor_2 = np.deg2rad(bridge_WRF_nodes_coor_func()) * earth_R
#     WRF_node_coor_2[:, 1] = -WRF_node_coor_2[:, 1]  # attention! bridge_WRF_nodes_coor_func() gives coor in (lats,lons) which is a left-hand system! This converts to right-hand (lats,-lons).
#     WRF_node_coor_2 = (WRF_node_coor_2 - WRF_node_coor_2[0]) @ np.array([[np.cos(np.deg2rad(-10)), -np.sin(np.deg2rad(-10))], [np.sin(np.deg2rad(-10)), np.cos(np.deg2rad(-10))]])
#     assert np.allclose(WRF_node_coor[:, :2], WRF_node_coor_2)

# # todo: delete below
# from create_WRF_data_at_bridge_nodes_from_minigrid_data import Nw_ws_wd_func
# Nw_ws_wd_func()
# Nw_beta_DB_cos = interpolate_from_WRF_nodes_to_g_nodes(np.cos(wd_to_plot, dtype=float))
# Nw_beta_DB_sin = interpolate_from_WRF_nodes_to_g_nodes(np.sin(wd_to_plot, dtype=float))
# Nw_beta_DB = from_cos_sin_to_0_2pi(Nw_beta_DB_cos, Nw_beta_DB_sin, out_units='rad')
# Nw_beta_0 = np.array([beta_0_func(i) for i in Nw_beta_DB])
# print(np.rad2deg(Nw_beta_0))
# Nw_theta_0 = (copy.deepcopy(Nw_beta_0) * 0 + 1) * np.deg2rad(0)
# alpha = (copy.deepcopy(Nw_beta_0) * 0 + 1) *  np.deg2rad(0)
# # # todo: delete above


def Nw_U_bar_func(g_node_coor, Nw_U_bar_at_WRF_nodes, force_Nw_U_and_N400_U_to_have_same=None):
    """
    Returns a vector of Nonhomogeneous mean wind at each of the g_nodes
    force_Nw_and_U_bar_to_have_same_avg : None, 'mean', 'energy'. force the Nw_U_bar_at_WRF_nodes to have the same e.g. mean 1, and thus when multiplied with U_bar, the result will have the same mean (of all nodes) wind
    """
    assert Nw_U_bar_at_WRF_nodes.shape[-1] == n_WRF_nodes
    U_bar_10min = U_bar_func(g_node_coor)  # N400
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
    Nw_U_bar shape: (n_cases, n_nodes)
    Returns a homogeneous wind velocity field, equivalent to the input Nw_U_bar in terms of force_Nw_U_bar_and_U_bar_to_have_same
    force_Nw_U_bar_and_U_bar_to_have_same: None, 'mean', 'energy'. force the U_bar_equivalent to have the same mean or energy 1 as Nw_U_bar
    """
    if force_Nw_U_bar_and_U_bar_to_have_same is None:
        U_bar_equivalent = U_bar_func(g_node_coor)
    elif force_Nw_U_bar_and_U_bar_to_have_same == 'mean':
        U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.mean(Nw_U_bar, axis=1)[:,None]
        assert all(np.isclose(np.mean(Nw_U_bar, axis=1)[:,None], np.mean(U_bar_equivalent, axis=1)[:,None]))
    elif force_Nw_U_bar_and_U_bar_to_have_same == 'energy':
        U_bar_equivalent = np.ones(Nw_U_bar.shape) * np.sqrt(np.mean(Nw_U_bar**2, axis=1)[:,None])
        assert all(np.isclose(np.mean(Nw_U_bar ** 2, axis=1)[:,None], np.mean(U_bar_equivalent ** 2, axis=1)[:,None]))  # same energy = same mean(U**2))
    return U_bar_equivalent


def Nw_beta_and_theta_bar_func(g_node_coor, Nw_beta_0, Nw_theta_0, alpha):
    """Returns the Nonhomogeneous beta_bar and theta_bar at each node, relative to the mean of the axes of the adjacent elements.
    Note: the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
    n_g_nodes = len(g_node_coor)
    assert len(Nw_beta_0) == len(Nw_theta_0) == n_g_nodes
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
    T_GsNw = T_GsNw_func(Nw_beta_0, Nw_theta_0)
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



class NwClass:
    """
    Non-homogeneous wind class
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
            self.equiv_Hw_U_bar = None  # Equivalent Homogeneous mean wind speeds at all the girder nodes

    def set_WRF_df(self, U_tresh=12, tresh_requirement_type='any', sort_by='time'):
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
        self.df_WRF = df_WRF.loc[idxs_sorted_by[sort_by]]
        self.props_WRF['df_WRF'] = {'U_tresh':U_tresh, 'tresh_requirement_type':tresh_requirement_type, 'sorted_by':sort_by}

    def set_structure(self, g_node_coor, p_node_coor, alpha):
        self.g_node_coor = g_node_coor
        self.p_node_coor = p_node_coor
        self.alpha = alpha
        # Reseting wind
        self.__init__(reset_structure=False, reset_WRF_database=False, reset_wind=True)

    # todo: wrapper function def set_wind(...arguments_from_all_functions...)

    def set_U_bar_beta_DB_beta_0_theta_0(self, df_WRF_idx, force_Nw_U_and_N400_U_to_have_same=None):
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
        self.U_bar = Nw_U_bar
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

    def set_beta_and_theta_bar(self):
        """Returns the Nonhomogeneous beta_bar and theta_bar at each node, relative to the mean of the axes of the adjacent elements.
        Note: the mean of -179 deg and 178 deg should be 179.5 deg and not -0.5 deg. See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
        assert self.beta_DB is not None
        assert self.g_node_coor is not None
        g_node_coor = self.g_node_coor
        alpha = self.alpha
        Nw_beta_0 = self.beta_0
        Nw_theta_0 = self.theta_0
        n_g_nodes = len(g_node_coor)
        assert len(Nw_beta_0) == len(Nw_theta_0) == n_g_nodes
        T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
        T_GsNw = T_GsNw_func(Nw_beta_0, Nw_theta_0)
        T_LsNw = np.einsum('nij,njk->nik', T_LsGs, T_GsNw)
        U_Gw_norm = np.array([1, 0, 0])  # U_Gw = (U, 0, 0), so the normalized U_Gw_norm is (1, 0, 0)
        U_Ls = np.einsum('nij,j->ni', T_LsNw, U_Gw_norm)
        Ux = U_Ls[:, 0]
        Uy = U_Ls[:, 1]
        Uz = U_Ls[:, 2]
        Uxy = np.sqrt(Ux ** 2 + Uy ** 2)
        Nw_beta_bar = np.array([-np.arccos(Uy[i] / Uxy[i]) * np.sign(Ux[i]) for i in range(len(g_node_coor))])
        Nw_theta_bar = np.array([np.arcsin(Uz[i] / 1) for i in range(len(g_node_coor))])
        self.beta_bar = Nw_beta_bar
        self.theta_bar = Nw_theta_bar

    def set_Ii(self, model='ANN'):
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

    def set_S_a(self, f_array):
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




    def set_S_aa(self, cospec_type=2):
        """
        In Hertz. The input coordinates are in Global Structural Gs (even though Gw is calculated and used in this function)
        """
        g_node_coor = self.g_node_coor  # shape (g_node_num,3)
        U_bar = self.U_bar
        iLj = self.iLj
        S_a = self.S_a
        beta_0 = self.beta_0
        theta_0 = self.theta_0
        Cij = Cij_func(cond_rand_C=False)
        n_g_nodes = len(g_node_coor)

        # Difficult part. We need a cross-transformation matrix T_GsNw_avg, which is an array with shape (n_g_nodes, n_g_nodes, 3) where each (i,j) entry is the T_GsNw_avg, where Nw_avg is the avg. between Nw_i (at node i) and Nw_j (at node j)
        T_GsNw = T_GsNw_func(beta_0, theta_0)  # shape (n_g_nodes,3,3)
        T_GsNw_avg = (T_GsNw[:,None] + T_GsNw) / 2  # from shape (n_g_nodes,3,3), to shape (n_g_nodes,n_g_nodes, 3, 3)! Average between the axes of both points
        U_bar_avg = (U_bar[:, None] + U_bar) / 2  # from shape (n_g_nodes) to shape (n_g_nodes,n_g_nodes)
        g_node_coor_expanded = np.repeat(g_node_coor[:,None,:], n_g_nodes, axis=1)  # (g_node_num,g_node_num,3)
        g_node_coor_Nw = np.einsum('mni,mnij->mnj', g_node_coor_expanded, T_GsNw_avg)
        delta_xyz = np.absolute(g_node_coor_Nw - g_node_coor_Nw)  # shape (n_g_nodes, n_g_nodes,3)

        # BENCHMARK 1: SLOW version -- INCOMPLETE --, but more intuitive, to validate the results:
        from transformations import T_GsGw_func
        T_GsNw_2 = np.array([T_GsGw_func(beta_0[i], theta_0[i]) for i in range(n_g_nodes)])
        T_GsNw_avg_2 = np.zeros((n_g_nodes, n_g_nodes,3,3))
        for i in range(n_g_nodes):
            for j in range(n_g_nodes):
                T_GsNw_avg_2[i,j,:,:] = ( T_GsGw_func(beta_0[i], theta_0[i]) + T_GsGw_func(beta_0[j], theta_0[j]) ) / 2
        g_node_coor_Nw_simple = np.einsum('ni,nij->nj', g_node_coor, T_GsNw_2)

        # BENCHMARK 2: ANOTHER METHOD
        X_Gs  = np.array([1, 0, 0])
        Y_Gs  = np.array([0, 1, 0])
        Z_Gs  = np.array([0, 0, 1])
        Xu_Nw = np.array([1, 0, 0])
        Yv_Nw = np.array([0, 1, 0])
        Zw_Nw = np.array([0, 0, 1])
        Nw_Xu_in_Gs = np.array([T_GsNw_2[i] @ Xu_Nw for i in range(n_g_nodes)])
        Nw_Yv_in_Gs = np.array([T_GsNw_2[i] @ Yv_Nw for i in range(n_g_nodes)])
        Nw_Zw_in_Gs = np.array([T_GsNw_2[i] @ Zw_Nw for i in range(n_g_nodes)])
        Nw_Xu_in_Gs_avg = np.zeros((n_g_nodes, n_g_nodes, 3))
        Nw_Yv_in_Gs_avg = np.zeros((n_g_nodes, n_g_nodes, 3))
        Nw_Zw_in_Gs_avg = np.zeros((n_g_nodes, n_g_nodes, 3))
        def cos(i,j):
            return np.dot(i, j) / (np.linalg.norm(i) * np.linalg.norm(j))

        T_GsNw_avg_2 = np.zeros((n_g_nodes, n_g_nodes, 3, 3))
        for i in range(n_g_nodes):
            for j in range(n_g_nodes):
                Nw_Xu_in_Gs_avg[i,j] = (Nw_Xu_in_Gs[i] + Nw_Xu_in_Gs[j]) / 2
                Nw_Yv_in_Gs_avg[i,j] = (Nw_Yv_in_Gs[i] + Nw_Yv_in_Gs[j]) / 2
                Nw_Zw_in_Gs_avg[i,j] = (Nw_Zw_in_Gs[i] + Nw_Zw_in_Gs[j]) / 2
                T_GsNw_avg_2[i,j] = np.array([[cos(X_Gs, Nw_Xu_in_Gs_avg[i,j]), cos(X_Gs, Nw_Yv_in_Gs_avg[i,j]), cos(X_Gs, Nw_Zw_in_Gs_avg[i,j])],
                                              [cos(Y_Gs, Nw_Xu_in_Gs_avg[i,j]), cos(Y_Gs, Nw_Yv_in_Gs_avg[i,j]), cos(Y_Gs, Nw_Zw_in_Gs_avg[i,j])],
                                              [cos(Z_Gs, Nw_Xu_in_Gs_avg[i,j]), cos(Z_Gs, Nw_Yv_in_Gs_avg[i,j]), cos(Z_Gs, Nw_Zw_in_Gs_avg[i,j])]])

        # BENCHMARK 3: YET ANOTHER METHOD: CALCULATE delta_xyz as if the Nw vector was that of the left node, then as that of right node, and finally average both delta_xyz
        np.all(np.isclose(T_GsNw_2,T_GsNw))
        np.all(np.isclose(T_GsNw_avg,T_GsNw_avg_2))  # use  rtol=0.08 to get True...
        print(np.max(np.abs(T_GsNw_avg-T_GsNw_avg_2)))
        np.all(np.isclose(g_node_coor_Nw, g_node_coor_Nw_2))


        # COMPARING WITH OLD HOMOGENEOUS FORMULATION
        from transformations import T_GsGw_func
        OLD_T_GsNw = T_GsGw_func(beta_0[0], theta_0[0])
        OLDg_node_coor_Gw = np.einsum('ni,ij->nj', g_node_coor, OLD_T_GsNw)  # Nodes in wind coordinates. X along, Y across, Z vertical
        OLD_delta_xyz = np.absolute(OLDg_node_coor_Gw[:, None] - OLDg_node_coor_Gw[:])  # shape (n_g_nodes, n_g_nodes,3)



        # Cross-spectral density of fluctuating wind components. Adapted Davenport formulation. Note that coherence drops down do negative values, where it stays for quite some distance:


        # Alternative 1: LD Zhu coherence and cross-spectrum. Developed in radians? So it is converted to Hertz in the end.
        if cospec_type == 1:
            raise NotImplementedError
        if cospec_type == 2:
            f_hat_aa = np.einsum('f,mna->fmna', f_array,
                                 np.divide(np.sqrt((Cij[:, 0] * delta_xyz[:, :, 0, None]) ** 2 + (Cij[:, 1] * delta_xyz[:, :, 1, None]) ** 2 + (Cij[:, 2] * delta_xyz[:, :, 2, None]) ** 2),
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
        return S_aa






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

    def plot_U(self, df_WRF_idx):
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
        plt.quiver(*np.array([lons_bridge, lats_bridge]), -ws_to_plot * np.sin(wd_to_plot), -ws_to_plot * np.cos(wd_to_plot), color=ws_colors, angles='uv', scale=100, width=0.015, headlength=3, headaxislength=3)
        cbar = plt.colorbar(sm)
        cbar.set_label('U [m/s]')
        plt.title(f'Nw U_bar')
        plt.xlim(5.366, 5.386)
        plt.ylim(60.082, 60.133)
        plt.xlabel('Longitude [$\degree$]')
        plt.ylabel('Latitude [$\degree$]')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()



#todo:delete
alpha = np.zeros(g_node_coor.shape[0])
f_min = 0.002
f_max = 0.5
n_freq = 128
f_array = np.linspace(f_min, f_max, n_freq)
#todo:delete

Nw = NwClass()
Nw.set_WRF_df(sort_by='ws_max')
Nw.set_structure(g_node_coor, p_node_coor, alpha)
Nw.set_U_bar_beta_DB_beta_0_theta_0(df_WRF_idx=-2)
Nw.plot_U(df_WRF_idx=-2)
Nw.set_beta_and_theta_bar()
Nw.set_Ii()
Nw.set_S_a(f_array)
Nw.set_S_aa()


Nw.U_bar
np.rad2deg(Nw.beta_DB)
Nw.beta_0
Nw.beta_bar
Nw.Ii
Nw.S_a.shape



