"""
created: 2019
author: Bernardo Costa
email: bernamdc@gmail.com
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mass_and_stiffness_matrix import mass_matrix_func, stiff_matrix_func, geom_stiff_matrix_func
from simple_5km_bridge_geometry import g_node_coor, p_node_coor, g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func
from transformations import mat_Ls_node_Gs_node_all_func, from_cos_sin_to_0_2pi, beta_within_minus_Pi_and_Pi_func, mat_6_Ls_node_12_Ls_elem_girder_func, NumpyEncoder
from modal_analysis import modal_analysis_func, simplified_modal_analysis_func
from static_loads import static_wind_func
from WRF_500_interpolated.create_minigrid_data_from_raw_WRF_500_data import n_bridge_WRF_nodes, bridge_WRF_nodes_coor_func, earth_R
# from create_WRF_data_at_bridge_nodes_from_minigrid_data import Nw_ws_wd_func  # todo: go get this function in the trash folder "old_wrong_files"
from nonhomogeneity import Nw_static_wind_all, NwOneCase, NwAllCases  # Nw_U_bar_func, Nw_beta_and_theta_bar_func, interpolate_from_WRF_nodes_to_g_nodes, n_WRF_nodes, WRF_node_coor, U_bar_equivalent_to_Nw_U_bar
from buffeting import buffeting_FD_func, rad, deg, list_of_cases_FD_func, parametric_buffeting_FD_func, U_bar_func, buffeting_TD_func, list_of_cases_TD_func, parametric_buffeting_TD_func, beta_0_func
from my_utils import normalize, normalize_mode_shape
import copy
from static_loads import static_dead_loads_func, R_loc_func

start_time = time.time()
run_modal_analysis = False
run_DL = False  # include Dead Loads, for all analyses.
run_sw_for_modal = False # include Static wind for the modal_analysis_after_static_loads. For other analyses use include_SW (inside buffeting function).
run_new_Nw_sw = False

run_modal_analysis_after_static_loads = False
generate_new_C_Ci_grid = True  # todo: attention!

########################################################################################################################
# Initialize structure:
########################################################################################################################
from simple_5km_bridge_geometry import g_node_coor, p_node_coor

g_node_num = len(g_node_coor)
g_elem_num = g_node_num - 1
p_node_num = len(p_node_coor)
all_node_num = g_node_num + p_node_num
all_elem_num = g_elem_num + p_node_num
R_loc = np.zeros((all_elem_num, 12))  # No initial element internal forces
D_loc = np.zeros((all_node_num, 6))  # No initial nodal displacements
girder_N = copy.deepcopy(R_loc[:g_elem_num, 0])  # No girder axial forces
c_N = copy.deepcopy(R_loc[g_elem_num:, 0])  # No columns axial forces
alpha = copy.deepcopy(D_loc[:g_node_num, 3])  # No girder nodes torsional rotations

########################################################################################################################
# Modal analysis:
########################################################################################################################
if run_modal_analysis:
    # Importing mass and stiffness matrices:
    mass_matrix = mass_matrix_func(g_node_coor, p_node_coor, alpha)  # (N)
    stiff_matrix = stiff_matrix_func(g_node_coor, p_node_coor, alpha)  # (N)
    geom_stiff_matrix = geom_stiff_matrix_func(g_node_coor, p_node_coor, girder_N, c_N, alpha)

    _, _, omegas, shapes = simplified_modal_analysis_func(mass_matrix, stiff_matrix - geom_stiff_matrix)
    periods = 2*np.pi/omegas

    # New modal plots
    def plot_mode_shapes(shapes, omegas, g_node_coor, p_node_coor, n_modes_plot=100):
        """
        global_shapes_Gs: global matrix. shape(n_all_modes, n_all_DOF), in Gs coordinates. Usually square matrix
        """
        periods = 2 * np.pi / omegas
        n_g_nodes = len(g_node_coor)
        n_p_nodes = len(p_node_coor)

        flat_shapes_Gs = shapes[:n_modes_plot].copy()
        assert len(flat_shapes_Gs.shape) == 2
        assert flat_shapes_Gs.shape[1] == (n_g_nodes + n_p_nodes) * 6
        assert (flat_shapes_Gs.shape[1]/6).is_integer()
        shapes_Gs = np.reshape(flat_shapes_Gs, (flat_shapes_Gs.shape[0], int(flat_shapes_Gs.shape[1]/6), 6))
        shapes_Ls = np.array([mat_Ls_node_Gs_node_all_func(shapes_Gs[f], g_node_coor, p_node_coor, alpha) for f in range(shapes_Gs.shape[0])])  # this matrix requires the shapes matrix to be in format 'ni', with n: number of nodes and i: 6 DOF.
        g_shapes_Ls = shapes_Ls[:, :n_g_nodes]
        g_shapes_Ls = np.array([normalize_mode_shape(x) for x in g_shapes_Ls])

        # Plotting
        fig, axs = plt.subplots(10, 5, sharex=True,sharey=True, dpi=200, figsize=(8,10.4))
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i, ax in enumerate(axs.ravel()):
            ax.plot(np.linspace(0, arc_length, n_g_nodes), g_shapes_Ls[i, :, 1], c='tab:green', label='Horizontal (y-axis)')
            ax.plot(np.linspace(0, arc_length, n_g_nodes), g_shapes_Ls[i, :, 2], c='tab:blue', label='Vertical (z-axis)')
            ax.plot(np.linspace(0, arc_length, n_g_nodes), g_shapes_Ls[i, :, 3], c='tab:orange', label='Torsional (rx-axis)')
            ax.text(0.049, 0.65, '$T_{'+f'{i+1}'+'}='+f'{np.round(periods[i],1)}s$', bbox={'fc': 'white', 'alpha': 0.6})
            ax.set_xticks([0, 2500, 5000])
        fig.supxlabel('x-axis [m]')
        plt.tight_layout(pad=0.5) # w_pad=0.04, h_pad=0.06)
        plt.savefig(r'_mode_shapes/first_50_modes.png')
        plt.show()

        h, l = ax.get_legend_handles_labels()
        plt.figure(dpi=500, figsize=(6,1))
        plt.axis('off')
        plt.legend(h,l, ncol=3)
        plt.tight_layout()
        plt.savefig(r'_mode_shapes/new_mode_legend.png')
        plt.show()

        fig, axs = plt.subplots(10, 5, sharex=True,sharey=True, dpi=200, figsize=(8,10.4))
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i, ax in enumerate(axs.ravel()):
            j = i + int(n_modes_plot/2)
            ax.plot(np.linspace(0, arc_length, n_g_nodes), g_shapes_Ls[j, :, 1], c='tab:green', label='Horizontal (y-axis)')
            ax.plot(np.linspace(0, arc_length, n_g_nodes), g_shapes_Ls[j, :, 2], c='tab:blue', label='Vertical (z-axis)')
            ax.plot(np.linspace(0, arc_length, n_g_nodes), g_shapes_Ls[j, :, 3], c='tab:orange', label='Torsional (rx-axis)')
            ax.text(0.049, 0.65, '$T_{'+f'{j+1}'+'}='+f'{np.round(periods[j],1)}s$', bbox={'fc': 'white', 'alpha': 0.6})
            ax.set_xticks([0, 2500, 5000])
        fig.supxlabel('x-axis [m]')
        plt.tight_layout(pad=0.5) # w_pad=0.04, h_pad=0.06)
        plt.savefig(r'_mode_shapes/other_50_modes.png')
        plt.show()
        return None
    plot_mode_shapes(shapes, omegas, g_node_coor, p_node_coor)

    # OLD plots:
    def plot_mode_shape_old(n_modes_plot):
        deformation_ratio = 200
        for m in range(n_modes_plot):
            # Girder:
            g_shape_v1 = shapes[m, 0: g_node_num * 6: 6]  # girder shape. v1 is vector 1.
            g_shape_v2 = shapes[m, 1: g_node_num * 6: 6]
            g_shape_v3 = shapes[m, 2: g_node_num * 6: 6]
            g_shape_undeformed_X = g_node_coor[:, 0]
            g_shape_undeformed_Y = g_node_coor[:, 1]
            g_shape_undeformed_Z = g_node_coor[:, 2]
            g_shape_deformed_X = g_shape_undeformed_X + deformation_ratio * g_shape_v1
            g_shape_deformed_Y = g_shape_undeformed_Y + deformation_ratio * g_shape_v2
            g_shape_deformed_Z = g_shape_undeformed_Z + deformation_ratio * g_shape_v3
            # Pontoons:
            p_shape_v1 = shapes[m, g_node_num * 6 + 0: g_node_num * 6 + p_node_num * 6: 6]
            p_shape_v2 = shapes[m, g_node_num * 6 + 1: g_node_num * 6 + p_node_num * 6: 6]
            p_shape_v3 = shapes[m, g_node_num * 6 + 2: g_node_num * 6 + p_node_num * 6: 6]
            p_shape_undeformed_X = p_node_coor[:, 0]
            p_shape_undeformed_Y = p_node_coor[:, 1]
            p_shape_undeformed_Z = p_node_coor[:, 2]
            p_shape_deformed_X = p_shape_undeformed_X + deformation_ratio * p_shape_v1
            p_shape_deformed_Y = p_shape_undeformed_Y + deformation_ratio * p_shape_v2
            p_shape_deformed_Z = p_shape_undeformed_Z + deformation_ratio * p_shape_v3
            # Plotting:
            fig, ax = plt.subplots(2,1,sharex=True,sharey=False, figsize=(6,5))
            fig.subplots_adjust(hspace=0.2)  # horizontal space between axes
            fig.suptitle('Mode shape ' + str(m + 1) + '.  T = ' + str(round(periods[m], 1)) + ' s.  Scale = ' + str(deformation_ratio), fontsize=15)
            ax[0].set_title('X-Y plane')
            ax[1].set_title('X-Z plane')
            ax[0].plot(g_shape_undeformed_X, g_shape_undeformed_Y, label='Undeformed', color='grey', alpha=0.3)
            ax[0].plot(g_shape_deformed_X, g_shape_deformed_Y, label='Deformed', color='orange')
            ax[0].scatter(p_shape_undeformed_X, p_shape_undeformed_Y, color='grey', alpha=0.3, s=10)
            ax[0].scatter(p_shape_deformed_X, p_shape_deformed_Y, color='orange', s=10)
            ax[1].plot(g_shape_undeformed_X, g_shape_undeformed_Z, label='Undeformed', color='grey', alpha=0.3)
            ax[1].plot(g_shape_deformed_X, g_shape_deformed_Z, label='Deformed', color='orange')
            ax[1].scatter(p_shape_undeformed_X, p_shape_undeformed_Z, color='grey', alpha=0.3, s=10)
            ax[1].scatter(p_shape_deformed_X, p_shape_deformed_Z, color='orange', s=10)
            ax[0].grid()
            ax[1].grid()
            ax[0].axis('equal')
            # ax[1].axis('equal')
            # ax[0].set_ylim([-1000, 500])  # ax[0].axis('equal') forces other limits than those defined here
            # ax[1].set_ylim([-1000, 1000])
            ax[1].set_xlabel('[m]')
            plt.tight_layout()
            for i in [0,1]:
                for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
                    item.set_fontsize(14)
            plt.savefig('_mode_shapes/' + str(m + 1) + '.png', dpi=300)
            handles,labels = plt.gca().get_legend_handles_labels()
            plt.close()
            # Plotting leggend
            empty_ax = [None] * 2
            empty_ax[0] = plt.scatter(0,0, color='grey', alpha=0.3, s=10)
            empty_ax[1] = plt.scatter(0,0, color='orange', s=10, label='Pontoons')
            plt.close()
            plt.figure(figsize=(6, 3), dpi=300)
            plt.axis("off")
            from matplotlib.legend_handler import HandlerTuple
            plt.legend(handles + [tuple(empty_ax)], labels + ['Pontoons'], handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=3)
            plt.tight_layout()
            plt.savefig(r'_mode_shapes/mode_shape_legend.png')
            plt.close()
        print("--- %s seconds ---" % (time.time() - start_time))
        return None
    # plot_mode_shape_old(n_modes_plot = 10)

########################################################################################################################
# Dead loads analysis (DL)
########################################################################################################################
if run_DL:
    # Displacements
    g_node_coor_DL, p_node_coor_DL, D_glob_DL = static_dead_loads_func(g_node_coor, p_node_coor, alpha)
    D_loc_DL = mat_Ls_node_Gs_node_all_func(D_glob_DL, g_node_coor, p_node_coor, alpha)  # orig. coord used.
    alpha_DL = copy.deepcopy(D_loc_DL[:g_node_num, 3])  # Global nodal torsional rotation.
    # Internal forces
    R_loc_DL = R_loc_func(D_glob_DL, g_node_coor, p_node_coor, alpha)  # orig. coord. + displacem. used to calc. R.
    girder_N_DL = copy.deepcopy(R_loc_DL[:g_elem_num, 0])  # local girder element axial force. Positive = compression!
    c_N_DL = copy.deepcopy(R_loc_DL[g_elem_num:, 0])  # local column element axial force Positive = compression!
    # Updating structure. Subsequent analyses will take the dead-loaded structure as input.
    g_node_coor, p_node_coor = copy.deepcopy(g_node_coor_DL), copy.deepcopy(p_node_coor_DL)
    R_loc += copy.deepcopy(R_loc_DL)  # element local forces
    D_loc += copy.deepcopy(D_loc_DL)  # nodal global displacements. Includes the alphas.
    girder_N += copy.deepcopy(girder_N_DL)
    c_N += copy.deepcopy(c_N_DL)
    alpha += copy.deepcopy(alpha_DL)

########################################################################################################################
# Aerodynamic coefficients grid
########################################################################################################################
project_path = sys.path[1]  # To be used in Python Console! When a console is opened, the current project path should be automatically added to sys.path.
C_Ci_grid_path = project_path + r'\\aerodynamic_coefficients\\C_Ci_grid.npy'
# Deleting aerodynamic coefficient grid input file, for a new one to be created.
if not os.path.exists(C_Ci_grid_path):
    print('No C_Ci_grid.npy found. New one will be created.')
elif generate_new_C_Ci_grid and os.path.exists(C_Ci_grid_path):
    os.remove(C_Ci_grid_path)
    print('C_Ci_grid.npy found and deleted.')
else:  # file exists but generate_new_C_Ci_grid == False:
    print('Warning: Already existing C_Ci_grid.npy file will be used!')

########################################################################################################################
# Separate static wind (sw) analysis, with DL as input and modal analysis output. Buffeting has its own SW analysis.
########################################################################################################################
if run_modal_analysis_after_static_loads:
    # Temporary structure, only for new modal analysis
    g_node_coor_temp, p_node_coor_temp = copy.deepcopy(g_node_coor), copy.deepcopy(p_node_coor)
    R_loc_temp = copy.deepcopy(R_loc)  # element local forces
    D_loc_temp = copy.deepcopy(D_loc)  # nodal global displacements. Includes the alphas.
    girder_N_temp = copy.deepcopy(girder_N)
    c_N_temp = copy.deepcopy(c_N)
    alpha_temp = copy.deepcopy(alpha)
    if run_sw_for_modal:
        # Displacements
        g_node_coor_sw, p_node_coor_sw, D_glob_sw = static_wind_func(g_node_coor, p_node_coor, alpha, U_bar=U_bar_func(g_node_coor), beta_DB=rad(280), theta_0=rad(0), aero_coef_method='2D_fit_cons',
                                                                     n_aero_coef=6, skew_approach='3D')
        D_loc_sw = mat_Ls_node_Gs_node_all_func(D_glob_sw, g_node_coor, p_node_coor, alpha)
        alpha_sw = copy.deepcopy(D_loc_sw[:g_node_num, 3])  # Global nodal torsional rotation.
        # Internal forces
        R_loc_sw = R_loc_func(D_glob_sw, g_node_coor, p_node_coor, alpha)  # orig. coord. + displacem. used to calc. R.
        girder_N_sw = copy.deepcopy(R_loc_sw[:g_elem_num, 0])  # local girder element axial force. Positive = compression!
        c_N_sw = copy.deepcopy(R_loc_sw[g_elem_num:, 0])  # local column element axial force Positive = compression!
        # Temporary structure is updated. This is a separate analysis for the new modal analysis only.
        g_node_coor_temp, p_node_coor_temp = copy.deepcopy(g_node_coor_sw), copy.deepcopy(p_node_coor_sw)
        R_loc_temp += copy.deepcopy(R_loc_sw)  # element local forces
        D_loc_temp += copy.deepcopy(D_loc_sw)  # nodal global displacements. Includes the alphas.
        girder_N_temp += copy.deepcopy(girder_N_sw)
        c_N_temp += copy.deepcopy(c_N_sw)
        alpha_temp += copy.deepcopy(alpha_sw)
    # Modal analysis:
    mass_matrix = mass_matrix_func(g_node_coor_temp, p_node_coor_temp, alpha_temp)
    stiff_matrix = stiff_matrix_func(g_node_coor_temp, p_node_coor_temp, alpha_temp)
    geom_stiff_matrix = geom_stiff_matrix_func(g_node_coor_temp, p_node_coor_temp, girder_N_temp, c_N_temp, alpha_temp)
    _, _, omegas, shapes = simplified_modal_analysis_func(mass_matrix, stiff_matrix - geom_stiff_matrix)
    periods = 2*np.pi/omegas
    # Plotting:
    def plot_mode_shape_old(n_modes_plot):
        deformation_ratio = 200
        for m in range(n_modes_plot):
            # Girder:
            g_shape_v1 = shapes[m, 0: g_node_num * 6: 6]  # girder shape. v1 is vector 1.
            g_shape_v2 = shapes[m, 1: g_node_num * 6: 6]
            g_shape_v3 = shapes[m, 2: g_node_num * 6: 6]
            g_shape_undeformed_X = g_node_coor_temp[:, 0]
            g_shape_undeformed_Y = g_node_coor_temp[:, 1]
            g_shape_undeformed_Z = g_node_coor_temp[:, 2]
            g_shape_deformed_X = g_shape_undeformed_X + deformation_ratio * g_shape_v1
            g_shape_deformed_Y = g_shape_undeformed_Y + deformation_ratio * g_shape_v2
            g_shape_deformed_Z = g_shape_undeformed_Z + deformation_ratio * g_shape_v3
            # Pontoons:
            p_shape_v1 = shapes[m, g_node_num * 6 + 0: g_node_num * 6 + p_node_num * 6: 6]
            p_shape_v2 = shapes[m, g_node_num * 6 + 1: g_node_num * 6 + p_node_num * 6: 6]
            p_shape_v3 = shapes[m, g_node_num * 6 + 2: g_node_num * 6 + p_node_num * 6: 6]
            p_shape_undeformed_X = p_node_coor_temp[:, 0]
            p_shape_undeformed_Y = p_node_coor_temp[:, 1]
            p_shape_undeformed_Z = p_node_coor_temp[:, 2]
            p_shape_deformed_X = p_shape_undeformed_X + deformation_ratio * p_shape_v1
            p_shape_deformed_Y = p_shape_undeformed_Y + deformation_ratio * p_shape_v2
            p_shape_deformed_Z = p_shape_undeformed_Z + deformation_ratio * p_shape_v3
            # Plotting:
            fig, ax = plt.subplots(2,1,sharex=True,sharey=False)
            fig.subplots_adjust(hspace=0.2)  # horizontal space between axes
            fig.suptitle('Mode ' + str(m + 1) + '.  T = ' + str(round(periods[m], 1)) + ' s.  Scale = ' + str(deformation_ratio) + '. $K_G$ effects included', fontsize=13)
            ax[0].set_title('X-Y plane')
            ax[1].set_title('X-Z plane')
            ax[0].plot(g_shape_undeformed_X, g_shape_undeformed_Y, label='Undeformed', color='grey', alpha=0.3)
            ax[0].plot(g_shape_deformed_X, g_shape_deformed_Y, label='Deformed', color='orange')
            ax[0].scatter(p_shape_undeformed_X, p_shape_undeformed_Y, color='grey', alpha=0.3, s=10)
            ax[0].scatter(p_shape_deformed_X, p_shape_deformed_Y, color='orange', s=10)
            ax[1].plot(g_shape_undeformed_X, g_shape_undeformed_Z, label='Undeformed', color='grey', alpha=0.3)
            ax[1].plot(g_shape_deformed_X, g_shape_deformed_Z, label='Deformed', color='orange')
            ax[1].scatter(p_shape_undeformed_X, p_shape_undeformed_Z, color='grey', alpha=0.3, s=10)
            ax[1].scatter(p_shape_deformed_X, p_shape_deformed_Z, color='orange', s=10)
            ax[0].legend(loc=9)
            ax[0].grid()
            ax[1].grid()
            ax[0].axis('equal')
            # ax[1].axis('equal')
            ax[0].set_ylim([-1000, 500])  # ax[0].axis('equal') forces other limits than those defined here
            ax[1].set_ylim([-1000, 1000])
            ax[1].set_xlabel('[m]')
            plt.savefig('_mode_shapes/static_loads_mode_' + str(m + 1) + '.png')
            plt.close()
        print("--- %s seconds ---" % (time.time() - start_time))
        return None
    plot_mode_shape_old(n_modes_plot = 10)

########################################################################################################################
# Separate nonhomogeneous static wind (Nw_sw) analysis. This will store the results of the Nw_sw analysis into a folder, for efficiency! This folder is later accessed by buffeting.py
########################################################################################################################
static_aero_coef_method = 'cos_rule'
static_skew_approach = '2D_cos_law'
if run_new_Nw_sw:
    # Nw = NwOneCase()
    # Nw.set_df_WRF(sort_by='ws_max')
    # # n_WRF_cases
    # Nw.df_WRF
    # Nw.props_WRF
    # Nw.aux_WRF
    # Nw.set_structure(g_node_coor, p_node_coor, alpha)
    # Nw.set_Nw_wind(df_WRF_idx='all', set_static_wind_only=True)
    # Nw.plot_U(df_WRF_idx=Nw.df_WRF_idx)
    Nw_all = NwAllCases()
    sort_by = 'ws_max'
    n_Nw_cases = 'all'
    Nw_all.set_df_WRF(sort_by=sort_by, U_tresh=18)
    Nw_all.set_structure(g_node_coor, p_node_coor, alpha)
    Nw_all.set_Nw_wind(n_Nw_cases=n_Nw_cases, force_Nw_U_and_N400_U_to_have_same=None, Iu_model='ANN', cospec_type=2, f_array='static_wind_only')
    Nw_all.plot_Ii_at_WRF_points()
    Nw_all.plot_U(df_WRF_idx=-1)

    Nw_all.set_equivalent_Hw_U_bar_and_beta(U_method='quadratic_vector_mean', beta_method='quadratic_vector_mean')
    Nw_all.set_equivalent_Hw_Ii(eqv_Hw_Ii_method='Hw_U*Hw_sigma_i=mean(Nw_U*Nw_sigma_i)')

    print(f'Static analysis. static_aero_coef_method: {static_aero_coef_method}. static_skew_approach: {static_skew_approach}')
    Nw_g_node_coor_all, Nw_p_node_coor_all, Nw_D_glob_all, Nw_D_loc_all, Nw_R_loc_all, Nw_R6g_all = Nw_static_wind_all(g_node_coor, p_node_coor, alpha, Nw_all.U_bar, Nw_all.beta_bar, Nw_all.theta_bar, aero_coef_method=static_aero_coef_method, n_aero_coef=6, skew_approach=static_skew_approach)
    Hw_g_node_coor_all, Hw_p_node_coor_all, Hw_D_glob_all, Hw_D_loc_all, Hw_R_loc_all, Hw_R6g_all = Nw_static_wind_all(g_node_coor, p_node_coor, alpha, Nw_all.equiv_Hw_U_bar, Nw_all.equiv_Hw_beta_bar, Nw_all.equiv_Hw_theta_bar, aero_coef_method=static_aero_coef_method, n_aero_coef=6, skew_approach=static_skew_approach)

    raise Exception  # remove this (used for safety)

    # Other post-processing
    Nw_R6g_abs_all = np.abs(Nw_R6g_all)  # Instead of plotting envelopes of bending moments, it's better to just plot their absolute values
    Hw_R6g_abs_all = np.abs(Hw_R6g_all)  # Instead of plotting envelopes of bending moments, it's better to just plot their absolute values
    Nw_R6g_abs_max = np.max(Nw_R6g_abs_all, axis=0)
    Hw_R6g_abs_max = np.max(Hw_R6g_abs_all, axis=0)

    # Saving dict with all results. Add more features if needed (e.g. Ai, iLj)
    Nw_dict_all_results = {'Nw_g_node_coor':Nw_g_node_coor_all.tolist(), 'Nw_p_node_coor':Nw_p_node_coor_all.tolist(), 'Nw_D_glob':Nw_D_glob_all.tolist(), 'Nw_D_loc':Nw_D_loc_all.tolist(),
                           'Nw_R_loc':Nw_R_loc_all.tolist(), 'n_cases':Nw_all.n_Nw_cases, 'Nw_U_bar':Nw_all.U_bar.tolist(), 'Nw_beta_bar':Nw_all.beta_bar.tolist(), 'Nw_theta_bar':Nw_all.theta_bar.tolist(),
                           'Nw_beta_0':Nw_all.beta_0.tolist(), 'Nw_theta_0':Nw_all.theta_0.tolist(), 'Nw_Ii':Nw_all.Ii.tolist(), # 'Nw_S_a':Nw_all.S_a.tolist(),  # 'Nw_S_aa':Nw_all.S_aa.tolist(),  # Only static wind!. S_aa is not included because it is way too large...
                           'Hw_g_node_coor':Hw_g_node_coor_all.tolist(), 'Hw_p_node_coor':Hw_p_node_coor_all.tolist(), 'Hw_D_glob':Hw_D_glob_all.tolist(), 'Hw_D_loc':Hw_D_loc_all.tolist(),
                           'Hw_R_loc':Hw_R_loc_all.tolist(), 'Hw_U_bar':Nw_all.equiv_Hw_U_bar.tolist(), 'Hw_beta_bar':Nw_all.equiv_Hw_beta_bar.tolist(), 'Hw_theta_bar':Nw_all.equiv_Hw_theta_bar.tolist(),
                           'Hw_beta_0':Nw_all.equiv_Hw_beta_0.tolist(), 'Hw_theta_0':Nw_all.equiv_Hw_theta_0.tolist(), 'Hw_Ii':Nw_all.equiv_Hw_Ii.tolist()}

    # Storing each case in individual json files! If all cases were stored in one file, it would have over 1 Gb and buffeting.py would be too slow when opening it every time to only access 1 case
    for n in range(Nw_dict_all_results['n_cases']):   # Nw_dict_all_results['n_cases']):
        Nw_dict_1_case = {}
        for key in Nw_dict_all_results.keys():
            if isinstance(Nw_dict_all_results[key], list):
                Nw_dict_1_case[key]=Nw_dict_all_results[key][n]
        with open(r'intermediate_results\\static_wind_'+static_aero_coef_method+r'\\Nw_dict_'+str(n)+'.json', 'w', encoding='utf-8') as f:
            json.dump(Nw_dict_1_case, f, ensure_ascii=False, indent=4)

try:
    n_Nw_sw_cases = len([fname for fname in os.listdir(r'intermediate_results\\static_wind_'+static_aero_coef_method+r'\\') if 'Nw_dict_' in fname])
except FileNotFoundError:
    n_Nw_sw_cases = 0

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
########################################################################################################################
#                                                AERODYNAMIC ANALYSES
########################################################################################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
cospec_type_cases = [2]  # Best choices: 2, or 5.... 1: L.D.Zhu (however, the Cij params should be fitted again to measurements using this Zhu's formula and not Davenports). 2: Davenport, adapted for a 3D wind field (9 coherence coefficients). 3: Further including 1-pt spectrum Suw as in (Kaimal et al, 1972) and 2-pt spectrum Suw,ij as in (Katsuchi et al, 1999). 4: Fully according to (Hémon and Santi, 2007)(WRONG. No Coh_uw information required which is strange!). 5: Fully according to "A turbulence model based on principal components" (Solari and Tubino, 2002), supported by (Oiseth et al, 2013). 6: same as 2 but now with cosine term that allows negative coh
Ii_simplified = True  # Turbulence intensities. Simplified -> same turbulence intensities everywhere for all directions.
include_modal_coupling = True  # True: CQC. False: SRSS. Off-diag of modal M, C and K in the Freq. Dom. (modal coupling).
include_SE_in_modal = False  # includes effects from Kse when calculating mode shapes (only relevant in Freq. Domain). True gives complex mode shapes!
########################################################################################################################
# Frequency domain buffeting analysis:
# ########################################################################################################################
# # ONE CASE (Can be used to generate new spectra of response for further use in the frequency discretization)
# dtype_in_response_spectra = 'float32'
# include_sw = True
# include_KG = True
# n_aero_coef = 6
# include_SE = True
# make_M_C_freq_dep = False
# aero_coef_method = '2D_fit_cons'
# skew_approach = '3D'
# flutter_derivatives_type = '3D_full'
# n_freq = 2050  # Needs to be (much) larger than the number of frequencies used when 'equal_energy_bins'. E.g. 2050 for 'equal_width_bins', or 256 otherwise
# f_min = 0.002
# f_max = 0.5
# f_array_type = 'equal_width_bins'  # Needs to be 'equal_width_bins' in order to generate the spectra which then enables obtaining 'equal_energy_bins'
# n_modes = 100
# beta_DB = rad(100)
# generate_spectra_for_discretization = True if (f_array_type == 'equal_width_bins' and n_freq >= 1024) else False
# std_delta_local = buffeting_FD_func(include_sw, include_KG, aero_coef_method, n_aero_coef, skew_approach, include_SE, flutter_derivatives_type, n_modes, f_min, f_max, n_freq, g_node_coor, p_node_coor,
#                       Ii_simplified, beta_DB, R_loc, D_loc, cospec_type, include_modal_coupling, include_SE_in_modal, f_array_type, make_M_C_freq_dep, dtype_in_response_spectra, generate_spectra_for_discretization)['std_delta_local']

# MULTIPLE CASES
dtype_in_response_spectra_cases = ['float64']  # complex128, float64, float32. It doesn't make a difference in accuracy, nor in computational time (only when memory is an issue!).
include_sw_cases = [False]  # include static wind effects or not (initial angle of attack and geometric stiffness)
include_KG_cases = [True]  # include the effects of geometric stiffness (both in girder and columns)
n_aero_coef_cases = [6]  # Include 3 coef (Drag, Lift, Moment), 4 (..., Axial) or 6 (..., Moment xx, Moment zz). Only working for the '3D' skew wind approach!!
include_SE_cases = [True]  # include self-excited forces or not. If False, then flutter_derivatives_type must be either '3D_full' or '2D_full'
make_M_C_freq_dep_cases = [False]  # include frequency-dependent added masses and added damping, or instead make an independent approach (using only the dominant frequency of each dof)
aero_coef_method_cases = ['2D_fit_cons']  # method of interpolation & extrapolation. '2D_fit_free', '2D_fit_cons', 'cos_rule', '2D'
skew_approach_cases = ['3D']  # '3D', '2D', '2D+1D', '2D_cos_law'
flutter_derivatives_type_cases = ['3D_full']  # '3D_full', '3D_Scanlan', '3D_Scanlan confirm', '3D_Zhu', '3D_Zhu_bad_P5', '2D_full','2D_in_plane'
n_freq_cases = [128]  # Use 1024 with 'equal_width_bins' or 128 with 'equal_energy_bins'
f_min_cases = [0.002]  # Hz. Use 0.002
f_max_cases = [0.5]  # Hz. Use 0.5! important to not overstretch this parameter
f_array_type_cases = ['equal_energy_bins']  # 'equal_width_bins', 'equal_energy_bins'
# n_modes_cases = [(g_node_num+len(p_node_coor))*6]
n_modes_cases = [100]
n_nodes_cases = [len(g_node_coor)]
# Nw_idxs = np.arange(n_Nw_sw_cases)  # Use: [None] or np.arange(positive integer) (e.g. np.arange(n_Nw_sw_cases)). [None] -> Homogeneous wind only (as in Paper 2). Do not use np.arange(0)
Nw_idxs = [None]  # Use: [None] or np.arange(positive integer) (e.g. np.arange(n_Nw_sw_cases)). [None] -> Homogeneous wind only (as in Paper 2). Do not use np.arange(0)
Nw_or_equiv_Hw_cases = [None]  # Use [Nw] to analyse Nw only. Use ['Nw', 'Hw'] to analyse both Nw and the equivalent Hw!
beta_DB_cases = np.array([rad(280), rad(340)])  # np.arange(rad(100), rad(359), rad(1000))  # wind (from) directions. Interval: [rad(0), rad(360)]

if Nw_idxs != [None]:
    assert len(beta_DB_cases) == 1
assert Nw_idxs == [None] or np.max(Nw_idxs) <= n_Nw_sw_cases, "Decrease the Nw_idxs! The static analysis"
list_of_cases = list_of_cases_FD_func(n_aero_coef_cases, include_SE_cases, aero_coef_method_cases, beta_DB_cases,
                                      flutter_derivatives_type_cases, n_freq_cases, n_modes_cases, n_nodes_cases,
                                      f_min_cases, f_max_cases, include_sw_cases, include_KG_cases, skew_approach_cases,
                                      f_array_type_cases, make_M_C_freq_dep_cases, dtype_in_response_spectra_cases, Nw_idxs,
                                      Nw_or_equiv_Hw_cases, cospec_type_cases)

# import cProfile
# pr = cProfile.Profile()
# pr.enable()
# Writing results
parametric_buffeting_FD_func(list_of_cases, g_node_coor, p_node_coor, Ii_simplified, R_loc, D_loc, include_modal_coupling, include_SE_in_modal)
# pr.disable()
# pr.print_stats(sort='cumtime')

raise Exception  # remove this (used to stop the code from reaching the time-domain calculations)

########################################################################################################################
# Time domain buffeting analysis:
########################################################################################################################
# Input (change the numbers only)
wind_block_T = 600  # (s). Desired duration of each wind block. To be increased due to overlaps.
wind_overlap_T = 8  # (s). Total overlapping duration between adjacent blocks.
# transient_T = 3 * wind_block_T  # (s). Transient time due to initial conditions, to be later discarded in the response analysis.
transient_T = 1 * wind_block_T  # (s). Transient time due to initial conditions, to be later discarded in the response analysis.
ramp_T = 0  # (s). Ramp up time, inside the transient_T, where windspeeds are linearly increased.
# wind_T = 3 * 6 * wind_block_T + transient_T  # (s). Total time-domain simulation duration, including transient time, after overlapping. Keep it in this format (multiple of each wind block time).
wind_T = 4 * wind_block_T + transient_T  # (s). Total time-domain simulation duration, including transient time, after overlapping. Keep it in this format (multiple of each wind block time).

# # ONE CASE
# include_sw = True
# include_KG = True
# n_aero_coef = True
# include_SE = True
# aero_coef_method = 'hybrid'
# flutter_derivatives_type = 'QS non-skew'
# aero_coef_linearity = 'NL'
# beta_DB = rad(100)
# n_seeds = 1
# dt = 4  # s. Time step in the calculation
# std_delta_local = buffeting_TD_func(aero_coef_method, n_aero_coef, include_SE, flutter_derivatives_type, include_sw,
#                                     include_KG, g_node_coor, p_node_coor, Ii_simplified_bool, R_loc, D_loc, n_seeds, dt,
#                                     wind_block_T, wind_overlap_T, wind_T, transient_T, beta_DB, aero_coef_linearity,
#                                     cospec_type, plots=False, save_txt=True)['std_delta_local_mean']

# LIST OF CASES
include_sw_cases = [False]  # include static wind effects or not (initial angle of attack and geometric stiffness)
include_KG_cases = [False]  # include the effects of geometric stiffness (both in girder and columns)
n_aero_coef_cases = [6]  # Include 3 coef (Drag, Lift, Moment), 4 (..., Axial) or 6 (..., Moment xx, Moment zz)
include_SE_cases = [True]  # include self-excited forces or not. If False, then flutter_derivatives_type must be either '3D_full' or '2D_full'
aero_coef_method_cases = ['2D_fit_cons']  # method of interpolation & extrapolation. '2D_fit_free', '2D_fit_cons', 'cos_rule', '2D'
skew_approach_cases = ['3D']  # '3D', '2D', '2D+1D', '2D_cos_law' # todo: not working for aero_coef 'NL'
flutter_derivatives_type_cases = ['3D_full']  # '3D_full', '3D_Scanlan', '3D_Scanlan_confirm', '3D_Zhu', '3D_Zhu_bad_P5'
aero_coef_linearity_cases = ['NL']  # 'L': Taylor formula. 'NL': aero_coeff from instantaneous beta and theta
SE_linearity_cases = ['L']  # 'L': Constant Fb in Newmark, SE (if included!) taken as linear Kse and Cse (KG is not updated) 'NL': Fb is updated each time step, no Kse nor Cse (KG is updated each dt).
geometric_linearity_cases = ['L']  # 'L': Constant M,K in Newmark. 'NL': M,K are updated each time step from deformed node coordinates.
n_nodes_cases = [len(g_node_coor)]
n_seeds_cases = [2]
dt_cases = [4]  # Not all values possible! wind_overlap_size must be even!
beta_DB_cases = np.arange(rad(0), rad(359), rad(1000))  # wind (from) directions. Interval: [rad(0), rad(360)]
list_of_cases = list_of_cases_TD_func(aero_coef_method_cases, n_aero_coef_cases, include_SE_cases, flutter_derivatives_type_cases, n_nodes_cases, include_sw_cases, include_KG_cases, n_seeds_cases,
                                      dt_cases, aero_coef_linearity_cases, SE_linearity_cases, geometric_linearity_cases, skew_approach_cases, beta_DB_cases)

# Writing results
parametric_buffeting_TD_func(list_of_cases, g_node_coor, p_node_coor, Ii_simplified, wind_block_T, wind_overlap_T,
                      wind_T, transient_T, ramp_T, R_loc, D_loc, cospec_type, plots=False, save_txt=False)

# # Plotting
# import buffeting_plots
# buffeting_plots.response_polar_plots(symmetry_180_shifts=False, error_bars=True, closing_polygon=True, tables_of_differences=False, shaded_sector=True, show_bridge=True, order_by=['skew_approach', 'Analysis', 'g_node_num', 'n_freq', 'SWind', 'KG',  'Method', 'SE', 'FD_type', 'C_Ci_linearity', 'f_array_type', 'make_M_C_freq_dep', 'dtype_in_response_spectra', 'beta_DB'])
# # # # # buffeting_plots.response_polar_plots(symmetry_180_shifts=False, error_bars=True, closing_polygon=True, tables_of_differences=False, shaded_sector=True, show_bridge=True, order_by=['skew_approach', 'Analysis', 'g_node_num', 'n_freq', 'SWind', 'KG',  'Method', 'SE', 'FD_type', 'n_aero_coef', 'beta_DB'])


# todo: to accelerate the code, calculate the polynomial coefficients of the constrained fits only once and save them in a separate file to be accessed for each mean wind direction.
# #######################################################################################################################
# Validating the wind field:
# #######################################################################################################################
from wind_field.wind_field_3D_applied_validation import wind_field_3D_applied_validation_func
from simple_5km_bridge_geometry import arc_length, R
from buffeting import wind_field_3D_all_blocks_func, rad, deg

beta_DB = rad(100)  # wind direction

# Input (change the numbers only)
cospec_type = 2
wind_block_T = 600  # (s). Desired duration of each wind block. To be increased due to overlaps.
wind_overlap_T = 8  # (s). Total overlapping duration between adjacent blocks.
transient_T = 0 * wind_block_T  # (s). Transient time due to initial conditions, to be later discarded in the response analysis.
ramp_T = 0  # (s). Ramp up time, inside the transient_T, where windspeeds are linearly increased.
wind_T = 18 * wind_block_T + transient_T  # (s). Total time-domain simulation duration, including transient time, after overlapping. Keep it in this format (multiple of each wind block time).
dt = 0.25  # s. Time step in the calculation

# Wind speeds at each node, in Gw coordinates (XuYvZw).
Ii_simplified_bool = True
windspeed = wind_field_3D_all_blocks_func(g_node_coor, beta_DB, dt, wind_block_T, wind_overlap_T, wind_T, ramp_T,
                                          cospec_type, Ii_simplified_bool, plots=False)

# Validation
freq_array = np.arange(1 / wind_T, (1/dt) / 2, 1 / wind_T)
f_min = np.min(freq_array)
f_max = np.max(freq_array)
n_freq = len(freq_array)
# n_freq = 128
# f_min = 0.002
# f_max = 0.5
n_nodes_validated = 10  # total number of nodes to assess wind speeds: STD, mean, co-spectra, correlation
node_test_S_a = 0  # node tested for auto-spectrum
n_nodes_val_coh = 5  # num nodes tested for assemblage of 2D correlation decay plots

wind_field_3D_applied_validation_func(g_node_coor, windspeed, dt, wind_block_T, beta_DB, arc_length, R, Ii_simplified_bool, f_min, f_max,
                                      n_freq,  n_nodes_validated, node_test_S_a, n_nodes_val_coh)

print('all is done')


