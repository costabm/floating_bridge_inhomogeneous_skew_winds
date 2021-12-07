import json
import numpy as np
from simple_5km_bridge_geometry import g_node_coor, p_node_coor, g_s_3D_func
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import copy
import pandas as pd
import os


def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

# SENSITIVITY ANALYSIS
# Plotting response
def response_polar_plots(symmetry_180_shifts=False, error_bars=True, closing_polygon=False, tables_of_differences=False, shaded_sector=True, show_bridge=True, order_by=['skew_approach', 'Analysis', 'g_node_num', 'n_freq', 'SWind', 'KG',  'Method', 'SE', 'FD_type', 'C_Ci_linearity', 'beta_DB']):
    ####################################################################################################################
    # ORGANIZING DATA
    ####################################################################################################################
    # Getting the paths of the results tables
    results_paths_FD = []
    results_paths_TD = []
    all_files_paths = []
    my_path = os.path.join(os.getcwd(), r'results')

    for item in os.listdir(my_path):
        all_files_paths.append(item)

    for path in all_files_paths:
        if path[:16] == "FD_std_delta_max":
            results_paths_FD.append(path)
        if path[:16] == "TD_std_delta_max":
            results_paths_TD.append(path)

    # Getting the DataFrames of the results. Adding column for Analysis type ('TD' or 'FD').
    results_df_list = []
    n_results_FD = len(results_paths_FD)
    n_results_TD = len(results_paths_TD)
    for r in range(n_results_FD):
        df = pd.read_csv(os.path.join(my_path, results_paths_FD[r]))
        df['Analysis'] = 'FD'
        results_df_list.append(df)
    for r in range(n_results_TD):
        df = pd.read_csv(os.path.join(my_path, results_paths_TD[r]))
        df['Analysis'] = 'TD'
        results_df_list.append(df)

    # Merging DataFrames, changing NaNs to string 'NA'
    results_df = pd.concat(results_df_list, ignore_index=True).rename(columns={'Unnamed: 0': 'old_indexes'})
    results_df = results_df.fillna('NA')  # replace nan with 'NA', so that 'NA' == 'NA' when counting betas

    # Re-ordering! Change here: Order first the parameters being studied, and include 'beta_DB' in the end.
    results_df = results_df.sort_values(by=order_by).reset_index(drop=True)

    # DataFrames without results and betas, to find repeated parameters and count number of betas in each case
    if n_results_TD > 0:  # if we have TD results
        list_of_cases_df_repeated = results_df.drop(
            ['old_indexes', 'beta_DB', 'std_max_dof_0', 'std_max_dof_1', 'std_max_dof_2', 'std_max_dof_3',
             'std_max_dof_4', 'std_max_dof_5', 'std_std_max_dof_0', 'std_std_max_dof_1', 'std_std_max_do    f_2',
             'std_std_max_dof_3', 'std_std_max_dof_4', 'std_std_max_dof_5'], axis=1)  # removing columns
    else:
        list_of_cases_df_repeated = results_df.drop(
            ['old_indexes', 'beta_DB', 'std_max_dof_0', 'std_max_dof_1', 'std_max_dof_2', 'std_max_dof_3',
             'std_max_dof_4', 'std_max_dof_5'], axis=1)  # removing columns
    # Counting betas of each case
    count_betas = pd.DataFrame({'beta_DB_count': list_of_cases_df_repeated.groupby(
        list_of_cases_df_repeated.columns.tolist()).size()}).reset_index()
    list_of_cases_df = list_of_cases_df_repeated.drop_duplicates().reset_index().rename(
        columns={'index': '1st_result_index'})

    list_of_headers = list(list_of_cases_df.columns.values)

    # list with headers:
    # 'Method', 'n_aero_coef', 'SE', 'FD_type', 'n_modes', 'n_freq',
    # 'g_node_num', 'f_min', 'f_max', 'SWind', 'KG',
    # 'cospec_type',
    # 'damping_ratio', 'damping_Ti', 'damping_Tj',
    # 'Analysis',
    # 'N_seeds', 'dt', 'C_Ci_linearity', 'SE_linearity', 'geometric_linearity'

    list_of_cases_df = pd.merge(list_of_cases_df, count_betas, on=list_of_headers[1:])  # SENSITIVITY ANALYSIS. First header "1st_result_index" not used.

    pd.set_option("display.max_rows", None, "display.max_columns", None, 'expand_frame_repr', False)
    print('Cases available to plot: \n', list_of_cases_df)
    idx_cases_to_plot = eval(input('''Enter list of index numbers from '1st_result_index' to plot:'''))  # choose the index numbers to plot in 1 plot, from '1st result_index' column
    list_of_cases_to_plot_df = list_of_cases_df.loc[list_of_cases_df['1st_result_index'].isin(idx_cases_to_plot)]
    list_of_cases_to_plot_df = list_of_cases_to_plot_df.assign(Method=list_of_cases_to_plot_df['Method'].replace(['2D_fit_free', '2D_fit_cons', '2D_fit_cons_2', 'cos_rule', '2D'], ['Free fit','Constrained fit', '2-var. constr. fit (2)', 'Cosine rule', '2D']))
    list_of_cases_to_plot_df = list_of_cases_to_plot_df.assign(FD_type=list_of_cases_to_plot_df['FD_type'].replace(['3D_Zhu', '3D_Scanlan','3D_full'], ['(QS) Zhu', '(QS) Scanlan', '(QS) full']))
    ####################################################################################################################
    # PLOTTING
    ####################################################################################################################
    from cycler import cycler
    angle_idx = list(list_of_cases_to_plot_df['1st_result_index'])
    angle_idx = [list(range(angle_idx[i], angle_idx[i]+list_of_cases_to_plot_df['beta_DB_count'].iloc[i])) for i in range(len(angle_idx))]

    str_dof = ["Max. $\sigma_x$ $[m]$",
               "Max. $\sigma_y$ $[m]$",
               "Max. $\sigma_z$ $[m]$",
               "Max. $\sigma_{rx}$ $[\degree]$",
               "Max. $\sigma_{ry}$ $[\degree]$",
               "Max. $\sigma_{rz}$ $[\degree]$"]
    str_dof2 = ['x','y','z','rx','ry','rz']

    new_colors = [plt.get_cmap('jet')(1. * i / len(idx_cases_to_plot)) for i in range(len(idx_cases_to_plot))]
    # # 3 colors for FD
    # custom_cycler = (cycler(color=['orange', 'blue', 'green', 'cyan', 'cyan', 'cyan', 'gold', 'gold', 'gold', 'red', 'red', 'red']) +
    #                  #cycler(color=new_colors) +
    #                  cycler(linestyle=['-', '-', ':', '-.', '--', '--', '-', '--', '--', '-', '--', '--']) +
    #                  cycler(lw=[3.0, 2.5, 2.55, 1.2, .8, .8, 1.8, .8, .8, 1.8, .8, .8]) +
    #                  cycler(alpha=[0.6, 0.5, 0.7, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.7, 0.5, 0.5]))
    # 1 TD case with +- STD of STD
    # custom_cycler = (cycler(color=['orange', 'darkorange', 'darkorange', 'darkorange']) +
    #                  #cycler(color=new_colors) +
    #                  cycler(linestyle=['-', '-.', '--', '--']) +
    #                  cycler(lw=[4.0, 2.0, 1.5, 1.5]) +
    #                  cycler(alpha=[0.6, 0.9, 0.9, 0.9]))
    # FD vs 1 TD case with +- STD of STD
    # lineweight_list = [3.0, 2.8, 1.5, 2., 2.]
    # custom_cycler = (cycler(color=['brown', 'deepskyblue', 'gold', 'darkorange', 'darkorange']) +
    #                  #cycler(color=new_colors) +
    #                  cycler(linestyle=['-', '-', '-', ':', '-.']) +
    #                  cycler(lw=lineweight_list) +
    #                  cycler(alpha=[0.8, 0.4, 0.8, 0.8, 0.8]))
    # # Sensitivity
    # custom_cycler = (cycler(color=['cyan', 'orange', 'red', 'blue', 'magenta', 'blue', 'red']) +
    #                  # cycler(color=new_colors) +
    #                  cycler(linestyle=['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot']) +
    #                  cycler(lw=[1, 1, 1., 1, 1, 1., 1]) +
    #                  cycler(alpha=[0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]))
    # FD aero method:
    # lineweight_list = [2., 2., 2., 2.]
    # custom_cycler = (cycler(color=['gold', 'brown', 'green', 'blue', ]) +
    #                  #cycler(color=new_colors) +
    #                  cycler(linestyle=['--', '-', '-.', (0, (3, 1.5, 1, 1.5, 1, 1.5))]) +
    #                  cycler(lw=lineweight_list) +
    #                  cycler(marker=["o"]*len(lineweight_list)) +
    #                  cycler(markersize=np.array(lineweight_list)*1.2) +
    #                  cycler(alpha=[0.8, 0.8, 0.8, 0.4]))
    # FD approach:
    # lineweight_list = [2., 2., 2., 2.]
    # custom_cycler = (cycler(color=['gold','green','brown', 'blue', ]) +
    #                  #cycler(color=new_colors) +
    #                  cycler(linestyle=['--','-.', '-', (0, (3, 1.5, 1, 1.5, 1, 1.5))]) +
    #                  cycler(lw=lineweight_list) +
    #                  cycler(marker=["o"]*len(lineweight_list)) +
    #                  cycler(markersize=np.array(lineweight_list)*1.2) +
    #                  cycler(alpha=[0.8, 0.8, 0.8, 0.4]))
    # Only one FD case:
    # lineweight_list = [2., 2., 2., 2.]
    # custom_cycler = (cycler(color=['brown', 'green', 'gold', 'blue', ]) +
    #                  #cycler(color=new_colors) +
    #                  cycler(linestyle=['-', '-.', '--', (0, (3, 1.5, 1, 1.5, 1, 1.5))]) +
    #                  cycler(lw=lineweight_list) +
    #                  cycler(marker=["o"]*len(lineweight_list)) +
    #                  cycler(markersize=np.array(lineweight_list)*1.2) +
    #                  cycler(alpha=[0.8, 0.8, 0.8, 0.4]))
    # SE forces:
    lineweight_list = [2.0, 3.5, 2.5, 1.5, 2.]
    custom_cycler = (cycler(color=['deepskyblue', 'gold', 'green', 'brown', 'darkorange']) +
                     #cycler(color=new_colors) +
                     cycler(linestyle=['-', '-', '--', ':', '-.']) +
                     cycler(lw=lineweight_list) +
                     cycler(marker=["o"]*len(lineweight_list)) +
                     cycler(markersize=np.array(lineweight_list)*1.2) +
                     cycler(alpha=[0.4, 0.8, 0.8, 0.8, 0.8]))

    for dof in [0,1,2,3,4,5]:
        plt.figure(figsize=(3.7, 3.7), dpi=600)
        ax = plt.subplot(111, projection='polar')
        ax.set_prop_cycle(custom_cycler)
        k = -1 # counter
        # for _, (_, aero_coef_method, n_aero_coef, include_SE, flutter_derivatives_type, n_modes, n_freq, g_node_num, f_min, f_max, include_sw, include_KG, skew_approach,
        #         f_array_type, make_M_C_freq_dep, dtype_in_response_spectra, cospec_type,
        #         damping_ratio, damping_Ti, damping_Tj, analysis_type, n_seeds, dt, C_Ci_linearity, SE_linearity, geometric_linearity, _) in list_of_cases_to_plot_df.iterrows():
        for _, (_, aero_coef_method, n_aero_coef, include_SE, flutter_derivatives_type, n_modes, n_freq, g_node_num, f_min, f_max, include_sw, include_KG, skew_approach, f_array_type,
                make_M_C_freq_dep, dtype_in_response_spectra, cospec_type, damping_ratio, damping_Ti, damping_Tj, analysis_type, _) in list_of_cases_to_plot_df.iterrows():
            k += 1  # starts with 0
            # str_plt_0 = aero_coef_method[:6] + '. '
            # str_plt_1 = 'Ca: ' + str(n_aero_coef)[:1] + '. '
            # str_plt_2 = 'w/ SE. ' if include_SE else 'w/o SE. '
            # str_plt_3 = flutter_derivatives_type + '.' if include_SE else ''
            # str_plt = str_plt_0 + str_plt_1 + str_plt_2 + str_plt_3
            if analysis_type == 'FD':
                # str_plt = 'Frequency-domain'
                # str_plt = str(skew_approach) + '.  MD: ' + str(include_SE)[0]
                # str_plt = str(aero_coef_method)
                str_plt = str(skew_approach)
                str_plt = str(skew_approach) + '. ' + str(n_aero_coef)
                if not include_SE:
                    str_plt = 'No self-excited forces'
                if include_SE:
                    str_plt = str(flutter_derivatives_type)
                    # str_plt = str(aero_coef_method)
                # str_plt = r'Frequency-domain'
                # str_plt = str(int(n_freq))
                # str_plt = str(int(g_node_num))
            if analysis_type == 'TD':
                str_plt = r'Time-domain: $\mu$ ('+ u"\u00B1" +' $\sigma$)'
                # if C_Ci_linearity == 'L':
                #     str_plt = 'Linear time-domain'
                # elif C_Ci_linearity == 'NL':
                #     str_plt = 'Non-linear time-domain'
            angle = np.array(results_df['beta_DB'][angle_idx[k][0]:angle_idx[k][-1]+1])
            radius = np.array(results_df['std_max_dof_' + str(dof)][angle_idx[k][0]:angle_idx[k][-1]+1])
            if n_results_TD > 0 and error_bars and analysis_type=='TD': # Include error_bars
                radius_std = np.array(results_df['std_std_max_dof_' + str(dof)][angle_idx[k][0]:angle_idx[k][-1]+1])
            if dof >= 3:
                import copy
                radius = deg(copy.deepcopy(radius))  # converting from radians to degrees!
                if n_results_TD > 0 and error_bars and analysis_type == 'TD':
                    radius_std = deg(copy.deepcopy(radius_std))  # converting from radians to degrees!
            if symmetry_180_shifts:
                angle = np.append(angle, angle+np.pi)
                radius = np.append(radius, radius)
                if n_results_TD > 0 and error_bars and analysis_type=='TD':
                    radius_std = np.append(radius_std, radius_std)  # for error bars
            if closing_polygon:
                angle = np.append(angle, angle[0])  # Closing the polygon, adding same value to end:
                radius = np.append(radius, radius[0])  # Closing the polygon, adding same value to end:
            if not (n_results_TD > 0 and analysis_type=='TD' and error_bars):  # plot only if FD or if in TD no error bars are desired
                ax.plot(angle, radius, label=str_plt) #, zorder=0.7)
            else:
                if closing_polygon:
                    radius_std = np.append(radius_std, radius_std[0])  # Closing the polygon, adding same value to end:
                ax.errorbar(angle, radius, yerr=radius_std, label=str_plt) #, zorder=0.7)
                # ax.plot(angle, radius - radius_std, label=r'Time-domain: $\mu$ '+u"\u00B1"+' $\sigma$') #, zorder=0.7)
                # ax.plot(angle, radius + radius_std) #, zorder=0.7)
        if shaded_sector:
            ylim = ax.get_ylim()
            # Shade interpolation area:
            ax.fill_between(np.linspace(rad(100-30), rad(100+30), 100), ylim[0], ylim[1], color='lime', alpha=0.1, edgecolor='None') #, zorder=0.8)
            ax.fill_between(np.linspace(rad(280-30), rad(280+30), 100), ylim[0], ylim[1], color='lime', alpha=0.1, edgecolor='None', label='Domain of available ' + r'$C_{i}$' + ' data') #, zorder=0.8)
            # Shade extrapolation area:
            ax.fill_between(np.linspace(rad(100+30), rad(280-30), 100), ylim[0], ylim[1], color='grey', alpha=0.1, edgecolor='None') #, zorder=0.8)
            ax.fill_between(np.linspace(rad(280+30), rad(360)   , 100), ylim[0], ylim[1], color='grey', alpha=0.1, edgecolor='None') #, zorder=0.8)
            ax.fill_between(np.linspace(rad(0)     , rad(100-30), 100), ylim[0], ylim[1], color='grey', alpha=0.1, edgecolor='None', label='Domain of ' + r'$C_{i}$' + ' extrapolation') #, zorder=0.8)
        if show_bridge:
            from simple_5km_bridge_geometry import g_node_coor
            # rotate from Gs system to Gmagnetic system, where x-axis is aligned with S-N direction:
            rot_angle = rad(10)
            g_node_coor_Gmagnetic = np.einsum('ij,nj->ni', np.transpose(np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                                                                                  [np.sin(rot_angle),  np.cos(rot_angle), 0],
                                                                                  [                0,                  0, 1]])), g_node_coor)
            ylim = ax.get_ylim()[1] * 0.9
            half_chord = np.max(abs(g_node_coor[:,0])) / 2
            sagitta = np.max(abs(g_node_coor[:, 1]))
            # Translating grom Gmagnetic to Gcompasscenter, for the coor. sys. to be at the center of the compass
            g_node_coor_Gcompasscenter = np.array([g_node_coor_Gmagnetic[:,0]-half_chord*np.cos(rot_angle)  + sagitta*np.sin(rot_angle),  # the sagitta terms can alternatively be removed
                                                   g_node_coor_Gmagnetic[:,1]+half_chord*np.sin(rot_angle)  + sagitta*np.cos(rot_angle),  # the sagitta terms can alternatively be removed
                                                   g_node_coor_Gmagnetic[:,2]]).transpose()
            bridge_node_radius = np.sqrt(g_node_coor_Gcompasscenter[:,0]**2 + g_node_coor_Gcompasscenter[:,1]**2)
            bridge_node_radius_norm = bridge_node_radius / np.max(abs(g_node_coor_Gcompasscenter)) * ylim
            bridge_node_angle = np.arctan2(-g_node_coor_Gcompasscenter[:,1],g_node_coor_Gcompasscenter[:,0])
            ax.plot(bridge_node_angle, bridge_node_radius_norm, linestyle='-', linewidth=3., alpha=0.5, color='black', marker="None", label='Bridge axis')#,zorder=k+1)

        ax.grid(True)
        # ax.legend(bbox_to_anchor=(1.63,0.93), loc="upper right", title='Analysis:')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(str_dof[dof], va='bottom')
        # ax.set_title(str_dof[dof]+'\n (Fitting: ' + str(aero_coef_method)+')', va='bottom')
        # ax.set_title('\n Sensitivity: No. frequency bins\n'+str_dof[dof][:-2], va='bottom')
        # ax.set_title('\n Sensitivity: No. girder nodes\n'+str_dof[dof][:-2], va='bottom')

        plt.tight_layout()

        plt.savefig(r'results\Polar_std_delta_local_' + str_dof2[dof] +'_Spec-' + str(cospec_type) + \
                    '_zeta-' + str(damping_ratio) + '_Ti-' + str(damping_Ti) + '_Tj-' + str(damping_Tj) + '_Nodes-' + \
                    str(g_node_num) + '_Modes-' + str(n_modes) + '_FD-f-' + str(f_min) + '-' + str(f_max) + \
                    '-' + str(n_freq) + '_NAeroCoef-' + str(n_aero_coef) + '.png')

        handles, labels = plt.gca().get_legend_handles_labels()
        # order = [0, 3, 1, 4, 2, 5]
        # FD skew approach:
        order = list(range(len(handles)))
        # FD aero fit method:
        # order = [3,0,2,1,4,5,6]
        # order = list(range(len(handles)))
        # SE:
        order = list(range(len(handles)))
        # FD vs 1 TD:
        # order = [0, 4, 1, 2, 3]  # with shaded sectors
        # order = [0, 2, 3, 1]  # without shaded sectors
        # # N aero coef:
        # labels[0] = '(3D) ' + r'$[0,C_y,C_z,C_{rx},0,0]$'
        # labels[1] = '(3D) ' + r'$[C_x,C_y,C_z,C_{rx},0,0]$'
        # labels[2] = '(3D) ' + r'$[C_x,C_y,C_z,C_{rx},C_{ry},C_{rz}]$'
        # aero Method:
        # labels = [l.replace('Cosine rule', '(3D) Univar. fit + Cosine rule') for l in labels]
        # labels = [l.replace('2D', '(3D) Univar. fit + 2D approach') for l in labels]
        # labels = [l.replace('Free fit', '(3D) Free bivariate fit') for l in labels]
        # labels = [l.replace('Constrained fit', '(3D) Constrained bivariate fit') for l in labels]

        legend = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(10.63,0.93)) #, ncol=1)

        def export_legend(legend, filename=r'results\legend.png', expand=[-5, -5, 5, 5]):
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)
        export_legend(legend)
        plt.close()

    if tables_of_differences:
        # Table of differences
        num_cases = len(list_of_cases_df)
        num_betas = list_of_cases_df.iloc[0]['beta_DB_count']
        table_max_diff_all_betas = pd.DataFrame()
        table_all_diff = pd.DataFrame(np.zeros((num_cases*num_betas, num_cases*6))*np.NaN)
        table_all_diff.columns = ['std_max_dof_0', 'std_max_dof_1', 'std_max_dof_2', 'std_max_dof_3', 'std_max_dof_4', 'std_max_dof_5']*num_cases
        table_all_diff['Case'] = 'N/A'  # adding a new column at the end.
        table_all_diff.loc[-1] = 'N/A'  # apparently this adds a new row to the end..... . ... ..
        assert all([num_betas == list_of_cases_df.iloc[i]['beta_DB_count'] for i in range(len(list_of_cases_df))])  # all cases have same number of betas
        betas_deg_df = np.round(results_df['beta_DB'].iloc[0:num_betas] * 180 / np.pi)
        for i in range(num_cases):
            str_case_i = list_of_cases_df.iloc[i]['skew_approach'] + '. MD: ' + str(list_of_cases_df.iloc[i]['SE'])[0]
            if list_of_cases_df.iloc[i]['SE']:
                str_case_i += '. ' + list_of_cases_df.iloc[i]['FD_type']
            for j in range(num_cases):
                if i >= j:
                    str_case_j = list_of_cases_df.iloc[j]['skew_approach'] + '. MD: ' + str(list_of_cases_df.iloc[j]['SE'])[0]
                    if list_of_cases_df.iloc[j]['SE']:
                        str_case_j += '. ' + list_of_cases_df.iloc[j]['FD_type']
                    results_case_i = results_df.iloc[i * num_betas: i * num_betas + num_betas][['std_max_dof_0', 'std_max_dof_1', 'std_max_dof_2', 'std_max_dof_3', 'std_max_dof_4', 'std_max_dof_5']].reset_index(drop=True)
                    results_case_j = results_df.iloc[j * num_betas: j * num_betas + num_betas][['std_max_dof_0', 'std_max_dof_1', 'std_max_dof_2', 'std_max_dof_3', 'std_max_dof_4', 'std_max_dof_5']].reset_index(drop=True)
                    results_diff_ij = (results_case_i - results_case_j)/results_case_j * 100
                    results_diff_ij_arr = np.array(results_diff_ij)
                    table_all_diff.iloc[i * num_betas: i * num_betas + num_betas, j*6:j*6+6] = results_diff_ij_arr
                    table_all_diff.iloc[i * num_betas: i * num_betas + num_betas,-1] = [str_case_i] * num_betas  # naming the cases in the last column
                    table_all_diff.iloc[-1, j*6:j*6+6] = [str_case_j]*6   # naming the cases in the last row
                    # Table of maximum differences
                    if i > j:
                        if not (('3D' in str_case_i and '2D' in str_case_j) or ('2D' in str_case_i and '3D' in str_case_j) or ('MD: T' in str_case_i and 'MD: F' in str_case_j) or ('MD: F' in str_case_i and 'MD: T' in str_case_j)):  # put your conditions here to compare what you want
                            max_diff_all_betas_ij = pd.DataFrame(abs(results_diff_ij).max(axis=0))
                            max_diff_all_betas_ij.columns = [ '(' + str_case_i + ')   VS   (' + str_case_j + ')']
                            table_max_diff_all_betas = pd.concat([table_max_diff_all_betas, max_diff_all_betas_ij], axis=1)
        table_all_diff.insert(0,"beta[deg]", betas_deg_df.tolist()*num_cases + ['Cases:'])
        table_max_diff_all_betas = table_max_diff_all_betas.T
        from time import gmtime, strftime
        table_all_diff.to_csv(r'results\Table_of_all_differences_between_all_cases_' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '.csv',index = False)
        table_max_diff_all_betas.to_csv(r'results\Table_of_the_maximum_difference_for_pairs_of_cases_' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '.csv')

# response_polar_plots(symmetry_180_shifts=False, error_bars=True, closing_polygon=True, tables_of_differences=False, shaded_sector=True, show_bridge=True, order_by=['skew_approach', 'Analysis', 'g_node_num', 'n_freq', 'SWind', 'KG',  'Method', 'SE', 'FD_type', 'C_Ci_linearity', 'f_array_type', 'make_M_C_freq_dep', 'dtype_in_response_spectra', 'beta_DB'])
# response_polar_plots(symmetry_180_shifts=False, error_bars=True, closing_polygon=True, tables_of_differences=False, shaded_sector=True, show_bridge=True, order_by=['skew_approach', 'Analysis', 'g_node_num', 'n_freq', 'SWind', 'KG',  'Method', 'SE', 'FD_type', 'n_aero_coef', 'make_M_C_freq_dep', 'beta_DB'])

def plot_contourf_spectral_response(f_array, S_delta_local, g_node_coor, S_by_freq_unit='rad', zlims_bool=False, cbar_extend='min', filename='Contour_', idx_plot=[1,2,3]):
    """
    S_delta_local: shape(n_freq, n_nodes, n_dof)
    """
    g_s_3D = g_s_3D_func(g_node_coor)
    x = np.round(g_s_3D)
    y = f_array
    x, y = np.meshgrid(x, y)
    for i in idx_plot:
        idx_str = ['x','y','z','rx','ry','rz'][i]
        idx_str_2 = ['$[m^2/Hz]$','$[m^2/Hz]$','$[m^2/Hz]$','$[\degree^2/Hz]$','$[\degree^2/Hz]$','$[\degree^2/Hz]$'][i]
        plt.figure(figsize=(4, 3), dpi=400)
        plt.title(r'$S_{\Delta_{'+idx_str+'}}$'+' '+idx_str_2)
        cmap = plt.get_cmap(matplotlib.cm.viridis)
        vmin = [10**-3,10**-2,10**-3,10**-3,10**-3,10**-3][i]
        factor = 2*np.pi if S_by_freq_unit == 'rad' else 1
        z = np.real(S_delta_local[:, :, i]) * factor  # This is to change the spectral units from m^2/rad to m^2/Hz
        if i >= 3:
            z = copy.deepcopy(z) / np.pi**2 * 180**2  # This is to change the spectral units from rad^2/Hz to degree^2/Hz
        vmax = np.max(z)
        if zlims_bool:
            zlims = [None, [1E-2, 8.9E3], [1E-3,4.4E-1], [1E-3, 3.7E-1], [None], [None]]
            vmax = zlims[i][1]
        levels_base_outer = np.power(10, np.arange(np.floor(np.log10(vmin)), np.ceil(np.log10(vmax))+1))
        levels_base_inner = np.power(10, np.arange(np.ceil(np.log10(vmin)), np.floor(np.log10(vmax))+1))
        levels = np.logspace(np.log10(levels_base_outer[0]), np.log10(vmax), num=200)
        plt.contourf(x, y, z, cmap=cmap, extend=cbar_extend, levels=levels, norm=colors.LogNorm(min(levels),vmax))
        plt.ylabel('Frequency [Hz]')
        plt.yscale('log')
        plt.ylim([0.002, 0.5])
        plt.xlabel('Along arc length [m]')
        plt.xticks([0,2500,5000])

        plt.colorbar(ticks=levels_base_inner)
        plt.tight_layout()
        plt.savefig(r'results\\' + str(filename) + idx_str + ".png")
        plt.close()

def plot_contourf_time_history_response(u_loc, time_array, g_node_coor, filename='Contour_TimeHistory_', idx_plot=[1,2,3]):
    """
    S_delta_local: shape(n_freq, n_nodes, n_dof)
    """
    import matplotlib
    from simple_5km_bridge_geometry import g_s_3D_func
    import copy
    g_s_3D = g_s_3D_func(g_node_coor)
    dt = time_array[1] - time_array[0]
    g_node_num = len(g_s_3D)
    x_base = np.round(g_s_3D)
    for i in idx_plot:
        idx_str = ['x','y','z','rx','ry','rz'][i]
        idx_str_2 = ['$[m]$','$[m]$','$[m]$','$[\degree]$','$[\degree]$','$[\degree]$'][i]
        plt.figure(figsize=(4, 3), dpi=400)
        plt.title(r'$\Delta_{'+idx_str+'}$'+' '+idx_str_2)
        cmap = plt.get_cmap(matplotlib.cm.seismic)
        time_idx_max_response = int(np.where(u_loc[i,:,:g_node_num]==np.max(u_loc[i,:,:g_node_num]))[0])
        dof_eigen_period_of_interest = [100, 100, 6, 6, 6, 6][i]
        time_window_idxs = [time_idx_max_response - int(dof_eigen_period_of_interest*10/dt), time_idx_max_response + int(dof_eigen_period_of_interest*10/dt)]
        z = u_loc[i,time_window_idxs[0]:time_window_idxs[1],:g_node_num]  # excluding pontoon nodes
        y = time_array[time_window_idxs[0]:time_window_idxs[1]]
        x, y = np.meshgrid(x_base, y)
        if i >= 3:
            z = copy.deepcopy(z) / np.pi * 180  # This is to change the spectral units from rad^2/Hz to degree^2/Hz
        vabsmax = np.max(np.abs(z))
        levels = np.linspace(-vabsmax, vabsmax, num=201)
        plt.contourf(x, y, z, cmap=cmap, levels=levels)
        plt.ylabel('Time [s]')
        plt.xlabel('Along arc length [m]')
        plt.xticks([0,2500,5000])
        vabsmax_round = np.round(vabsmax,1)
        # ticks = np.round(np.linspace(-vabsmax_round, vabsmax_round, num=10),1)
        # plt.colorbar(ticks = ticks)
        plt.colorbar()
        ax = plt.gca()
        ax.spines['bottom'].set_linestyle((0, (5, 5)))
        ax.spines['top'].set_linestyle((0, (5, 5)))
        plt.tight_layout()
        plt.savefig(r'results\\' + str(filename) + idx_str + ".png")
        plt.close()
    pass

def time_domain_plots():
    from simple_5km_bridge_geometry import g_node_coor
    my_path = os.path.join(os.getcwd(), r'results')
    u_loc_path = []
    TD_df_path = []  # to obtain the time step dt
    for item in os.listdir(my_path):
        if item[:5] == 'u_loc':
            u_loc_path.append(item)
        elif item[:6] == 'TD_std':
            TD_df_path.append(item)
    idx_u_loc = -1
    idx_TD_std = -1
    TD_df = pd.read_csv(my_path + r"\\" + TD_df_path[idx_TD_std])
    dt = float(TD_df['dt'])  # obtaining the time step dt
    g_node_num = int(TD_df['g_node_num'])  # obtaining the time step dt
    u_loc = np.loadtxt(my_path + r"\\" + u_loc_path[idx_u_loc])
    u_loc = np.array([u_loc[:, 0::6], u_loc[:, 1::6], u_loc[:, 2::6], u_loc[:, 3::6], u_loc[:, 4::6], u_loc[:, 5::6]])  # convert to shape(n_dof, time, n_nodes)
    time_array = np.arange(0, len(u_loc[0])) * dt

    # Plotting time history for y
    plot_contourf_time_history_response(u_loc, time_array, g_node_coor, filename='Contour_TimeHistory_', idx_plot=[1, 2, 3])

    from scipy import signal
    Pxx_den = []
    fs = 1/dt
    nperseg = 6000
    f = signal.welch(u_loc[0,:, 0], fs=fs, nperseg=nperseg)[0]
    # f = signal.periodogram(u_loc[0,:, 0], fs=fs)[0]
    for d in range(6):
        Pxx_den_1_dof_all_nodes = []
        for n in range(g_node_num):
            Pxx_den_1_dof_all_nodes.append(signal.welch(u_loc[d,:, n], fs=fs, nperseg=nperseg)[1])
            # Pxx_den_1_dof_all_nodes.append(signal.periodogram(u_loc[d, :, n], fs=fs)[1])
        Pxx_den.append(Pxx_den_1_dof_all_nodes)
    Pxx_den = np.moveaxis(np.moveaxis(np.array(Pxx_den), 0, -1), 0,1)  # convert to shape(n_freq, n_nodes, n_dof)
    plot_contourf_spectral_response(f_array=f, S_delta_local=Pxx_den, g_node_coor=g_node_coor, S_by_freq_unit='Hz', zlims_bool=True, cbar_extend='both', filename='Contour_TD_', idx_plot=[1,2,3])
# time_domain_plots()


def Nw_sw_plot():
    """Inhomogeneous static wind plots"""

    n_g_nodes = len(g_node_coor)
    n_p_nodes = len(p_node_coor)
    g_s_3D = g_s_3D_func(g_node_coor)
    x = np.round(g_s_3D)
    # Getting the Nw wind properties into the same df
    my_Nw_path = os.path.join(os.getcwd(), r'intermediate_results', 'static_wind')
    n_Nw_sw_cases = len(os.listdir(my_Nw_path))
    Nw_dict_all, Nw_D_loc, Hw_D_loc, Nw_U_bar_RMS, Hw_U_bar_RMS = [], [], [], [], []  # RMS = Root Mean Square, such that the U_bar averages along the fjord are energy-equivalent
    for i in range(n_Nw_sw_cases):
        Nw_path = os.path.join(my_Nw_path, f'Nw_dict_{i}.json')
        with open(Nw_path, 'r') as f:
            Nw_dict_all.append(json.load(f))
            Nw_U_bar_RMS.append(np.sqrt(np.mean(np.array(Nw_dict_all[i]['Nw_U_bar'])**2)))
            Hw_U_bar_RMS.append(np.sqrt(np.mean(np.array(Nw_dict_all[i]['Hw_U_bar'])**2)))
            Nw_D_loc.append(np.array(Nw_dict_all[i]['Nw_D_loc']))
            Hw_D_loc.append(np.array(Nw_dict_all[i]['Hw_D_loc']))
    n_cases = len(Nw_dict_all)
    for dof in [1,2,3]:
        if dof >= 3:
            func = deg
        else:
            def func(x): return x
        ##################################
        # LINE PLOTS
        ##################################
        str_dof = ["$\Delta_x$ $[m]$",
                   "$\Delta_y$ $[m]$",
                   "$\Delta_z$ $[m]$",
                   "$\Delta_{rx}$ $[\degree]$",
                   "$\Delta_{ry}$ $[\degree]$",
                   "$\Delta_{rz}$ $[\degree]$"]
        plt.figure(dpi=500)
        plt.title(f'Static wind response ({n_cases} worst storms)')
        for case in range(n_cases):
            label1, label2 = ('Inhomogeneous (all cases)', 'Homogeneous (all cases)') if case == 0 else (None,None)
            plt.plot(x, func(Nw_D_loc[case][:n_g_nodes, dof]), lw=1.2, alpha=0.25, c='orange', label=label1)
            plt.plot(x, func(Hw_D_loc[case][:n_g_nodes, dof]), lw=1.2, alpha=0.25, c='blue', label=label2)
        plt.plot(x, func(np.max(np.array([Nw_D_loc[case][:n_g_nodes, dof] for case in range(n_cases)]), axis=0)), alpha=0.7, c='orange', lw=3, label=f'Inhomogeneous (envelope)')
        plt.plot(x, func(np.min(np.array([Nw_D_loc[case][:n_g_nodes, dof] for case in range(n_cases)]), axis=0)), alpha=0.7, c='orange', lw=3)
        plt.plot(x, func(np.max(np.array([Hw_D_loc[case][:n_g_nodes, dof] for case in range(n_cases)]), axis=0)), alpha=0.7, c='blue', lw=3, label=f'Homogeneous (envelope)')
        plt.plot(x, func(np.min(np.array([Hw_D_loc[case][:n_g_nodes, dof] for case in range(n_cases)]), axis=0)), alpha=0.7, c='blue', lw=3)
        plt.xlabel('x [m]  (Position along the arc)')
        plt.ylabel(str_dof[dof])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
        plt.grid()
        plt.tight_layout()
        plt.savefig(rf'results\sw_lines_inhomog_VS_homog_dof_{dof}.png')
        plt.show()
        ##################################
        # SCATTER PLOTS
        ##################################
        str_dof = ["$|\Delta_x|_{max}$ $[m]$",
                   "$|\Delta_y|_{max}$ $[m]$",
                   "$|\Delta_z|_{max}$ $[m]$",
                   "$|\Delta_{rx}|_{max}$ $[\degree]$",
                   "$|\Delta_{ry}|_{max}$ $[\degree]$",
                   "$|\Delta_{rz}|_{max}$ $[\degree]$"]
        plt.figure(dpi=100)
        plt.title(f'Static wind response ({n_cases} worst storms)')
        for case in range(n_cases):
            label1, label2 = ('Inhomogeneous (all cases)', 'Homogeneous (all cases)') if case == 0 else (None, None)
            plt.scatter(Nw_U_bar_RMS[case], func(np.max(np.abs(Nw_D_loc[case][:n_g_nodes, dof]))), marker='x', s=10, alpha=0.7, c='orange', label=label1)
            plt.scatter(Hw_U_bar_RMS[case], func(np.max(np.abs(Hw_D_loc[case][:n_g_nodes, dof]))), marker='o', s=10, alpha=0.7, c='blue', label=label2)
        plt.xlabel(r'$\bar{U}_{RMS}$ [m/s]')
        plt.ylabel(str_dof[dof])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
        plt.grid()
        plt.tight_layout()
        plt.savefig(rf'results\sw_scatter_inhomog_VS_homog_dof_{dof}.png')
        plt.show()
Nw_sw_plot()

def Nw_scatter_plots():
    """Inhomogeneous wind buffeting plots"""
    # Getting the FD results df file
    my_result_path = os.path.join(os.getcwd(), r'results')
    results_paths_FD = []
    for path in os.listdir(my_result_path):
        if path[:16] == "FD_std_delta_max":
            results_paths_FD.append(path)
    for obj in list(enumerate(results_paths_FD)): print(obj)  # print list of files for user to choose
    file_idx = input('Select which file to plot:')
    file_to_plot = os.path.join(my_result_path, results_paths_FD[int(file_idx)])
    results_df = pd.read_csv(file_to_plot)
    n_Nw_idxs = results_df['Nw_idx'].max() + 1  # to account for 0 idx
    # Getting the Nw wind properties into the same df
    my_Nw_path = os.path.join(os.getcwd(), r'intermediate_results', 'static_wind')
    Nw_dict_all, Nw_U_bar_RMS, Hw_U_bar_RMS = [], [], []  # RMS = Root Mean Square, such that the U_bar averages along the fjord are energy-equivalent
    for i in range(n_Nw_idxs):
        Nw_path = os.path.join(my_Nw_path, f'Nw_dict_{i}.json')
        with open(Nw_path, 'r') as f:
            Nw_dict_all.append(json.load(f))
            Nw_U_bar_RMS.append(np.sqrt(np.mean(np.array(Nw_dict_all[i]['Nw_U_bar'])**2)))
            Hw_U_bar_RMS.append(np.sqrt(np.mean(np.array(Nw_dict_all[i]['Hw_U_bar'])**2)))
    results_df['Nw_U_bar_RMS'] = results_df['Nw_idx'].map(dict((i,j) for i,j in enumerate(Nw_U_bar_RMS)))
    results_df['Hw_U_bar_RMS'] = results_df['Nw_idx'].map(dict((i, j) for i, j in enumerate(Hw_U_bar_RMS)))
    # Plotting
    for dof in [1,2,3]:
        str_dof = ["$\sigma_{x, max}$ $[m]$",
                   "$\sigma_{y, max}$ $[m]$",
                   "$\sigma_{z, max}$ $[m]$",
                   "$\sigma_{rx, max}$ $[\degree]$",
                   "$\sigma_{ry, max}$ $[\degree]$",
                   "$\sigma_{rz, max}$ $[\degree]$"]
        Nw_row_bools = results_df['Nw_or_equiv_Hw'] == 'Nw'
        Hw_row_bools = results_df['Nw_or_equiv_Hw'] == 'Hw'
        Nw_x, Hw_x = results_df[Nw_row_bools]['Nw_U_bar_RMS'], results_df[Hw_row_bools]['Hw_U_bar_RMS']
        Nw_y, Hw_y = results_df[Nw_row_bools][f'std_max_dof_{dof}'], results_df[Hw_row_bools][f'std_max_dof_{dof}']
        if dof >= 3:
            Nw_y, Hw_y = deg(Nw_y), deg(Hw_y)
        plt.figure(figsize=(5,5), dpi=300)
        plt.title(f'Buffeting response ({n_Nw_idxs} worst storms)')
        plt.scatter(Nw_x, Nw_y, marker='x', s=10, alpha=0.7, c='orange', label='Inhomogeneous')
        plt.scatter(Hw_x, Hw_y, marker='o', s=10, alpha=0.7, c='blue', label='Homogeneous')
        plt.ylabel(str_dof[dof])
        plt.xlabel(r'$\bar{U}_{RMS}$ [m/s]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(rf'results\buffeting_inhomog_VS_homog_dof_{dof}.png')
        plt.show()
Nw_scatter_plots()


