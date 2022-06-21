# -*- coding: utf-8 -*-
"""
modified: 05-2020
author: Bernardo Costa
email: bernamdc@gmail.com

This script opens an Aqwa file, reads the values of added mass and damping for each simulated frequency, and gives back
interpolated values at the desired frequencies.
'Hydrodynamic Solver Unit System : Metric: kg, m [N]'
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformations import R_z
import os
import sys

########################################################################################################################
# RAW DATA
########################################################################################################################

# READING AQWA FILE
try:  # works only when "Run"
    project_path = os.path.dirname(os.path.abspath(__file__))
except:  # works when running directly in console
    project_path = os.path.join(sys.path[1], 'frequency_dependencies')  # Path of the project directory. To be used in the Python Console! When a console is opened in Pycharm, the current project path should be automatically added to sys.path.
f = open(project_path + r'\Aqwa_Analysis_(AMC).LIS', 'r')
f = f.readlines()

# AXES REFERENCE CONVENTION. IF PHI=0 -> 'pontoon' local coord. sys. IF PHI = 90 -> 'bridge' local coord. sys (confirm this if new Aqwa file is used):
axes_string = "ANGLE THE PRINCIPAL AXES MAKE WITH"
axes_idx = [i for i, s in enumerate(f) if axes_string in s][0]
if '90' in f[axes_idx]:
    axes_ref = 'bridge'  # x - bridge surge (pontoon sway), y - bridge sway (pontoon surge), z - heave, rx - bridge roll (pontoon pitch) ...
    T_LpLb = R_z(90 * np.pi / 180, dim='6x6').T  # Transformation matrix, from Local bridge to Local pontoon coordinate system
else:
    axes_ref = 'pontoon'  # x - pontoon surge, y - pontoon sway, z - heave, rx - pontoon roll ...

def pontoon_displacement_func():
    # Finding line where info about displacement is
    ini_string = "MASS BASED DISPLACEMENT"
    ini_idx = [i for i, s in enumerate(f) if ini_string in s][0]
    str_line = f[ini_idx] # e.g. 'MASS BASED DISPLACEMENT  . . . . . . . . =   3.70979E+03'
    equal_idx = [i for i, s in enumerate(str_line) if '=' in s][0]  # index of the equal sign.
    str_displacement = str_line[equal_idx+1:]  # string, right from the equal sign
    p_displacement = float(str_displacement)  # float
    return p_displacement

def pontoon_area_func():
    ini_string = "CUT WATER PLANE AREA ."
    ini_idx = [i for i, s in enumerate(f) if ini_string in s][0]
    str_line = f[ini_idx] # e.g. 'MASS BASED DISPLACEMENT  . . . . . . . . =   3.70979E+03'
    equal_idx = [i for i, s in enumerate(str_line) if '=' in s][0]  # index of the equal sign.
    str_area = str_line[equal_idx+1:]  # string, right from the equal sign
    p_area = float(str_area)  # float
    return p_area

def pontoon_Ixx_Iyy_func():
    """PRINCIPAL SECOND MOMENTS OF AREA at CUT WATER PLANE"""  # Apparently always in the local "pontoon" coordinate system
    Ixx_ini_string = "PRINCIPAL SECOND MOMENTS OF AREA"
    Ixx_ini_idx = [i for i, s in enumerate(f) if Ixx_ini_string in s][0]
    Iyy_ini_idx = Ixx_ini_idx + 1
    Ixx_str_line = f[Ixx_ini_idx] # e.g. 'PRINCIPAL SECOND MOMENTS OF AREA        IXX=   1.29190E+04'
    Iyy_str_line = f[Iyy_ini_idx] # e.g. '                                        IYY=   1.55310E+05'
    Ixx_equal_idx = [i for i, s in enumerate(Ixx_str_line) if '=' in s][0]  # index of the equal sign.
    Iyy_equal_idx = [i for i, s in enumerate(Iyy_str_line) if '=' in s][0]  # index of the equal sign.
    Ixx = float(Ixx_str_line[Ixx_equal_idx+1:])
    Iyy = float(Iyy_str_line[Iyy_equal_idx+1:])
    return Ixx, Iyy

def pontoon_stiffness_func():
    ini_string = "STIFFNESS MATRIX"
    ini_idx = [i for i, s in enumerate(f) if ini_string in s][-1] + 7  # [-1] because it's last occurence of "STIFFNESS MATRIX". 5 more rows until table data_in actually starts
    # ...and where it ends
    end_string = "* * * * H Y D R O D Y N A M I C   P A R A M E T E R S   F O R   S T R U C T U R E   1 * * * *"
    for i, s in enumerate(f[ini_idx:]):  # from start, onwards
        if end_string in s:
            end_idx = ini_idx + i - 2  # last row of table (2 rows before end_string)
            break
    # Treating our table with uneven spaces to list of lists of floats
    stiffness_table = [[string for string in row.split()[1:]] for row in f[ini_idx:end_idx + 1]]
    p_stiffness = pd.DataFrame(stiffness_table).dropna().to_numpy(dtype=float)  # drop empty rows. convert to numpy floats
    if axes_ref == 'bridge':
        p_stiffness = T_LpLb @ p_stiffness @ T_LpLb.T  # convert to local pontoon coordinates
    return p_stiffness

def added_mass_full_table_func():
    # ADDED MASS
    # Finding line where info about added mass starts
    ini_string = "ADDED MASS-VARIATION WITH"  # This is the string in AMC's file.

    ini_idx = [i for i, s in enumerate(f) if ini_string in s][0] + 5  # 5 more rows until table data_in actually starts
    # ...and where it ends
    end_string = "* * * * H Y D R O D Y N A M I C   P A R A M E T E R S   F O R   S T R U C T U R E   1 * * * *"
    for i, s in enumerate(f[ini_idx:]):  # from start, onwards
        if end_string in s:
            end_idx = ini_idx + i - 2  # last row of table (2 rows before end_string)
            break
    # Treating our table with uneven spaces to list of lists of floats
    add_mass_table = np.array([[eval(string) for string in row.split()] for row in f[ini_idx:end_idx + 1]])
    w_array_Aqwa = add_mass_table[:, 1]
    add_mass = np.zeros((len(w_array_Aqwa), 6, 6))
    add_mass[:, 0, 0] = add_mass_table[:, 2]
    add_mass[:, 1, 1] = add_mass_table[:, 3]
    add_mass[:, 2, 2] = add_mass_table[:, 4]
    add_mass[:, 3, 3] = add_mass_table[:, 5]
    add_mass[:, 4, 4] = add_mass_table[:, 6]
    add_mass[:, 5, 5] = add_mass_table[:, 7]
    add_mass[:, 0, 2] = add_mass_table[:, 8]
    add_mass[:, 2, 0] = add_mass[:, 0, 2]  # symmetry
    add_mass[:, 0, 4] = add_mass_table[:, 9]
    add_mass[:, 4, 0] = add_mass[:, 0, 4]  # symmetry
    add_mass[:, 1, 3] = add_mass_table[:, 10]
    add_mass[:, 3, 1] = add_mass[:, 1, 3]  # symmetry
    add_mass[:, 1, 5] = add_mass_table[:, 11]
    add_mass[:, 5, 1] = add_mass[:, 1, 5]  # symmetry
    add_mass[:, 2, 4] = add_mass_table[:, 12]
    add_mass[:, 4, 2] = add_mass[:, 2, 4]  # symmetry
    add_mass[:, 3, 5] = add_mass_table[:, 13]
    add_mass[:, 5, 3] = add_mass[:, 3, 5]  # symmetry
    if axes_ref == 'bridge':
        add_mass = T_LpLb @ add_mass @ T_LpLb.T  # convert to local pontoon coordinates
    return w_array_Aqwa, add_mass

def added_damping_full_table_func():
    # ADDED DAMPING
    # Finding line where info about added mass starts
    ini_string = "DAMPING-VARIATION WITH"  # This is the string in AMC's file.

    ini_idx = [i for i, s in enumerate(f) if ini_string in s][0] + 5  # 5 more rows until table data_in actually starts
    # ...and where it ends
    end_string = "* * * * H Y D R O D Y N A M I C   P A R A M E T E R S   F O R   S T R U C T U R E   1 * * * *"
    for i, s in enumerate(f[ini_idx:]):  # from start, onwards
        if end_string in s:
            end_idx = ini_idx + i - 2  # last row of table (2 rows before end_string)
            break
    # Treating our table with uneven spaces to list of lists of floats
    add_damp_table = np.array([[eval(string) for string in row.split()] for row in f[ini_idx:end_idx + 1]])
    w_array_Aqwa = add_damp_table[:, 1]
    add_damp = np.zeros((len(w_array_Aqwa), 6, 6))
    add_damp[:, 0, 0] = add_damp_table[:, 2]
    add_damp[:, 1, 1] = add_damp_table[:, 3]
    add_damp[:, 2, 2] = add_damp_table[:, 4]
    add_damp[:, 3, 3] = add_damp_table[:, 5]
    add_damp[:, 4, 4] = add_damp_table[:, 6]
    add_damp[:, 5, 5] = add_damp_table[:, 7]
    add_damp[:, 0, 2] = add_damp_table[:, 8]
    add_damp[:, 2, 0] = add_damp[:, 0, 2]  # symmetry
    add_damp[:, 0, 4] = add_damp_table[:, 9]
    add_damp[:, 4, 0] = add_damp[:, 0, 4]  # symmetry
    add_damp[:, 1, 3] = add_damp_table[:, 10]
    add_damp[:, 3, 1] = add_damp[:, 1, 3]  # symmetry
    add_damp[:, 1, 5] = add_damp_table[:, 11]
    add_damp[:, 5, 1] = add_damp[:, 1, 5]  # symmetry
    add_damp[:, 2, 4] = add_damp_table[:, 12]
    add_damp[:, 4, 2] = add_damp[:, 2, 4]  # symmetry
    add_damp[:, 3, 5] = add_damp_table[:, 13]
    add_damp[:, 5, 3] = add_damp[:, 3, 5]  # symmetry
    if axes_ref == 'bridge':
        add_damp = T_LpLb @ add_damp @ T_LpLb.T  # convert to local pontoon coordinates
    return w_array_Aqwa, add_damp

########################################################################################################################
# INTERPOLATING
########################################################################################################################
def added_mass_func(w_array, plot = True):
    """
    Interpolates or extrapolates added mass from given Aqwa frequencies, to desired "w_array" frequencies.
    Plots the results if desired.
    :param w_array: array of angular frequencies to which the added mass is evaluated [rad/s]
    :return: add_mass_interp
    """
    # Obtain original full tables of added mass and damping
    w_array_Aqwa, add_mass = added_mass_full_table_func()
    # INTERPOLATING
    add_mass_interp = np.zeros((len(w_array), 6, 6))
    for i in range(6):
        for j in range(6):
            add_mass_interp[:,i,j] = np.interp(w_array, w_array_Aqwa, add_mass[:,i,j])
    # PLOTTING
    if plot:
        plt.figure()
        plt.title('Added Mass')
        plt.plot(2 * np.pi / w_array_Aqwa, add_mass[:, 0, 0], label='C11')
        plt.plot(2 * np.pi / w_array_Aqwa, add_mass[:, 1, 1], label='C22')
        plt.plot(2 * np.pi / w_array_Aqwa, add_mass[:, 2, 2], label='C33')
        plt.plot(2 * np.pi / w_array_Aqwa, add_mass[:, 3, 3], label='C44')
        plt.plot(2 * np.pi / w_array_Aqwa, add_mass[:, 4, 4], label='C55')
        plt.plot(2 * np.pi / w_array_Aqwa, add_mass[:, 5, 5], label='C66')
        plt.legend()
        plt.xlabel('T [s]')
        plt.yscale('log')
        plt.show()
    return add_mass_interp

def added_damping_func(w_array, plot = True):
    """
    Interpolates or extrapolates added damping from given Aqwa frequencies, to desired "w_array" frequencies.
    Plots the results if desired.
    :param w_array: array of angular frequencies to which the added damping is evaluated [rad/s]
    :return: add_damp_interp
    """
    # Obtain original full tables of added mass and damping
    w_array_Aqwa, add_damp = added_damping_full_table_func()
    # INTERPOLATING
    add_damp_interp = np.zeros((len(w_array), 6, 6))
    for i in range(6):
        for j in range(6):
            add_damp_interp[:, i, j] = np.interp(w_array, w_array_Aqwa, add_damp[:, i, j])
    # PLOTTING
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('Added Damping')
        plt.plot(2 * np.pi / w_array_Aqwa, add_damp[:, 0, 0], label='C11')
        plt.plot(2 * np.pi / w_array_Aqwa, add_damp[:, 1, 1], label='C22')
        plt.plot(2 * np.pi / w_array_Aqwa, add_damp[:, 2, 2], label='C33')
        plt.plot(2 * np.pi / w_array_Aqwa, add_damp[:, 3, 3], label='C44')
        plt.plot(2 * np.pi / w_array_Aqwa, add_damp[:, 4, 4], label='C55')
        plt.plot(2 * np.pi / w_array_Aqwa, add_damp[:, 5, 5], label='C66')
        plt.scatter(2 * np.pi / w_array, add_damp_interp[:, 0, 0], label='C11 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_damp_interp[:, 1, 1], label='C22 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_damp_interp[:, 2, 2], label='C33 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_damp_interp[:, 3, 3], label='C44 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_damp_interp[:, 4, 4], label='C55 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_damp_interp[:, 5, 5], label='C66 interpolation', alpha = 0.4, s=10)
        plt.legend()
        plt.xlabel('T [s]')
        # plt.yscale('log')
        plt.show()
    return add_damp_interp

def plot_added_mass_and_damping_func():
    # Obtain original full tables of added mass and damping
    w_array_Aqwa, add_mass = added_mass_full_table_func()
    _w_array_Aqwa, add_damp = added_damping_full_table_func()
    assert all(w_array_Aqwa == _w_array_Aqwa)
    f_array_Aqwa = w_array_Aqwa / (2 * np.pi)
    # PLOTTING
    from cycler import cycler
    default_cycler = (cycler(color=['orange','brown','cornflowerblue','orange','brown', 'cornflowerblue'])) #
                      # + cycler(linestyle=['-','-','-','--','--','--']) +
                      # cycler(linewidth=[2.5,2.5,2.5,2.5,2.5,2.5])) # cycler(linewidth=[1.5,2.0,2.5,1.5,2.0,2.5]))
    plt.rc('axes', prop_cycle=default_cycler)
    # Added mass
    fig,axs = plt.subplots(2,3, dpi=400, figsize=(7,5), sharex=True)
    axs[0,0].plot(f_array_Aqwa, add_mass[:, 0, 0], linewidth=2.5, linestyle='-', label='x') # linewidth=1.5)
    axs[0,0].plot(f_array_Aqwa, add_mass[:, 1, 1], linewidth=2.5, linestyle='-', label='y') # linewidth=2.0)
    axs[0,0].plot(f_array_Aqwa, add_mass[:, 2, 2], linewidth=2.5, linestyle='-', label='z') # linewidth=2.5)
    axs[0,1].plot(f_array_Aqwa, add_mass[:, 3, 3], linewidth=2.5, linestyle='--', label='rx') #  linewidth=1.5, linestyle='--')
    axs[0,1].plot(f_array_Aqwa, add_mass[:, 4, 4], linewidth=2.5, linestyle='--', label='ry') #  linewidth=2.0, linestyle='--')
    axs[0,1].plot(f_array_Aqwa, add_mass[:, 5, 5], linewidth=2.5, linestyle='--', label='rz') #  linewidth=2.5, linestyle='--')
    axs[1,0].set_xlabel(r'Frequency [Hz]')
    axs[1,1].set_xlabel(r'Frequency [Hz]')
    axs[1,2].set_xlabel(r'Frequency [Hz]')
    axs[0,0].set_ylabel(r'Added mass [$kg$]')  # Mass * acc = Newton <=> Mass = N / (m/s**2) = kg m/s**2 / (m/s**2) = kg
    axs[1,0].set_ylabel(r'Potential damping [$kg/s$]')  # Damp * vel = N <=> Damp = N / (m/s) = kg m/s**2 / (m/s) = kg/s
    axs[0,1].set_ylabel(r'Added mass [$kg\/m^2$]')  # Mass * acc = N*m <=> Mass = Nm / (rad/s**2) = kg m**2/s**2 / (rad/s**2) = kg m**2
    axs[1,1].set_ylabel(r'Potential damping [$kg\/m^2/s$]')  # Damp * vel = N*m <=> Damp = Nm / (rad/s) = kg m**2/s**2 / (rad/s) = kg m**2 /s
    # And now the off-diagonals between a translation and a rotation. The units can be tricky. Let's focus on the upper/right matrix triangle of off-diagonals and forget about the lower/left triangle. The process is different for both triangles, but the conclusions would be the same. Here, the results are in N, but these entries are multiplied by rotations. In the lower triangle, C51, the results are in Nm but these entries are multiplied by translations.
    axs[0,2].set_ylabel(r'Added mass [$kg\/m$]')           # For entry, e.g., M15: Mass * acc = N <=> Mass = N / (rad/s**2) = kg m/s**2 / (rad/s**2) = kg m. ALTERNATIVE, e.g. for M51: Mass * acc = Nm <=> Mass = Nm / (m/s**2) = kg m**2/s**2 / (m/s**2) = kg m. Note: When there is no current, added     Mass matrix is symmetric in values AND in units ( https://www.orcina.com/webhelp/OrcaFlex/Content/html/Vesseltheory,Stiffness,addedmassanddamping.htm#VesselTypeDampingLoad )
    axs[1,2].set_ylabel(r'Potential damping [$kg\/m/s$]')  # For entry, e.g., C15: Damp * acc = N <=> Damp = N / (rad/s)    = kg m/s**2 / (rad/s)  = kg m/s. ALTERNATIVE, e.g. for C51: Damp * acc = Nm <=> Damp = Nm / (m/s)    = kg m**2/s**2 / (m/s) =  kg m/s. Note: When there is no current, potential Damp matrix is symmetric in values AND in units ( https://www.orcina.com/webhelp/OrcaFlex/Content/html/Vesseltheory,Stiffness,addedmassanddamping.htm#VesselTypeDampingLoad )

    # Added damping
    axs[1,0].plot(f_array_Aqwa, add_damp[:, 0, 0], linewidth=2.5, linestyle='-', label='x')  # linewidth=1.5)
    axs[1,0].plot(f_array_Aqwa, add_damp[:, 1, 1], linewidth=2.5, linestyle='-', label='y')  # linewidth=2.0)
    axs[1,0].plot(f_array_Aqwa, add_damp[:, 2, 2], linewidth=2.5, linestyle='-', label='z')  # linewidth=2.5)
    axs[1,1].plot(f_array_Aqwa, add_damp[:, 3, 3], linewidth=2.5, linestyle='--', label='rx')  # linewidth=1.5, linestyle='--')
    axs[1,1].plot(f_array_Aqwa, add_damp[:, 4, 4], linewidth=2.5, linestyle='--', label='ry')  # linewidth=2.0, linestyle='--')
    axs[1,1].plot(f_array_Aqwa, add_damp[:, 5, 5], linewidth=2.5, linestyle='--', label='rz')  # linewidth=2.5, linestyle='--')

    # Finding the off diagonals that are more important to plot, and plotting them
    label_list = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    assert(all([check_symmetric(add_mass[i]) for i in range(len(add_mass))]))  # check if all matrices are symmetric
    assert (all([check_symmetric(add_damp[i]) for i in range(len(add_damp))]))  # check if all matrices are symmetric
    n_off_diags_to_plot = 3  # e.g. the 3 largest off-diagonal entries of the added mass will be plotted
    off_diag_add_mass_max = {}
    off_diag_add_damp_max = {}
    for i in range(6):
        for j in range(6):
            if j > i:
                off_diag_add_mass_max[str(i)+str(j)] = np.max(np.abs(add_mass[:, i, j]))
                off_diag_add_damp_max[str(i) + str(j)] = np.max(np.abs(add_damp[:, i, j]))
    off_diag_add_mass_max_sort = sorted(off_diag_add_mass_max.items(), key=lambda x: x[1], reverse=True)  # sorting the max off-diag added masses, to know which are more important
    off_diag_add_damp_max_sort = sorted(off_diag_add_damp_max.items(), key=lambda x: x[1], reverse=True)
    off_diag_add_mass_idxs_to_plot = [x[0] for x in off_diag_add_mass_max_sort[:n_off_diags_to_plot]]
    off_diag_add_damp_idxs_to_plot = [x[0] for x in off_diag_add_damp_max_sort[:n_off_diags_to_plot]]
    assert off_diag_add_mass_idxs_to_plot == off_diag_add_damp_idxs_to_plot, 'Improve legend since different off-diagonals of the added mass and damping are being plotted. Alternatively, reduce n_off_diag_to_plot in hope of getting same off-diags for both cases'
    off_diag_color_list =['black', 'lightgrey', 'gold']
    plt.rc('axes', prop_cycle=default_cycler)
    for c, (i,j) in enumerate(off_diag_add_mass_idxs_to_plot):
        i,j = int(i),int(j)
        axs[0, 2].plot(f_array_Aqwa, add_mass[:, i, j], c=off_diag_color_list[c], linewidth=2., linestyle='dashdot', label=label_list[i]+'-'+label_list[j])
    for c, (i,j) in enumerate(off_diag_add_damp_idxs_to_plot):
        i,j = int(i),int(j)
        axs[1, 2].plot(f_array_Aqwa, add_damp[:, i, j], c=off_diag_color_list[c],  linewidth=2., linestyle='dashdot', label=label_list[i]+'-'+label_list[j])

    h1, l1 = axs[1,0].get_legend_handles_labels()
    h2, l2 = axs[1,1].get_legend_handles_labels()
    h3, l3 = axs[1,2].get_legend_handles_labels()
    for ax in axs.flat:
        ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(project_path, 'added_mass_and_damping.jpg'))
    plt.show()
    # Legend
    plt.figure(dpi=400, figsize=(5+n_off_diags_to_plot,1))
    plt.legend(h1+h2+h3, l1+l2+l3, ncol=6+n_off_diags_to_plot)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(project_path, 'added_mass_and_damping_legend.jpg'))
    plt.show()

