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
    project_path = sys.path[-2]  # Path of the project directory. To be used in the Python Console! When a console is opened in Pycharm, the current project path should be automatically added to sys.path.
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
        plt.scatter(2 * np.pi / w_array, add_mass_interp[:, 0, 0], label='C11 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_mass_interp[:, 1, 1], label='C22 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_mass_interp[:, 2, 2], label='C33 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_mass_interp[:, 3, 3], label='C44 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_mass_interp[:, 4, 4], label='C55 interpolation', alpha = 0.4, s=10)
        plt.scatter(2 * np.pi / w_array, add_mass_interp[:, 5, 5], label='C66 interpolation', alpha = 0.4, s=10)
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

