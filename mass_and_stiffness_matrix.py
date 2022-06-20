"""
Generates the mass (kg), stiffness and geometric stiffness (N/m or N/rad) matrices  of a Bjornafjord-like
floating bridge.

Notation:
b - beam (usually associated with a 12x12 matrix)
g - girder
p - pontoon
c - column

Updated: 04/2020
author: Bernardo Costa
email: bernamdc@gmail.com
"""

import numpy as np
from simple_5km_bridge_geometry import p_node_idx, c_height
from transformations import T_LsGs_12b_func, T_LsGs_12c_func, T_LsGs_6p_func, g_elem_L_3D_func
from frequency_dependencies.read_Aqwa_file import pontoon_area_func, pontoon_Ixx_Iyy_func, pontoon_displacement_func, pontoon_stiffness_func, added_mass_func, added_damping_func


########################################################################################################################
# Global variables: Mostly obtained from "Appendix B - SBJ-30-C3-NOR-90-RE-102 Global analyses rev A.pdf"
########################################################################################################################
E = 210E9  # (Pa) (Young Modulus)
A = 1.43  # (m2) (CS Area)
Iy = 2.67  # (m4) (CS Weak Axis Inertia)
Iz = 114.83  # (m4) (CS Strong Axis Inertia)
J = 6.88  # (m4) (CS Torsional Inertia)
poissonratio = 0.3  # (for steel)
zbridge = 14.5  # (m) (deck height above water, measured at the Shear Centre!))
linmass = 175.1 * 1000  # (N/m) (linear mass) (== 17850 kg/m)  # Obtained from Phase 3 document SBJ-30-C3-NOR-90-RE-102 Version: A, Table 2-2. Slightly different from Phase 5 Appendix F – Global Analyses - Modelling and assumptions, Table 4-1
c_linmass = 7200 * 9.81  # (N/m) (column linear mass)
c_A = 0.872  # (m2) (Column area)
c_Iy = 5.53  # (m4) (Column Inertia y)
c_Iz = c_Iy  # (m4) (Column Inertia z)
c_J = 11.06  # (m4) (Column torsional inertia. Circular -> J = I0 = Iy+Iz)
c_diameter = 7.16  # (m4)
c_CD = 1.05  # drag coefficient.
stiffspring = 1E15  # (N/m) or (Nm/rad) (value of a fixed support spring)
g = 9.81  # (m/s2) (gravity acc.)
water_gamma = 10000  # (N/m3) (water density)
SDL = (4.89+0.6+1.33)*1000 * g  # (N/m) (asphalt + railings + transv stiffeners. See phase 3, MUL, App. A, Table 4-3)
# Dependent variables:
G = E / (2 * (1 + poissonratio))  # (Pa) (Shear Modulus)
########################################################################################################################


# # todo: TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
# E = 210E9 / 10 # (Pa) (Young Modulus)
# A = 1.43 / 1  # (m2) (CS Area)
# Iy = 2.67 / 1  # (m4) (CS Weak Axis Inertia)
# Iz = 114.83 / 3000  # (m4) (CS Strong Axis Inertia)
# J = 6.88 / 100  # (m4) (CS Torsional Inertia)
# poissonratio = 0.3  # (for steel)
# zbridge = 14.5  # (m) (deck height above water, measured at the Shear Centre!))
# linmass = 175.1 * 1000 * 100  # (N/m) (linear mass)
# c_linmass = 7200 * 9.81  # (N/m) (column linear mass)
# c_A = 0.872  # (m2) (Column area)
# c_Iy = 5.53  # (m4) (Column Inertia y)
# c_Iz = c_Iy  # (m4) (Column Inertia z)
# c_J = 11.06  # (m4) (Column torsional inertia. Circular -> J = I0 = Iy+Iz)
# c_diameter = 7.16  # (m4)
# c_CD = 1.05  # drag coefficient.
# stiffspring = 1E14  # (N/m) or (Nm/rad) (value of a fixed support spring)
# g = 9.81  # (m/s2) (gravity acc.)
# water_gamma = 10000  # (N/m3) (water density)
# SDL = (4.89+0.6+1.33)*1000 * g  # (N/m) (asphalt + railings + transverse stiffeners. See phase 3, MUL, App. A, Table 4-3)
# # Dependent variables:
# G = E / (2 * (1 + poissonratio))  # (Pa) (Shear Modulus)
# # todo: TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG


########################################################################################################################
# MASS MATRIX
########################################################################################################################

# Pontoons properties. Mass ("self-mass" and added-mass) in Tons:
def P1_mass_self_func():  # Pontoon type 1. Local Pontoon Coordinates.
    # # OLD VERSION #######
    # p_length = 58  # (m)
    # p_width = 10  # (m)
    # displacement = 2793  # (m3) (Pontoon water displacement, probably for all Permanent Loads)
    # p11 = displacement * water_gamma - (linmass + SDL) * 100 - 7.5 * c_linmass  # (N).((100 span. 7.5 column height))
    # p22 = p11
    # p33 = p11
    # p44 = p11/(p_length*p_width) * p_length * p_width**3 / 12
    # p55 = p11/(p_length*p_width) * p_width * p_length**3 / 12
    # p66 = p11 * (p_length**2 + p_width**2) / 12
    # # return np.diag([p11, p22, p33, p44, p55, p66]) / g  # Units in kg (N/g)

    # NEW VERSION
    p11 = 985 * 1000  # Table 4-2. PDF p. 44 Appendix F. AMX "Global Analyses - Modelling and assumptions"
    p22 = p11  # Table 4-2. PDF p. 44 Appendix F. AMX "Global Analyses - Modelling and assumptions"
    p33 = p11  # Table 4-2. PDF p. 44 Appendix F. AMX "Global Analyses - Modelling and assumptions"
    p44 = 33.1E6  # Table 4-2. PDF p. 44 Appendix F. AMX "Global Analyses - Modelling and assumptions"
    p55 = 252E6  # Table 4-2. PDF p. 44 Appendix F. AMX "Global Analyses - Modelling and assumptions"
    p66 = 252E6  # Table 4-2. PDF p. 44 Appendix F. AMX "Global Analyses - Modelling and assumptions"
    return np.diag([p11, p22, p33, p44, p55, p66])  # Units in kg (N/g)

def P1_mass_added_func(w_array=None, make_freq_dep=False):  # todo: freq. dependency
    """
    :param w_array: array with circular frequencies. None is used when make_freq_dep = False
    :param make_freq_dep: (bool) Make it frequency-dependent.
    :return: One pontoon hydrodynamic added mass, in pontoon local coordinates (x_pontoon = y_girder and y_pontoon = -x_girder)
    """
    # # OLD VERSION #######
    # p11 = 3E5 * g  # (N) (Pontoon Surge. added mass, ONLY correct for T>15 sec))
    # p22 = 3E6 * g  # (N) (Pontoon Sway. added mass, ONLY correct for T>15 sec)
    # p33 = 0.3E7 * g  # (N) (Pontoon Heave. added mass, ONLY correct for T=0 to 10 sec)
    # p44 = 0
    # p55 = 0
    # p66 = 0
    # return np.diag([p11, p22, p33, p44, p55, p66]) / g  # Units in kg (N/g)

    # NEW VERSION
    if not make_freq_dep:
        assert w_array is None
        w_infinite = np.array([2*np.pi * (1/1000)])
        w_horizontal = np.array([2*np.pi * (1/100)])
        w_vertical = np.array([2*np.pi * (1/6)])
        w_torsional = np.array([2*np.pi * (1/5)])
        added_mass = added_mass_func(w_infinite, plot=False)[0]  # match infinite frequency (should still be equal to T = 100 s), for all off-diagonals
        # A hybrid choice of frequencies. Each DOF is fixed at its dominant response frequency:
        added_mass[0,0] = added_mass_func(w_horizontal, plot=False)[0][0,0]  # match T = 100 s (pontoon surge)
        added_mass[1,1] = added_mass_func(w_horizontal, plot=False)[0][1,1]  # match T = 100 s (pontoon sway)
        added_mass[2,2] = added_mass_func(w_vertical  , plot=False)[0][2,2]  # match T = 6 s (pontoon heave)
        added_mass[3,3] = added_mass_func(w_vertical  , plot=False)[0][3,3]  # match T = 6 s (pontoon roll)
        added_mass[4,4] = added_mass_func(w_torsional , plot=False)[0][4,4]  # match T = 5 s (pontoon pitch)
        added_mass[5,5] = added_mass_func(w_horizontal, plot=False)[0][5,5]  # match T = 100 s (pontoon yaw)
        # # IF OFF-DIAGONALS ARE TO BE FORCED TO 0 (to avoid torsional modes with strong vertical component)
        # added_mass = np.diag(np.diag(added_mass))
        return added_mass  # shape (6, 6)

    elif make_freq_dep:
        return added_mass_func(w_array, plot=False)  # shape (n_freq, 6, 6)

# Girder 12dof element mass matrix:
def mass_matrix_12b_local_func(g_node_coor, matrix_type='consistent'):
    """Creates a 'lumped' or 'consistent' mass matrix. Read assumptions in the code. Unit: kg (linmass/g)
    References:
        Structural Dynamics Theory and Computation - Mario Paz & William Leigh
        Structural Dynamics - Einar N. Strommen
    """
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem_L_3D = g_elem_L_3D_func(g_node_coor)

    # Rotational Mass is assumed to be described by I0 = Iy + Iz and linmass. It would be correct if we only
    # had self-weight (and no transverse stiffeners). Slightly incorrect for the super imposed dead loads.
    I0 = Iy + Iz  # (m4) (Polar moment of inertia. Used for rotational mass)
    mass_elem_loc = np.zeros((g_elem_num, 12, 12))
    if matrix_type == 'lumped':
        rotmass = linmass * I0 / A  # (Nm2/m) (torsional mass moment of inertia)
        mass_elem_loc_0 = np.array([[linmass, rotmass]] * g_elem_num)
        for n in range(g_elem_num):
            mass_elem_loc[n] = mass_elem_loc_0[n][0] * g_elem_L_3D[n] / 2 * np.diag(
                [1, 1, 1, mass_elem_loc_0[n][1] / mass_elem_loc_0[n][0], 0, 0, 1, 1, 1,
                 mass_elem_loc_0[n][1] / mass_elem_loc_0[n][0], 0, 0]) / g
    elif matrix_type == 'consistent':  # todo: update to include eccentricities from shear centre. See Strommens book.
        mass_elem_loc[:, 0, 0] = linmass / g * g_elem_L_3D * 1 / 3
        mass_elem_loc[:, 1, 1] = linmass / g * g_elem_L_3D * 13 / 35
        mass_elem_loc[:, 2, 2] = linmass / g * g_elem_L_3D * 13 / 35
        mass_elem_loc[:, 3, 3] = linmass / g * g_elem_L_3D * I0 / (3 * A)
        mass_elem_loc[:, 4, 4] = linmass / g * g_elem_L_3D * g_elem_L_3D ** 2 / 105
        mass_elem_loc[:, 5, 5] = linmass / g * g_elem_L_3D * g_elem_L_3D ** 2 / 105
        mass_elem_loc[:, 6, 6] = linmass / g * g_elem_L_3D * 1 / 3
        mass_elem_loc[:, 7, 7] = linmass / g * g_elem_L_3D * 13 / 35
        mass_elem_loc[:, 8, 8] = linmass / g * g_elem_L_3D * 13 / 35
        mass_elem_loc[:, 9, 9] = linmass / g * g_elem_L_3D * I0 / (3 * A)
        mass_elem_loc[:, 10, 10] = linmass / g * g_elem_L_3D * g_elem_L_3D ** 2 / 105
        mass_elem_loc[:, 11, 11] = linmass / g * g_elem_L_3D * g_elem_L_3D ** 2 / 105
        mass_elem_loc[:, 0, 6] = linmass / g * g_elem_L_3D * 1 / 6
        mass_elem_loc[:, 1, 5] = linmass / g * g_elem_L_3D * 11 * g_elem_L_3D / 210
        mass_elem_loc[:, 1, 7] = linmass / g * g_elem_L_3D * 9 / 70
        mass_elem_loc[:, 1, 11] = linmass / g * g_elem_L_3D * (-13) * g_elem_L_3D / 420
        mass_elem_loc[:, 2, 4] = linmass / g * g_elem_L_3D * (-11) * g_elem_L_3D / 210
        mass_elem_loc[:, 2, 8] = linmass / g * g_elem_L_3D * 9 / 70
        mass_elem_loc[:, 2, 10] = linmass / g * g_elem_L_3D * 13 * g_elem_L_3D / 420  # positive! Error in book from Mario Paz!!
        mass_elem_loc[:, 3, 9] = linmass / g * g_elem_L_3D * I0 / (6 * A)
        mass_elem_loc[:, 4, 8] = linmass / g * g_elem_L_3D * (-13) * g_elem_L_3D / 420
        mass_elem_loc[:, 4, 10] = linmass / g * g_elem_L_3D * (-g_elem_L_3D ** 2) / 140
        mass_elem_loc[:, 5, 7] = linmass / g * g_elem_L_3D * 13 * g_elem_L_3D / 420
        mass_elem_loc[:, 5, 11] = linmass / g * g_elem_L_3D * (-g_elem_L_3D ** 2) / 140
        mass_elem_loc[:, 7, 11] = linmass / g * g_elem_L_3D * (-11) * g_elem_L_3D / 210
        mass_elem_loc[:, 8, 10] = linmass / g * g_elem_L_3D * 11 * g_elem_L_3D / 210
        # Symmetrizing:
        mass_elem_loc[:, 6, 0] = mass_elem_loc[:, 0, 6]
        mass_elem_loc[:, 5, 1] = mass_elem_loc[:, 1, 5]
        mass_elem_loc[:, 7, 1] = mass_elem_loc[:, 1, 7]
        mass_elem_loc[:, 11, 1] = mass_elem_loc[:, 1, 11]
        mass_elem_loc[:, 4, 2] = mass_elem_loc[:, 2, 4]
        mass_elem_loc[:, 8, 2] = mass_elem_loc[:, 2, 8]
        mass_elem_loc[:, 10, 2] = mass_elem_loc[:, 2, 10]
        mass_elem_loc[:, 9, 3] = mass_elem_loc[:, 3, 9]
        mass_elem_loc[:, 8, 4] = mass_elem_loc[:, 4, 8]
        mass_elem_loc[:, 10, 4] = mass_elem_loc[:, 4, 10]
        mass_elem_loc[:, 7, 5] = mass_elem_loc[:, 5, 7]
        mass_elem_loc[:, 11, 5] = mass_elem_loc[:, 5, 11]
        mass_elem_loc[:, 11, 7] = mass_elem_loc[:, 7, 11]
        mass_elem_loc[:, 10, 8] = mass_elem_loc[:, 8, 10]
    return mass_elem_loc

# Column 12dof element mass matrix (1st node: pontoon node; 2nd node: girder node):
def mass_matrix_12c_local_func(p_node_coor, matrix_type='consistent'):
    """Creates a 'lumped' or 'consistent' mass matrix. Read assumptions in the code. Unit: kg (linmass/g)
    References:
        Structural Dynamics Theory and Computation - Mario Paz & William Leigh
        Structural Dynamics - Einar N. Strommen
    """
    n_pontoons = len(p_node_coor)
    # Rotational Mass is assumed to be described by I0 = Iy + Iz and linmass. It would be correct if we only
    # had self-weight (and no transverse stiffeners). Slightly incorrect for the super imposed dead loads.
    c_I0 = c_Iy + c_Iz  # (m4) (Polar moment of inertia. Used for rotational mass)
    mass_elem_loc = np.zeros((n_pontoons, 12, 12))
    if matrix_type == 'lumped':
        rotmass = c_linmass * c_I0 / c_A  # (Nm2/m) (torsional mass moment of inertia)
        mass_elem_loc_0 = np.array([[c_linmass, rotmass]] * n_pontoons)
        for n in range(n_pontoons):
            mass_elem_loc[n] = mass_elem_loc_0[n][0] * c_height[n] / 2 * np.diag(
                [1, 1, 1, mass_elem_loc_0[n][1] / mass_elem_loc_0[n][0], 0, 0, 1, 1, 1,
                 mass_elem_loc_0[n][1] / mass_elem_loc_0[n][0], 0, 0]) / g
    elif matrix_type == 'consistent':
        mass_elem_loc[:, 0, 0] = c_linmass / g * c_height * 1 / 3
        mass_elem_loc[:, 1, 1] = c_linmass / g * c_height * 13 / 35
        mass_elem_loc[:, 2, 2] = c_linmass / g * c_height * 13 / 35
        mass_elem_loc[:, 3, 3] = c_linmass / g * c_height * c_I0 / (3 * c_A)
        mass_elem_loc[:, 4, 4] = c_linmass / g * c_height * c_height ** 2 / 105
        mass_elem_loc[:, 5, 5] = c_linmass / g * c_height * c_height ** 2 / 105
        mass_elem_loc[:, 6, 6] = c_linmass / g * c_height * 1 / 3
        mass_elem_loc[:, 7, 7] = c_linmass / g * c_height * 13 / 35
        mass_elem_loc[:, 8, 8] = c_linmass / g * c_height * 13 / 35
        mass_elem_loc[:, 9, 9] = c_linmass / g * c_height * c_I0 / (3 * c_A)
        mass_elem_loc[:, 10, 10] = c_linmass / g * c_height * c_height ** 2 / 105
        mass_elem_loc[:, 11, 11] = c_linmass / g * c_height * c_height ** 2 / 105
        mass_elem_loc[:, 0, 6] = c_linmass / g * c_height * 1 / 6
        mass_elem_loc[:, 1, 5] = c_linmass / g * c_height * 11 * c_height / 210
        mass_elem_loc[:, 1, 7] = c_linmass / g * c_height * 9 / 70
        mass_elem_loc[:, 1, 11] = c_linmass / g * c_height * (-13) * c_height / 420
        mass_elem_loc[:, 2, 4] = c_linmass / g * c_height * (-11) * c_height / 210
        mass_elem_loc[:, 2, 8] = c_linmass / g * c_height * 9 / 70
        mass_elem_loc[:, 2, 10] = c_linmass / g * c_height * (-13) * c_height / 420
        mass_elem_loc[:, 3, 9] = c_linmass / g * c_height * c_I0 / (6 * c_A)
        mass_elem_loc[:, 4, 8] = c_linmass / g * c_height * (-13) * c_height / 420
        mass_elem_loc[:, 4, 10] = c_linmass / g * c_height * (-c_height ** 2) / 140
        mass_elem_loc[:, 5, 7] = c_linmass / g * c_height * 13 * c_height / 420
        mass_elem_loc[:, 5, 11] = c_linmass / g * c_height * (-c_height ** 2) / 140
        mass_elem_loc[:, 7, 11] = c_linmass / g * c_height * (-11) * c_height / 210
        mass_elem_loc[:, 8, 10] = c_linmass / g * c_height * 11 * c_height / 210
        # Symmetrizing:
        mass_elem_loc[:, 6, 0] = mass_elem_loc[:, 0, 6]
        mass_elem_loc[:, 5, 1] = mass_elem_loc[:, 1, 5]
        mass_elem_loc[:, 7, 1] = mass_elem_loc[:, 1, 7]
        mass_elem_loc[:, 11, 1] = mass_elem_loc[:, 1, 11]
        mass_elem_loc[:, 4, 2] = mass_elem_loc[:, 2, 4]
        mass_elem_loc[:, 8, 2] = mass_elem_loc[:, 2, 8]
        mass_elem_loc[:, 10, 2] = mass_elem_loc[:, 2, 10]
        mass_elem_loc[:, 9, 3] = mass_elem_loc[:, 3, 9]
        mass_elem_loc[:, 8, 4] = mass_elem_loc[:, 4, 8]
        mass_elem_loc[:, 10, 4] = mass_elem_loc[:, 4, 10]
        mass_elem_loc[:, 7, 5] = mass_elem_loc[:, 5, 7]
        mass_elem_loc[:, 11, 5] = mass_elem_loc[:, 5, 11]
        mass_elem_loc[:, 11, 7] = mass_elem_loc[:, 7, 11]
        mass_elem_loc[:, 10, 8] = mass_elem_loc[:, 8, 10]
    return mass_elem_loc

# Complete nodal Mass Matrix (after each beam-element mass matrix (12x12) has been rotated to global coordinates):
def mass_matrix_func(g_node_coor, p_node_coor, alpha, w_array=None, make_freq_dep=False):
    """
    Mass matrix in global coordinates of the bridge girder + columns + pontoons.
    The first (g_node_num) rows & columns are respective to the bridge girder g_nodes. The remaining (n_pontoons) rows
    & columns are the pontoon g_nodes. The columns connect the pontoon g_nodes to the girder g_nodes.
    w_array: array with circular frequencies. None is used when make_freq_dep = False
    make_freq_dep: (bool) Make it frequency-dependent.
    """
    g_node_num = len(g_node_coor)
    n_pontoons = len(p_node_coor)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.

    T_LsGs_6p = T_LsGs_6p_func(g_node_coor, p_node_coor)  # to be used for the pontoons
    T_GsLs_6p = np.transpose(T_LsGs_6p, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LsGs_12b = T_LsGs_12b_func(g_node_coor, alpha)  # to be used for the bridge girder beams
    T_GsLs_12b = np.transpose(T_LsGs_12b, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LsGs_12c = T_LsGs_12c_func(g_node_coor, p_node_coor)  # to be used for the bridge columns
    T_GsLs_12c = np.transpose(T_LsGs_12c, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))

    mass_matrix_12b_local = mass_matrix_12b_local_func(g_node_coor)
    mass_matrix_12b_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_12b, mass_matrix_12b_local, T_LsGs_12b, optimize=True)
    mass_matrix_12c_local = mass_matrix_12c_local_func(p_node_coor)
    mass_matrix_12c_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_12c, mass_matrix_12c_local, T_LsGs_12c, optimize=True)

    matrix = np.zeros(((g_node_num + n_pontoons) * 6, (g_node_num + n_pontoons) * 6))
    # Bridge girder part:
    i = 0
    for n in g_elem:
        matrix[i:i+12, i:i+12] += mass_matrix_12b_global[n]  # adding (12x12) matrices, overlapped by (6x6).
        i += 6

    # Pontoon structural masses
    P1_mass_self = P1_mass_self_func()
    p_mass_self_local = np.repeat(P1_mass_self[np.newaxis,:,:], n_pontoons, axis=0)  # shape (n_pontoons, 6, 6)

    # Pontoon hydrodynamic added masses.
    # FREQUENCY-INDEPENDENT MASS MATRIX
    if not make_freq_dep:
        P1_mass_added = P1_mass_added_func(w_array=None, make_freq_dep=False)
        p_mass_added_local = np.repeat(P1_mass_added[np.newaxis,:,:], n_pontoons, axis=0)  # shape (n_pontoons, 6, 6)
        p_mass_local = p_mass_self_local + p_mass_added_local  # p_mass = self_weight + added mass
        p_mass_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_6p, p_mass_local, T_LsGs_6p, optimize=True)

        # Adding Columns' and Pontoons' masses
        # In the future, replace the following matrix building operations with scipy.sparse.diags, if possible.
        for p in range(n_pontoons):
            p_idx = p_node_idx[p]
            # Columns part:
            # adding each of the four (6x6) parts of the (12x12) matrices (See help picture in the "basis" folder!):
            # m22 part (2nd node of the column is the girder node):
            matrix[6*p_idx:6*p_idx+6, 6*p_idx:6*p_idx+6] += mass_matrix_12c_global[p, 6:12, 6:12]
            # m11 part (1st node of the column is the pontoon node). Pontoon mass is also added here:
            matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += mass_matrix_12c_global[p, 0:6, 0:6] + p_mass_global[p]
            # m21 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
            matrix[6*p_idx:6*p_idx+6, 6*(g_node_num + p):6 * (g_node_num + p) + 6] += mass_matrix_12c_global[p, 6:12, 0:6]
            # m12 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
            matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * p_idx:6 * p_idx + 6] += mass_matrix_12c_global[p, 0:6, 6:12]
        return matrix

    # FREQUENCY-DEPENDENT MASS MATRIX
    else:
        assert make_freq_dep
        assert w_array is not None, "w_array is None but it shouldn't since make_freq_dep is True"
        n_freq = len(w_array)
        P1_mass_added = P1_mass_added_func(w_array, make_freq_dep)  # shape (n_freq, 6, 6)
        p_mass_added_local = np.repeat(P1_mass_added[:,np.newaxis,:,:], n_pontoons, axis=1)  # shape (n_freq, n_pontoons, 6, 6)
        p_mass_local = p_mass_self_local[np.newaxis,:,:,:] + p_mass_added_local  # p_mass = self_weight + added mass
        p_mass_global = np.einsum('eij,wejk,ekl->weil', T_GsLs_6p, p_mass_local, T_LsGs_6p, optimize=True)

        for p in range(n_pontoons):
            p_idx = p_node_idx[p]
            # Columns part:
            # adding each of the four (6x6) parts of the (12x12) matrices (See help picture in the "basis" folder!):
            # m22 part (2nd node of the column is the girder node):
            matrix[6*p_idx:6*p_idx+6, 6*p_idx:6*p_idx+6] += mass_matrix_12c_global[p, 6:12, 6:12]
            # m11 part (1st node of the column is the pontoon node). Pontoon mass is also added here:
            matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += mass_matrix_12c_global[p, 0:6, 0:6]
            # m21 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
            matrix[6*p_idx:6*p_idx+6, 6*(g_node_num + p):6 * (g_node_num + p) + 6] += mass_matrix_12c_global[p, 6:12, 0:6]
            # m12 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
            matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * p_idx:6 * p_idx + 6] += mass_matrix_12c_global[p, 0:6, 6:12]
        matrix = np.repeat(matrix[np.newaxis,:,:], n_freq, axis=0)
        for p in range(n_pontoons):
            matrix[:, 6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += p_mass_global[:,p,:,:]

        return matrix

########################################################################################################################
# STIFFNESS MATRIX
########################################################################################################################

# Pontoons properties:
def P1_stiff_func():  # Pontoon type 1. Local Pontoon Coordinates.
    # # OLD VERSION:
    # p11 = 0
    # p22 = 0
    # p33 = 5613 * 1000  # (N/m). Heave.
    # p44 = 49541.7 * 1000  # (Nm/rad). Pontoon roll (bridge pitch)
    # p55 = 1666581.7 * 1000  # (Nm/rad). Pontoon pitch (bridge roll)
    # p66 = 0

    p_stiffness = pontoon_stiffness_func()
    return np.diag(np.diag(p_stiffness))  # removes the small values on the off-diagonals

# Girder 12dof element stiffness matrix:
def stiff_matrix_12b_local_func(g_node_coor):
    """Creates a stiffness matrix. Read assumptions in the code. Unit: N/m (Nm/rad)
    References:
        Structural Dynamics Theory and Computation - Mario Paz & William Leigh
        Structural Dynamics - Einar N. Strommen
    """

    g_elem_L_3D = g_elem_L_3D_func(g_node_coor)
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1

    # Rotational Mass is assumed to be described by I0 = Iy + Iz and linmass. It would be correct if we only
    # had self-weight (and no transverse stiffeners). Slightly incorrect for the superimposed dead loads.
    stiff_elem_loc = np.zeros((g_elem_num, 12, 12))
    stiff_elem_loc[:, 0, 0] = E * A / g_elem_L_3D
    stiff_elem_loc[:, 1, 1] = 12 * E * Iz / g_elem_L_3D ** 3
    stiff_elem_loc[:, 2, 2] = 12 * E * Iy / g_elem_L_3D ** 3
    stiff_elem_loc[:, 3, 3] = G * J / g_elem_L_3D
    stiff_elem_loc[:, 4, 4] = 4 * E * Iy / g_elem_L_3D
    stiff_elem_loc[:, 5, 5] = 4 * E * Iz / g_elem_L_3D
    stiff_elem_loc[:, 6, 6] = E * A / g_elem_L_3D
    stiff_elem_loc[:, 7, 7] = 12 * E * Iz / g_elem_L_3D ** 3
    stiff_elem_loc[:, 8, 8] = 12 * E * Iy / g_elem_L_3D ** 3
    stiff_elem_loc[:, 9, 9] = G * J / g_elem_L_3D
    stiff_elem_loc[:, 10, 10] = 4 * E * Iy / g_elem_L_3D
    stiff_elem_loc[:, 11, 11] = 4 * E * Iz / g_elem_L_3D
    stiff_elem_loc[:, 0, 6] = -E * A / g_elem_L_3D
    stiff_elem_loc[:, 1, 5] = 6 * E * Iz / g_elem_L_3D ** 2
    stiff_elem_loc[:, 1, 7] = -12 * E * Iz / g_elem_L_3D ** 3
    stiff_elem_loc[:, 1, 11] = 6 * E * Iz / g_elem_L_3D ** 2
    stiff_elem_loc[:, 2, 4] = -6 * E * Iy / g_elem_L_3D ** 2
    stiff_elem_loc[:, 2, 8] = -12 * E * Iy / g_elem_L_3D ** 3
    stiff_elem_loc[:, 2, 10] = -6 * E * Iy / g_elem_L_3D ** 2
    stiff_elem_loc[:, 3, 9] = -G * J / g_elem_L_3D
    stiff_elem_loc[:, 4, 8] = 6 * E * Iy / g_elem_L_3D ** 2
    stiff_elem_loc[:, 4, 10] = 2 * E * Iy / g_elem_L_3D
    stiff_elem_loc[:, 5, 7] = -6 * E * Iz / g_elem_L_3D ** 2
    stiff_elem_loc[:, 5, 11] = 2 * E * Iz / g_elem_L_3D
    stiff_elem_loc[:, 7, 11] = -6 * E * Iz / g_elem_L_3D ** 2
    stiff_elem_loc[:, 8, 10] = 6 * E * Iy / g_elem_L_3D ** 2
    # Symmetrizing:
    stiff_elem_loc[:, 6, 0] = stiff_elem_loc[:, 0, 6]
    stiff_elem_loc[:, 5, 1] = stiff_elem_loc[:, 1, 5]
    stiff_elem_loc[:, 7, 1] = stiff_elem_loc[:, 1, 7]
    stiff_elem_loc[:, 11, 1] = stiff_elem_loc[:, 1, 11]
    stiff_elem_loc[:, 4, 2] = stiff_elem_loc[:, 2, 4]
    stiff_elem_loc[:, 8, 2] = stiff_elem_loc[:, 2, 8]
    stiff_elem_loc[:, 10, 2] = stiff_elem_loc[:, 2, 10]
    stiff_elem_loc[:, 9, 3] = stiff_elem_loc[:, 3, 9]
    stiff_elem_loc[:, 8, 4] = stiff_elem_loc[:, 4, 8]
    stiff_elem_loc[:, 10, 4] = stiff_elem_loc[:, 4, 10]
    stiff_elem_loc[:, 7, 5] = stiff_elem_loc[:, 5, 7]
    stiff_elem_loc[:, 11, 5] = stiff_elem_loc[:, 5, 11]
    stiff_elem_loc[:, 11, 7] = stiff_elem_loc[:, 7, 11]
    stiff_elem_loc[:, 10, 8] = stiff_elem_loc[:, 8, 10]
    return stiff_elem_loc

# Column 12dof element stiffness matrix:
def stiff_matrix_12c_local_func(p_node_coor):
    """Creates a stiffness matrix. Read assumptions in the code. Unit: N/m (Nm/rad)
    References:
        Structural Dynamics Theory and Computation - Mario Paz & William Leigh
        Structural Dynamics - Einar N. Strommen
    """
    n_pontoons = len(p_node_coor)

    # Rotational Mass is assumed to be described by I0 = Iy + Iz and linmass. It would be correct if we only
    # had self-weight (and no transverse stiffeners). Slightly incorrect for the super imposed dead loads.
    stiff_elem_loc = np.zeros((n_pontoons, 12, 12))
    stiff_elem_loc[:, 0, 0] = E * c_A / c_height
    stiff_elem_loc[:, 1, 1] = 12 * E * c_Iz / c_height ** 3
    stiff_elem_loc[:, 2, 2] = 12 * E * c_Iy / c_height ** 3
    stiff_elem_loc[:, 3, 3] = G * c_J / c_height
    stiff_elem_loc[:, 4, 4] = 4 * E * c_Iy / c_height
    stiff_elem_loc[:, 5, 5] = 4 * E * c_Iz / c_height
    stiff_elem_loc[:, 6, 6] = E * c_A / c_height
    stiff_elem_loc[:, 7, 7] = 12 * E * c_Iz / c_height ** 3
    stiff_elem_loc[:, 8, 8] = 12 * E * c_Iy / c_height ** 3
    stiff_elem_loc[:, 9, 9] = G * c_J / c_height
    stiff_elem_loc[:, 10, 10] = 4 * E * c_Iy / c_height
    stiff_elem_loc[:, 11, 11] = 4 * E * c_Iz / c_height
    stiff_elem_loc[:, 0, 6] = -E * c_A / c_height
    stiff_elem_loc[:, 1, 5] = 6 * E * c_Iz / c_height ** 2
    stiff_elem_loc[:, 1, 7] = -12 * E * c_Iz / c_height ** 3
    stiff_elem_loc[:, 1, 11] = 6 * E * c_Iz / c_height ** 2
    stiff_elem_loc[:, 2, 4] = -6 * E * c_Iy / c_height ** 2
    stiff_elem_loc[:, 2, 8] = -12 * E * c_Iy / c_height ** 3
    stiff_elem_loc[:, 2, 10] = -6 * E * c_Iy / c_height ** 2
    stiff_elem_loc[:, 3, 9] = -G * c_J / c_height
    stiff_elem_loc[:, 4, 8] = 6 * E * c_Iy / c_height ** 2
    stiff_elem_loc[:, 4, 10] = 2 * E * c_Iy / c_height
    stiff_elem_loc[:, 5, 7] = -6 * E * c_Iz / c_height ** 2
    stiff_elem_loc[:, 5, 11] = 2 * E * c_Iz / c_height
    stiff_elem_loc[:, 7, 11] = -6 * E * c_Iz / c_height ** 2
    stiff_elem_loc[:, 8, 10] = 6 * E * c_Iy / c_height ** 2
    # Symmetrizing:
    stiff_elem_loc[:, 6, 0] = stiff_elem_loc[:, 0, 6]
    stiff_elem_loc[:, 5, 1] = stiff_elem_loc[:, 1, 5]
    stiff_elem_loc[:, 7, 1] = stiff_elem_loc[:, 1, 7]
    stiff_elem_loc[:, 11, 1] = stiff_elem_loc[:, 1, 11]
    stiff_elem_loc[:, 4, 2] = stiff_elem_loc[:, 2, 4]
    stiff_elem_loc[:, 8, 2] = stiff_elem_loc[:, 2, 8]
    stiff_elem_loc[:, 10, 2] = stiff_elem_loc[:, 2, 10]
    stiff_elem_loc[:, 9, 3] = stiff_elem_loc[:, 3, 9]
    stiff_elem_loc[:, 8, 4] = stiff_elem_loc[:, 4, 8]
    stiff_elem_loc[:, 10, 4] = stiff_elem_loc[:, 4, 10]
    stiff_elem_loc[:, 7, 5] = stiff_elem_loc[:, 5, 7]
    stiff_elem_loc[:, 11, 5] = stiff_elem_loc[:, 5, 11]
    stiff_elem_loc[:, 11, 7] = stiff_elem_loc[:, 7, 11]
    stiff_elem_loc[:, 10, 8] = stiff_elem_loc[:, 8, 10]
    return stiff_elem_loc

# Complete nodal Stiffness Matrix (after each beam-element stiff matrix (12x12) has been rotated to global coordinates):
def stiff_matrix_func(g_node_coor, p_node_coor, alpha):
    """
    Stiffness matrix in global coordinates of the bridge girder + columns + pontoons.
    The first (g_node_num*6) rows & columns are respective to the bridge girder g_nodes. Remaining (n_pontoons*6) rows
    & columns are the pontoon g_nodes. The columns connect the pontoon g_nodes to the girder g_nodes.
    """
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.
    n_pontoons = len(p_node_coor)

    T_LsGs_6p = T_LsGs_6p_func(g_node_coor, p_node_coor)  # to be used for the pontoons
    T_GsLs_6p = np.transpose(T_LsGs_6p, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LsGs_12b = T_LsGs_12b_func(g_node_coor, alpha)  # to be used for the bridge girder beams
    T_GsLs_12b = np.transpose(T_LsGs_12b, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LsGs_12c = T_LsGs_12c_func(g_node_coor, p_node_coor)  # to be used for the bridge columns
    T_GsLs_12c = np.transpose(T_LsGs_12c, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))

    p_stiff_local = np.zeros([n_pontoons, 6, 6])
    for i in range(n_pontoons):
        p_stiff_local[i] = P1_stiff_func()
    p_stiff_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_6p, p_stiff_local, T_LsGs_6p, optimize=True)

    stiff_matrix_12c_local = stiff_matrix_12c_local_func(p_node_coor)
    stiff_matrix_12c_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_12c, stiff_matrix_12c_local, T_LsGs_12c, optimize=True)
    stiff_matrix_12b_local = stiff_matrix_12b_local_func(g_node_coor)
    stiff_matrix_12b_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_12b, stiff_matrix_12b_local, T_LsGs_12b, optimize=True)

    matrix = np.zeros(((g_node_num + n_pontoons) * 6, (g_node_num + n_pontoons) * 6))
    # Bridge girder part:
    i = 0
    for n in g_elem:
        matrix[i:i + 12, i:i + 12] += stiff_matrix_12b_global[n]  # adding (12x12) matrices, overlapped by (6x6).
        i += 6
    # Columns part:
    for p in range(n_pontoons):
        p_idx = p_node_idx[p]
        # adding each of the four (6x6) parts of the (12x12) matrices (See help picture in the "basis" folder!):
        # m22 part (2nd node of the column is the girder node):
        matrix[6*p_idx:6*p_idx+6, 6*p_idx:6*p_idx+6] += stiff_matrix_12c_global[p, 6:12, 6:12]
        # m11 part (1st node of the column is the pontoon node):
        matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += stiff_matrix_12c_global[p, 0:6, 0:6]
        # m21 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
        matrix[6*p_idx:6*p_idx+6, 6*(g_node_num + p):6 * (g_node_num + p) + 6] += stiff_matrix_12c_global[p, 6:12, 0:6]
        # m12 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
        matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * p_idx:6 * p_idx + 6] += stiff_matrix_12c_global[p, 0:6, 6:12]
    # Pontoons part:
        matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += p_stiff_global[p]
    # Boundary conditions (first and last girder g_nodes):
    matrix[0:6, 0:6] += np.diag([stiffspring, stiffspring, stiffspring, stiffspring, stiffspring, stiffspring])
    matrix[(g_node_num - 1) * 6:g_node_num * 6, (g_node_num - 1) * 6:g_node_num * 6] += np.diag([stiffspring, stiffspring, stiffspring,
                                                                                                 stiffspring, stiffspring, stiffspring])



    # # # todo: TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG. DELETE FROM HERE
    # node_to_stiffen_idx = [2, 3, 5]
    # # SPECIAL Boundary conditions FOR THE 6 NODE BRIDGE:
    # for n in node_to_stiffen_idx: # node 0 and 4 (end nodes) already previously stiff from boundary conditions.
    #     matrix[n*6:(n+1)*6, n*6:(n+1)*6] += np.diag([stiffspring, stiffspring, stiffspring, stiffspring, stiffspring, stiffspring])
    # # # SUPER SPECIAL - ONLY 1 DOF - FOR THE 6 NODE BRIDGE:
    # # for n in [1]: # Only node 1 is free.
    # #     matrix[n*6:(n+1)*6, n*6:(n+1)*6] += np.diag([stiffspring, 0, stiffspring, stiffspring, stiffspring, stiffspring])
    # # # todo: TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG. DELETE TO HERE


    return matrix

########################################################################################################################
# GEOMETRIC STIFFNESS MATRIX
########################################################################################################################

# Girder 12dof element geometric stiffness matrix:
def geom_stiff_matrix_12b_local_func(g_node_coor, girder_N):
    """
    Local geometric stiffness matrix
    girder_N: Array with axial force at each girder element (K-KG(N)). positive N means compression!)
    """
    g_elem_L_3D = g_elem_L_3D_func(g_node_coor)
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1

    geom_stiff_elem_loc = np.zeros((g_elem_num, 12, 12))
    geom_stiff_elem_loc[:,1,1] = 36 / (30 * g_elem_L_3D) * girder_N
    geom_stiff_elem_loc[:,2,2] = 36 / (30 * g_elem_L_3D) * girder_N
    geom_stiff_elem_loc[:,4,4] = 2 * g_elem_L_3D / 15 * girder_N
    geom_stiff_elem_loc[:,5,5] = 2 * g_elem_L_3D / 15 * girder_N
    geom_stiff_elem_loc[:,7,7] = 36 / (30 * g_elem_L_3D) * girder_N
    geom_stiff_elem_loc[:,8,8] = 36 / (30 * g_elem_L_3D) * girder_N
    geom_stiff_elem_loc[:,10,10] = 2 * g_elem_L_3D / 15 * girder_N
    geom_stiff_elem_loc[:,11,11] = 2 * g_elem_L_3D / 15 * girder_N
    geom_stiff_elem_loc[:,1,5] = 1 / 10 * girder_N
    geom_stiff_elem_loc[:,1,7] = -36 / (30 * g_elem_L_3D) * girder_N
    geom_stiff_elem_loc[:,1,11] = 1 / 10 * girder_N
    geom_stiff_elem_loc[:,2,4] = -1 / 10 * girder_N
    geom_stiff_elem_loc[:,2,8] = -36 / (30 * g_elem_L_3D) * girder_N
    geom_stiff_elem_loc[:,2,10] = -1 / 10 * girder_N
    geom_stiff_elem_loc[:,4,8] = 1 / 10 * girder_N
    geom_stiff_elem_loc[:,4,10] = g_elem_L_3D / 30 * girder_N
    geom_stiff_elem_loc[:,5,7] = -1 / 10 * girder_N
    geom_stiff_elem_loc[:,5,11] = -g_elem_L_3D / 30 * girder_N
    geom_stiff_elem_loc[:,7,11] = -1 / 10 * girder_N
    geom_stiff_elem_loc[:,8,10] = 1 / 10 * girder_N
    # Symmetrizing:
    geom_stiff_elem_loc[:, 5, 1] = geom_stiff_elem_loc[:, 1, 5]
    geom_stiff_elem_loc[:, 7, 1] = geom_stiff_elem_loc[:, 1, 7]
    geom_stiff_elem_loc[:, 11, 1] = geom_stiff_elem_loc[:, 1, 11]
    geom_stiff_elem_loc[:, 4, 2] = geom_stiff_elem_loc[:, 2, 4]
    geom_stiff_elem_loc[:, 8, 2] = geom_stiff_elem_loc[:, 2, 8]
    geom_stiff_elem_loc[:, 10, 2] = geom_stiff_elem_loc[:, 2, 10]
    geom_stiff_elem_loc[:, 8, 4] = geom_stiff_elem_loc[:, 4, 8]
    geom_stiff_elem_loc[:, 10, 4] = geom_stiff_elem_loc[:, 4, 10]
    geom_stiff_elem_loc[:, 7, 5] = geom_stiff_elem_loc[:, 5, 7]
    geom_stiff_elem_loc[:, 11, 5] = geom_stiff_elem_loc[:, 5, 11]
    geom_stiff_elem_loc[:, 11, 7] = geom_stiff_elem_loc[:, 7, 11]
    geom_stiff_elem_loc[:, 10, 8] = geom_stiff_elem_loc[:, 8, 10]
    return geom_stiff_elem_loc

# Column 12dof element geometric stiffness matrix:
def geom_stiff_matrix_12c_local_func(p_node_coor, c_N):
    """
    Local geometric stiffness matrix
    c_N: Array with axial force at each girder element (K-KG(N)). positive N means compression!)
    """
    n_pontoons = len(p_node_coor)

    geom_stiff_elem_loc = np.zeros((n_pontoons, 12, 12))
    geom_stiff_elem_loc[:,1,1] = 36 / (30*c_height) * c_N
    geom_stiff_elem_loc[:,2,2] = 36 / (30*c_height) * c_N
    geom_stiff_elem_loc[:,4,4] = 2 * c_height / 15 * c_N
    geom_stiff_elem_loc[:,5,5] = 2 * c_height / 15 * c_N
    geom_stiff_elem_loc[:,7,7] = 36 / (30*c_height) * c_N
    geom_stiff_elem_loc[:,8,8] = 36 / (30*c_height) * c_N
    geom_stiff_elem_loc[:,10,10] = 2 * c_height / 15 * c_N
    geom_stiff_elem_loc[:,11,11] = 2 * c_height / 15 * c_N
    geom_stiff_elem_loc[:,1,5] = 1 / 10 * c_N
    geom_stiff_elem_loc[:,1,7] = -36 / (30*c_height) * c_N
    geom_stiff_elem_loc[:,1,11] = 1 / 10 * c_N
    geom_stiff_elem_loc[:,2,4] = -1 / 10 * c_N
    geom_stiff_elem_loc[:,2,8] = -36 / (30*c_height) * c_N
    geom_stiff_elem_loc[:,2,10] = -1 / 10 * c_N
    geom_stiff_elem_loc[:,4,8] = 1 / 10 * c_N
    geom_stiff_elem_loc[:,4,10] = c_height / 30 * c_N
    geom_stiff_elem_loc[:,5,7] = -1 / 10 * c_N
    geom_stiff_elem_loc[:,5,11] = -c_height / 30 * c_N
    geom_stiff_elem_loc[:,7,11] = -1 / 10 * c_N
    geom_stiff_elem_loc[:,8,10] = 1 / 10 * c_N
    # Symmetrizing:
    geom_stiff_elem_loc[:, 5, 1] = geom_stiff_elem_loc[:, 1, 5]
    geom_stiff_elem_loc[:, 7, 1] = geom_stiff_elem_loc[:, 1, 7]
    geom_stiff_elem_loc[:, 11, 1] = geom_stiff_elem_loc[:, 1, 11]
    geom_stiff_elem_loc[:, 4, 2] = geom_stiff_elem_loc[:, 2, 4]
    geom_stiff_elem_loc[:, 8, 2] = geom_stiff_elem_loc[:, 2, 8]
    geom_stiff_elem_loc[:, 10, 2] = geom_stiff_elem_loc[:, 2, 10]
    geom_stiff_elem_loc[:, 8, 4] = geom_stiff_elem_loc[:, 4, 8]
    geom_stiff_elem_loc[:, 10, 4] = geom_stiff_elem_loc[:, 4, 10]
    geom_stiff_elem_loc[:, 7, 5] = geom_stiff_elem_loc[:, 5, 7]
    geom_stiff_elem_loc[:, 11, 5] = geom_stiff_elem_loc[:, 5, 11]
    geom_stiff_elem_loc[:, 11, 7] = geom_stiff_elem_loc[:, 7, 11]
    geom_stiff_elem_loc[:, 10, 8] = geom_stiff_elem_loc[:, 8, 10]
    return geom_stiff_elem_loc

# Complete nodal Geometric Stiffness Matrix (after each local beam-element stiff matrix (12x12) has been rotated):
def geom_stiff_matrix_func(g_node_coor, p_node_coor, girder_N, c_N, alpha):
    """
    Geometric stiffness matrix in global coordinates of the bridge girder + columns.
    The first (g_node_num) rows & columns are respective to the bridge girder g_nodes. The remaining (n_pontoons) rows
    & columns are the pontoon g_nodes. The columns connect the pontoon g_nodes to the girder g_nodes.
    girder_N or c_N: Array with axial force at each girder or column element (K-KG(N)). Positive N means compression,
    which is in accordance with the current stiffness matrix formulation.)
    """
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.
    n_pontoons = len(p_node_coor)
    T_LsGs_12b = T_LsGs_12b_func(g_node_coor, alpha)  # to be used for the bridge girder beams
    T_GsLs_12b = np.transpose(T_LsGs_12b, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_LsGs_12c = T_LsGs_12c_func(g_node_coor, p_node_coor)  # to be used for the bridge columns
    T_GsLs_12c = np.transpose(T_LsGs_12c, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))

    geom_stiff_matrix_12b_local = geom_stiff_matrix_12b_local_func(g_node_coor, girder_N)
    geom_stiff_matrix_12b_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_12b, geom_stiff_matrix_12b_local, T_LsGs_12b, optimize=True)
    geom_stiff_matrix_12c_local = geom_stiff_matrix_12c_local_func(p_node_coor, c_N)
    geom_stiff_matrix_12c_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_12c, geom_stiff_matrix_12c_local, T_LsGs_12c, optimize=True)
    matrix = np.zeros(((g_node_num + n_pontoons) * 6, (g_node_num + n_pontoons) * 6))
    # Bridge girder part:
    i = 0
    for n in g_elem:
        matrix[i:i + 12, i:i + 12] += geom_stiff_matrix_12b_global[n]  # adding (12x12) matrices, overlapped by (6x6).
        i += 6
    # Columns part:
    for p in range(n_pontoons):
        p_idx = p_node_idx[p]
        # adding each of the four (6x6) parts of the (12x12) matrices (See help picture in the "basis" folder!):
        # m22 part (2nd node of the column is the girder node):
        matrix[6*p_idx:6*p_idx+6, 6*p_idx:6*p_idx+6] += geom_stiff_matrix_12c_global[p, 6:12, 6:12]
        # m11 part (1st node of the column is the pontoon node):
        matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += geom_stiff_matrix_12c_global[p, 0:6, 0:6]
        # m21 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
        matrix[6*p_idx:6*p_idx+6, 6*(g_node_num + p):6 * (g_node_num + p) + 6] += geom_stiff_matrix_12c_global[p, 6:12, 0:6]
        # m12 part (non-diagonal terms. Careful: not symmetric! See help picture in the "basis" folder!):
        matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * p_idx:6 * p_idx + 6] += geom_stiff_matrix_12c_global[p, 0:6, 6:12]
    return matrix
