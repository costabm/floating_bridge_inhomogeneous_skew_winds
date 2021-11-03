# -*- coding: utf-8 -*-
"""
created: 2019
author: Bernardo Costa

Rayleigh damping matrix

Useful references:
https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/default.htm?startat=pt05ch20s01abm43.html  -> the first equation in this page is the one being solved for omega 1 and omega 2
https://www.orcina.com/webhelp/OrcaFlex/Content/html/Rayleighdamping,Guidance.htm  -> alpha, beta notation is used instead of mu, lambda (respectively)
"""

import numpy as np
from simple_5km_bridge_geometry import g_node_coor, p_node_coor
from frequency_dependencies.read_Aqwa_file import added_damping_func
from transformations import T_LsGs_6p_func

def rayleigh_coefficients_func(damping_ratio, Ti, Tj):
    """
    The order of Ti and Tj is irrelevant.
    :param damping_ratio: example: 0.05 (5% damping ratio)
    :param Ti: period to which rayleigh damping will be tuned to match the desired damping ratio. Example: 1 (s)
    :param Tj: period to which rayleigh damping will be tuned to match the desired damping ratio. Example: 150 (s)
    :return:
    """
    a = np.array([[1.0 / 2.0 / (2 * np.pi / Ti), (2 * np.pi / Ti) / 2.0],
                  [1.0 / 2.0 / (2 * np.pi / Tj), (2 * np.pi / Tj) / 2.0]])
    b = np.array([damping_ratio, damping_ratio])

    coef = np.linalg.solve(a, b)  # Rayleigh coefficients
    alpha = coef[0]  # Rayleigh coefficient alpha
    beta = coef[1]  # Rayleigh coefficient beta

    return alpha, beta

def rayleigh_damping_matrix_func(M, K, damping_ratio, Ti, Tj):
    """
    The order of Ti and Tj is irrelevant.
    :param M: Mass matrix
    :param K: Stiffness matrix
    :param damping_ratio: example: 0.05 (5% damping ratio)
    :param Ti: period to which rayleigh damping will be tuned to match the desired damping ratio. Example: 1 (s)
    :param Tj: period to which rayleigh damping will be tuned to match the desired damping ratio. Example: 150 (s)
    :return:
    """
    alpha, beta = rayleigh_coefficients_func(damping_ratio, Ti, Tj)

    C = alpha * M + beta * K  # Formula for the Rayleigh damping

    return C

def P1_damping_added_func(w_array=None, make_freq_dep=False):
    """
    :param w_array: array with circular frequencies. None is used when make_freq_dep = False
    :param make_freq_dep: (bool) Make it frequency-dependent.
    :return: One pontoon hydrodynamic added damping, in pontoon local coordinates (x_pontoon = y_girder and y_pontoon = -x_girder)
    """

    if not make_freq_dep:
        assert w_array is None
        w_infinite = np.array([2*np.pi * (1/1000)])
        w_horizontal = np.array([2*np.pi * (1/100)])
        w_vertical = np.array([2*np.pi * (1/6)])
        w_torsional = np.array([2*np.pi * (1/5)])
        added_mass = added_damping_func(w_infinite, plot=False)[0]  # match infinite frequency (should still be equal to T = 100 s), for all off-diagonals
        # A hybrid choice of frequencies. Each DOF is fixed at its dominant response frequency:
        added_mass[0,0] = added_damping_func(w_horizontal, plot=False)[0][0,0]  # match T = 100 s (pontoon surge)
        added_mass[1,1] = added_damping_func(w_horizontal, plot=False)[0][1,1]  # match T = 100 s (pontoon sway)
        added_mass[2,2] = added_damping_func(w_vertical  , plot=False)[0][2,2]  # match T = 6 s (pontoon heave)
        added_mass[3,3] = added_damping_func(w_vertical  , plot=False)[0][3,3]  # match T = 6 s (pontoon roll)
        added_mass[4,4] = added_damping_func(w_torsional , plot=False)[0][4,4]  # match T = 5 s (pontoon pitch)
        added_mass[5,5] = added_damping_func(w_horizontal, plot=False)[0][5,5]  # match T = 100 s (pontoon yaw)
        # # IF OFF-DIAGONALS ARE TO BE FORCED TO 0 (to avoid torsional modes with strong vertical component)
        # added_mass = np.diag(np.diag(added_mass))
        return added_mass  # shape (6, 6)
    else:
        assert make_freq_dep
        assert w_array is not None
        return added_damping_func(w_array, plot=False)  # shape (n_freq, 6, 6)

def added_damping_global_matrix_func(w_array=None, make_freq_dep=False):
    g_node_num = len(g_node_coor)
    n_pontoons = len(p_node_coor)

    T_LsGs_6p = T_LsGs_6p_func(g_node_coor, p_node_coor)  # to be used for the pontoons
    T_GsLs_6p = np.transpose(T_LsGs_6p, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))

    if not make_freq_dep:
        P1_damping_added = P1_damping_added_func(w_array=None, make_freq_dep=False)
        p_damping_added_local = np.repeat(P1_damping_added[np.newaxis,:,:], n_pontoons, axis=0)  # shape (n_pontoons, 6, 6)
        p_damping_global = np.einsum('eij,ejk,ekl->eil', T_GsLs_6p, p_damping_added_local, T_LsGs_6p, optimize=True)
        matrix = np.zeros(((g_node_num + n_pontoons) * 6, (g_node_num + n_pontoons) * 6))
        for p in range(n_pontoons):
            matrix[6*(g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += p_damping_global[p]
        return matrix
    else:
        assert make_freq_dep
        assert w_array is not None
        n_freq = len(w_array)
        matrix = np.zeros((n_freq, (g_node_num + n_pontoons) * 6, (g_node_num + n_pontoons) * 6))
        P1_damping_added_local = P1_damping_added_func(w_array, make_freq_dep)  # shape (n_freq, 6, 6)
        p_damping_added_local = np.repeat(P1_damping_added_local[:,np.newaxis,:,:], n_pontoons, axis=1)  # shape (n_freq, n_pontoons, 6, 6)
        p_damping_global = np.einsum('eij,wejk,ekl->weil', T_GsLs_6p, p_damping_added_local, T_LsGs_6p, optimize=True)
        matrix = np.zeros((n_freq, (g_node_num + n_pontoons) * 6, (g_node_num + n_pontoons) * 6))
        for p in range(n_pontoons):
            matrix[:, 6 * (g_node_num + p):6 * (g_node_num + p) + 6, 6 * (g_node_num + p):6 * (g_node_num + p) + 6] += p_damping_global[:,p,:,:]
        return matrix

