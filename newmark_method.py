# -*- coding: utf-8 -*-
"""
created: 2019
author: Bernardo Costa

Useful reference:
https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femII/presentation_05_dynamics_v3.pdf

Integration method used:
Newmark Method (gama = 1/2 and beta = 1/4) <=> Constant-Average-Acceleration Method (or trapezoidal rule).
Unconditionally stable algorithm.

Note: For the SDOF, an adapted (by me) Newmark method is used: An analytical explicit (dependent only on previous time step values)
solution for the a_new, v_new, u_new was obtained, instead of using predictors for v_new and u_new.
"""

import numpy as np
# from profiling import profile
#
# @ profile
def MDOF_TD_solver(M, C, K, F, u0, v0, T, dt, gamma=1/2, beta=1/4):
    """Multi-degree of freedom time-domain solver. \n
    Reference: Page 19 (of 55) of "Time Domain Methods - ETHZ course material.pdf" (or of https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femII/presentation_05_dynamics_v3.pdf) \n
    M -- Mass matrix \n
    C -- Damping matrix = damp_ratio* 2*np.sqrt(K*M) \n
    K -- Stiffness matrix (already including KG and/or Kse if necessary) \n
    F -- External force array.       length = len( np.arange(0, T+dt, dt) ) \n
    u0 -- Initial displacement (m) \n
    v0 -- Initial velocity (m/s) \n
    T -- Simulation duration (s) \n
    dt -- Time step (s). Should be smaller then eigen period / 40. \n
    gamma -- Newmark parameter. \n
    beta -- Newmark parameter. \n
    [1] https://en.wikipedia.org/wiki/Newmark-beta_method \n
    """

    time = np.arange(0, T + dt, dt)
    dt_array = np.array([time[i + 1] - time[i] for i in range(len(time[:-1]))])

    # =========================================================================
    # Constant-Average-Acceleration Method. (see Newmark-Beta method)
    # =========================================================================
    # a & a_new are accelerations at time step i-1 & i respectively.
    # v & v_new are velocities    at time step i-1 & i respectively.
    # u & u_new are position      at time step i-1 & i respectively.

    i = 0  # counter.
    dt = dt_array[0]
    u_new = [u0]  # Initial displacement
    v_new = [v0]  # Initial velocity
    const = np.linalg.inv(M + C*gamma*dt + K*beta*dt**2)  # constant (linear TD simulation)
    a_new = [const @ (F[i] - K @ u_new[-1] - C @ v_new[-1])]  # Initial acceleration. F[0] is used to calculate initial acceleration, instead of prompting user for input.


    for _ in time[1:]:
        i += 1
        dt = dt_array[i-1]
        u = u_new[-1]
        v = v_new[-1]
        a = a_new[-1]
        # Calculating the predictors:
        u_new.append(u + v*dt + a*(1/2-beta)*dt**2)
        v_new.append(v + a*(1-gamma)*dt)
        # Solution of the linear problem:
        a_new.append( const @ (F[i] - K @ u_new[-1] - C @ v_new[-1]))
        # Correcting the predictors:
        u_new[-1] = u_new[-1] + a_new[-1]*beta*dt**2  # this was the missing term in "Calculating the predictors" step.
        v_new[-1] = v_new[-1] + a_new[-1]*gamma*dt  # this was the missing term in "Calculating the predictors" step.

        if i%1000==0:
            print('time step: '+str(i))

    return {'u': np.array(u_new),
            'v': np.array(v_new),
            'a': np.array(a_new)}
# Validation of the MDOF_TD_solver (2DOF example from the pdf in the Reference)
# import matplotlib.pyplot as plt
# sample_T = 10
# dt = 0.28
# time = np.arange(0, sample_T + dt, dt)
# M = np.array([[2,0],[0,1]])
# C = M*0
# K = np.array([[6,-2],[-2,4]])
# KG = False
# N = False
# F = np.array([[0,10]]*len(time))
# u0 = np.array([0]*len(M))
# v0 = np.array([0]*len(M))
# read_dict = MDOF_TD_solver(M=M, C=C, K=K, KG=KG, N=N, F=F, u0=u0, v0=v0, T=sample_T, dt=dt)
# u, v, a = read_dict['u'], read_dict['v'], read_dict['a']
# plt.plot(u)


# THIS FUNCTION IS COPIED TO BUFFETING_TD_FUNC, AND RUN THERE. Otherwise, many variables needed, including functions from buffeting.py
def MDOF_TD_NL_wind_solver(g_node_coor, p_node_coor, R_loc, D_loc, beta_bar, theta_bar, M, C, K, windspeed, u0, v0, T, dt, gamma=1/2, beta=1/4):
    """Multi-degree of freedom time-domain solver, for self-excited + wind forces. The instantaneous angles are calculated \n
    at each time step, due to both turbulence and structure motions, and then instantanous coefficients and forces are calculated. \n
    Reference: Page 19 (of 55) of "Time Domain Methods - ETHZ course material.pdf" (or of https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femII/presentation_05_dynamics_v3.pdf) \n
    M -- Mass matrix \n
    C -- Damping matrix = damp_ratio* 2*np.sqrt(K*M) \n
    K -- Stiffness matrix (already including KG and/or Kse if necessary) \n
    windspeed -- Array with wind speeds with components: [V,u,v,w]. shape:(4,g,t). Time length = len(np.arange(0,T+dt,dt)) \n
    u0 -- Initial displacement (m) \n
    v0 -- Initial velocity (m/s) \n
    T -- Simulation duration (s) \n
    dt -- Time step (s). Should be smaller then eigen period / 40. \n
    gamma -- Newmark parameter. \n
    beta -- Newmark parameter. \n
    [1] https://en.wikipedia.org/wiki/Newmark-beta_method \n
    """

    from scipy import interpolate

    time = np.arange(0, T + dt, dt)
    dt_array = np.array([time[i + 1] - time[i] for i in range(len(time[:-1]))])
    n_dof = len(M)
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1

    R_loc = copy.deepcopy(R_loc)  # Otherwise it would increase during repetitive calls of this function
    D_loc = copy.deepcopy(D_loc)  # Otherwise it would increase during repetitive calls of this function
    girder_N = copy.deepcopy(R_loc[:g_elem_num, 0])  # No girder axial forces
    c_N = copy.deepcopy(R_loc[g_elem_num:, 0])  # No columns axial forces
    alpha = copy.deepcopy(D_loc[:g_node_num, 3])  # No girder nodes torsional rotations

    # Windspeeds:
    w_U_u = windspeed[0]  # wind U+u (mean wind + along wind turbulence)
    w_u = windspeed[1]  # wind u component
    w_v = windspeed[2]  # wind v component
    w_w = windspeed[3]  # wind w component

    # Variables, calculated once
    g_node_L_3D = g_node_L_3D_func(g_node_coor)
    B_diag = np.diag((CS_width, CS_width, CS_width, CS_width ** 2, CS_width ** 2, CS_width ** 2))
    T_LrLwbar = T_LrLw_3_func(g_node_coor, beta_bar, theta_bar)
    T_GsGw = T_GsGw_func(beta_0, theta_0)
    # Creating linear interpolation functions of aerodynamic coefficients, from an extensive grid of possible angles:
    beta_grid = np.arange(-np.pi, np.pi, rad(0.1))  # change the grid discretization here, as desired!
    theta_grid = np.arange(-np.pi/4, np.pi/4, rad(0.1))  # change the grid interval and discretization here, as desired!
    xx, yy = np.meshgrid(beta_grid, theta_grid)
    C_Ci_grid_flat = C_Ci_func(xx.flatten(), yy.flatten(), aero_coef_method, include_Ca)
    C_Ci_grid = C_Ci_grid_flat.reshape((6,len(theta_grid), len(beta_grid)))
    # C_C0_func = interpolate.interp2d(beta_grid, theta_grid, C_Ci_grid[0], kind='linear')  # alternative interpolation, supposedlly slower, but not limited to a rectangular grid.
    C_C0_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1,2)[0], kx=1, ky=1)
    C_C1_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1,2)[1], kx=1, ky=1)
    C_C2_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1,2)[2], kx=1, ky=1)
    C_C3_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1,2)[3], kx=1, ky=1)
    C_C4_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1,2)[4], kx=1, ky=1)
    C_C5_func = interpolate.RectBivariateSpline(beta_grid, theta_grid, np.moveaxis(C_Ci_grid, 1,2)[5], kx=1, ky=1)

    # Start Newmark
    i = 0  # counter.
    dt = dt_array[0]
    u_new = [u0]  # Initial displacement
    v_new = [v0]  # Initial velocity

    #################################################################################################################################################
    # Variables, calculated every time step
    T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
    T_LsGs_6 = T_LsGs_6g_func(g_node_coor, alpha)
    T_GsLs = np.transpose(T_LsGs, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_GsLs_6 = np.transpose(T_LsGs_6, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
    T_GsLw = T_GsLs @ T_LrLwbar
    T_GsLw_6 = np.zeros((g_node_num, 6, 6))
    T_GsLw_6[:, :3, :3] = T_GsLw
    T_GsLw_6[:, 3:, 3:] = T_GsLw
    T_LrGw = T_LsGs @ T_GsGw
    t11 = T_LrGw[:, 0, 0]
    t12 = T_LrGw[:, 0, 1]
    t13 = T_LrGw[:, 0, 2]
    t21 = T_LrGw[:, 1, 0]
    t22 = T_LrGw[:, 1, 1]
    t23 = T_LrGw[:, 1, 2]
    t31 = T_LrGw[:, 2, 0]
    t32 = T_LrGw[:, 2, 1]
    t33 = T_LrGw[:, 2, 2]
    T_LsGs_full_2D_node_matrix = T_LsGs_full_2D_node_matrix_func(g_node_coor, p_node_coor, alpha)
    # Total relative windspeed vector, in local structural Ls (same as Lr) coordinates. See eq. (4-36) from L.D.Zhu thesis. shape: (3,n_nodes,time)
    v0_Ls = T_LsGs_full_2D_node_matrix @ v_new[-1]  # Initial structural speeds
    V_q = t11 * w_U_u[:,i] + t12 * w_v[:,i] + t13 * w_w[:,i]
    V_p = t21 + w_U_u[:,i] + t22 * w_v[:,i] + t23 * w_w[:,i]
    V_h = t31 + w_U_u[:,i] + t32 * w_v[:,i] + t33 * w_w[:,i]
    V_rel_q = V_q - v0_Ls[0:g_node_num*6:6]  # including structural motion. shape:(g_node_num).
    V_rel_p = V_p - v0_Ls[1:g_node_num*6:6]
    V_rel_h = V_h - v0_Ls[2:g_node_num*6:6]
    # Projection of V_Lr in local bridge xy plane (same as qp in L.D.Zhu). See L.D.Zhu eq. (4-44)
    V_rel_qp = np.sqrt(V_rel_q ** 2 + V_rel_p ** 2)  # SRSS of Vq and Vp
    V_rel_tot = np.sqrt(V_rel_q ** 2 + V_rel_p ** 2 + V_rel_h ** 2)
    theta_tilde = np.arccos(V_rel_qp / V_rel_tot)
    beta_tilde_outside_pi = np.arccos(V_rel_p / V_rel_qp)
    beta_tilde = np.array([beta_within_minus_Pi_and_Pi_func(b) for b in beta_tilde_outside_pi])
    T_LrLwtilde_6 = T_LrLw_6_func(beta_tilde, theta_tilde)  # shape:(g,6,6)
    C_Ci_tilde = np.array([C_C0_func.ev(beta_tilde, theta_tilde),  # .ev means "evaluate" the interpolation, at given points
                           C_C1_func.ev(beta_tilde, theta_tilde),
                           C_C2_func.ev(beta_tilde, theta_tilde),
                           C_C3_func.ev(beta_tilde, theta_tilde),
                           C_C4_func.ev(beta_tilde, theta_tilde),
                           C_C5_func.ev(beta_tilde, theta_tilde)])
    F_ad_tilde = 0.5 * rho * np.einsum('n,n,ij,jn->ni', g_node_L_3D, V_rel_tot ** 2, B_diag, C_Ci_tilde, optimize=True)  # in instantaneous local Lw_tilde coordinates
    F_ad_tilde_Ls = np.einsum('nij,nj->ni', T_LrLwtilde_6, F_ad_tilde)  # Local structural
    F_ad_tilde_Gs = np.einsum('nij,nj->ni', T_GsLs_6, F_ad_tilde_Ls)  # Global structural
    F = np.reshape(F_ad_tilde_Gs, (g_node_num * 6))  # reshaping from 'nd' (2D) to '(n*d)' (1D) so it resembles the stiffness matrix shape of (n*d)*(n*d)
    F = np.concatenate((F, np.zeros((p_node_num * 6))), axis=0)  # adding Fb = 0 to all remaining dof at the pontoon g_nodes
    #################################################################################################################################################

    # =========================================================================
    # Constant-Average-Acceleration Method. (see Newmark-Beta method)
    # =========================================================================
    # a & a_new are accelerations at time step i-1 & i respectively.
    # v & v_new are velocities    at time step i-1 & i respectively.
    # u & u_new are position      at time step i-1 & i respectively.

    a_new = [np.linalg.inv(M + C*gamma*dt + K*beta*dt**2) @ (F - K @ u_new[-1] - C @ v_new[-1])]  # Initial acceleration. F[0] is used to calculate initial acceleration, instead of prompting user for input.

    for _ in time[1:]:
        i += 1
        dt = dt_array[i-1]
        u = u_new[-1]
        v = v_new[-1]
        a = a_new[-1]
        # Calculating the predictors:
        u_new.append(u + v*dt + a*(1/2-beta)*dt**2)
        v_new.append(v + a*(1-gamma)*dt)

        # Updating the motion-dependent forces:
        alpha = u_new[-1,3::6]  #todo: confirm

        #################################################################################################################################################
        # Variables, calculated every time step
        T_LsGs = T_LsGs_3g_func(g_node_coor, alpha)
        T_LsGs_6 = T_LsGs_6g_func(g_node_coor, alpha)
        T_GsLs = np.transpose(T_LsGs, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
        T_GsLs_6 = np.transpose(T_LsGs_6, axes=(0, 2, 1))  # (transpose from (0,1,2) to (0,2,1))
        T_GsLw = T_GsLs @ T_LrLwbar
        T_GsLw_6 = np.zeros((g_node_num, 6, 6))
        T_GsLw_6[:, :3, :3] = T_GsLw
        T_GsLw_6[:, 3:, 3:] = T_GsLw
        T_LrGw = T_LsGs @ T_GsGw
        t11 = T_LrGw[:, 0, 0]
        t12 = T_LrGw[:, 0, 1]
        t13 = T_LrGw[:, 0, 2]
        t21 = T_LrGw[:, 1, 0]
        t22 = T_LrGw[:, 1, 1]
        t23 = T_LrGw[:, 1, 2]
        t31 = T_LrGw[:, 2, 0]
        t32 = T_LrGw[:, 2, 1]
        t33 = T_LrGw[:, 2, 2]
        T_LsGs_full_2D_node_matrix = T_LsGs_full_2D_node_matrix_func(g_node_coor, p_node_coor, alpha)
        # Total relative windspeed vector, in local structural Ls (same as Lr) coordinates. See eq. (4-36) from L.D.Zhu thesis. shape: (3,n_nodes,time)
        v0_Ls = T_LsGs_full_2D_node_matrix @ v_new[-1]  # Initial structural speeds
        V_q = t11 * w_U_u[:, i] + t12 * w_v[:, i] + t13 * w_w[:, i]
        V_p = t21 + w_U_u[:, i] + t22 * w_v[:, i] + t23 * w_w[:, i]
        V_h = t31 + w_U_u[:, i] + t32 * w_v[:, i] + t33 * w_w[:, i]
        V_rel_q = V_q - v0_Ls[0:g_node_num * 6:6]  # including structural motion. shape:(g_node_num).
        V_rel_p = V_p - v0_Ls[1:g_node_num * 6:6]
        V_rel_h = V_h - v0_Ls[2:g_node_num * 6:6]
        # Projection of V_Lr in local bridge xy plane (same as qp in L.D.Zhu). See L.D.Zhu eq. (4-44)
        V_rel_qp = np.sqrt(V_rel_q ** 2 + V_rel_p ** 2)  # SRSS of Vq and Vp
        V_rel_tot = np.sqrt(V_rel_q ** 2 + V_rel_p ** 2 + V_rel_h ** 2)
        theta_tilde = np.arccos(V_rel_qp / V_rel_tot)
        beta_tilde_outside_pi = np.arccos(V_rel_p / V_rel_qp)
        beta_tilde = np.array([beta_within_minus_Pi_and_Pi_func(b) for b in beta_tilde_outside_pi])
        T_LrLwtilde_6 = T_LrLw_6_func(beta_tilde, theta_tilde)  # shape:(g,6,6)
        C_Ci_tilde = np.array([C_C0_func.ev(beta_tilde, theta_tilde),  # .ev means "evaluate" the interpolation, at given points
                               C_C1_func.ev(beta_tilde, theta_tilde),
                               C_C2_func.ev(beta_tilde, theta_tilde),
                               C_C3_func.ev(beta_tilde, theta_tilde),
                               C_C4_func.ev(beta_tilde, theta_tilde),
                               C_C5_func.ev(beta_tilde, theta_tilde)])
        F_ad_tilde = 0.5 * rho * np.einsum('n,n,ij,jn->ni', g_node_L_3D, V_rel_tot ** 2, B_diag, C_Ci_tilde, optimize=True)  # in instantaneous local Lw_tilde coordinates
        F_ad_tilde_Ls = np.einsum('nij,nj->ni', T_LrLwtilde_6, F_ad_tilde)  # Local structural
        F_ad_tilde_Gs = np.einsum('nij,nj->ni', T_GsLs_6, F_ad_tilde_Ls)  # Global structural
        F = np.reshape(F_ad_tilde_Gs, (g_node_num * 6))  # reshaping from 'nd' (2D) to '(n*d)' (1D) so it resembles the stiffness matrix shape of (n*d)*(n*d)
        F = np.concatenate((F, np.zeros((p_node_num * 6))), axis=0)  # adding Fb = 0 to all remaining dof at the pontoon g_nodes
        #################################################################################################################################################

        # Solution of the linear problem:
        a_new.append(np.linalg.inv(M + C*gamma*dt + K*beta*dt**2) @ (F[i] - K @ u_new[-1] - C @ v_new[-1]))
        # Correcting the predictors:
        u_new[-1] = u_new[-1] + a_new[-1]*beta*dt**2  # this was the missing term in "Calculating the predictors" step.
        v_new[-1] = v_new[-1] + a_new[-1]*gamma*dt  # this was the missing term in "Calculating the predictors" step.

        if i%1000==0:
            print('time step: '+str(i))

    return {'u': np.array(u_new),
            'v': np.array(v_new),
            'a': np.array(a_new)}




# Careful in using these SDOF solvers. Some successful validations were performed already (parametric excitations),
# but these are not exactly the Newmark method. An explicit (dependent only on previous time step values) solution for
# the a_new was obtained analytically, instead of using "predictors" for u_new and v_new and then "correcting" them.
# Since the already "corrected" implicit (also dependent on the current time step) values of u_new and v_new
# are used to obtain the linear solution of the problem this "alternative method" should be even more accurate.

def SDOF_TD_solver(M, C, K, KG, N, F, u0, v0, T, dt):
    """Single degree of freedom time-domain solver. \n
    M -- Mass \n
    C -- Damping = damp_ratio* 2*np.sqrt(K*M) \n
    K -- Stiffness \n
    KG -- Geometric stiffness (still to be multiplied by N amplitude) \n  todo: Watch out! different in MDOF solver
    N -- Axial force array. positive = compression. length = len( np.arange(0, T+dt, dt) ) \n
    F -- External force array.       length = len( np.arange(0, T+dt, dt) ) \n
    u0 -- Initial displacement (m) \n
    v0 -- Initial velocity (m/s) \n
    T -- Simulation duration (s) \n
    dt -- Time step (s). Should be smaller then eigen period / 40. \n
    [1] https://en.wikipedia.org/wiki/Newmark-beta_method \n
    [2] https://www.wolframalpha.com \n
    """

    time = np.arange(0, T + dt, dt)
    dt_array = np.array([time[i + 1] - time[i] for i in range(len(time[:-1]))])

    # =========================================================================
    # Constant average acceleration method. (see Newmark-Beta method)
    # =========================================================================
    # a & a_new are accelerations at time step i-1 & i respectively.
    # v & v_new are velocities    at time step i-1 & i respectively.
    # u & u_new are position      at time step i-1 & i respectively.

    # To obtain "a_new", paste on wolfram the following: solve for a1, M*a1 + C*(v0+(a0+a1)*t/2) + K*(x0+(v0+(v0+(a0+a1)*t/2))*t/2) = F
    i = 0  # counter
    u_new = [u0]  # Initial displacement
    v_new = [v0]  # Initial velocity
    a_new = [F[0] / M]  # Initial acceleration. todo: wrong, should be: a_new = [(F[0] - C*v0 - (K-KG*N[0])*u0) / M]

    for _ in time[1:]:
        i += 1
        a = a_new[i - 1]
        v = v_new[i - 1]
        u = u_new[i - 1]
        t = dt_array[i - 1]
        a_new.append((-a * t * (2 * C + (K - KG * N[i]) * t) - 4 * (
                    C * v + (K - KG * N[i]) * t * v + (K - KG * N[i]) * u) + 4 * F[i]) / (
                                 t * (2 * C + (K - KG * N[i]) * t) + 4 * M))
        v_new.append(v + (a + a_new[i]) * t / 2)
        u_new.append(u + (v + v_new[i]) * t / 2)

    return {'u': u_new,
            'v': v_new,
            'a': a_new}


def SDOF_TD_NL_damping_solver(M, C, D, K, KG, N, F, u0, v0, T, dt):
    """Single degree of freedom time-domain solver. \n
    M -- Mass \n
    C -- Damping = damp_ratio* 2*np.sqrt(K*M) \n
    D -- Quadratic Damping. Shows up in the Eq. of Motion the term: D * v**2 \n
    K -- Stiffness \n
    KG -- Geometric stiffness (still to be multiplied by N amplitude) \n  #todo: watchout, different in MDOF solver
    N -- Axial force array. positive = compression. length = len( np.arange(0, T+dt, dt) ) \n
    F -- External force array.       length = len( np.arange(0, T+dt, dt) ) \n
    u0 -- Initial displacement (m) \n
    v0 -- Initial velocity (m/s) \n
    T -- Simulation duration (s) \n
    dt -- Time step (s). Should be smaller then eigen period / 40. \n
    [1] https://en.wikipedia.org/wiki/Newmark-beta_method \n
    [2] https://www.wolframalpha.com \n
    """

    time = np.arange(0, T + dt, dt)
    dt_array = np.array([time[i + 1] - time[i] for i in range(len(time[:-1]))])

    # =========================================================================
    # Constant average acceleration method. (see Newmark-Beta method)
    # =========================================================================
    # a & a_new are accelerations at time step i-1 & i respectively.
    # v & v_new are velocities    at time step i-1 & i respectively.
    # u & u_new are position      at time step i-1 & i respectively.

    # To obtain "a_new", paste on MATLAB the following:
    #    syms M a1 C D v0 a0 a1 t K u0 F
    #    eqn = M*a1 + C*(v0+(a0+a1)*t/2) + D*abs(v0)*(v0) + K*(u0+(v0+(v0+(a0+a1)*t/2))*t/2) == F        # Notice that D is using v0 and not v1=v0+(a0+a1)*t/2. This is an approximation. Otherwise the mathematics would be very complicated.
    #    solve(eqn2, a1)
    i = 0  # counter
    u_new = [u0]  # Initial displacement
    v_new = [v0]  # Initial velocity
    a_new = [F[0] / M]  # Initial acceleration

    for _ in time[1:]:
        i += 1
        a = a_new[i - 1]
        v = v_new[i - 1]
        u = u_new[i - 1]
        t = dt_array[i - 1]
        a_new.append(-(C * (v + (a * t) / 2) - F[i] + (K - KG * N[i]) * (
                    u + (t * (2 * v + (a * t) / 2)) / 2) + D * v * abs(v)) / (
                                 ((K - KG * N[i]) * t ** 2) / 4 + (C * t) / 2 + M))
        #       a_new.append(-(4*M + 2*C*t + 4*((C**2*t**2)/4 + (C*(K-KG*N[i])*t**3)/4 + C*M*t + ((K-KG*N[i])**2*t**4)/16 + ((K-KG*N[i])*M*t**2)/2 - (D*v*(K-KG*N[i])*t**3)/2 - D*u*(K-KG*N[i])*t**2 + M**2+D*a*M*t**2 + 2*D*v*M*t + D*F[i]*t**2)**(1/2) + (K-KG*N[i])*t**2 + 4*D*t*v + 2*D*a*t**2)/(2*D*t**2))
        v_new.append(v + (a + a_new[i]) * t / 2)
        u_new.append(u + (v + v_new[i]) * t / 2)

    return {'u': u_new,
            'v': v_new,
            'a': a_new}