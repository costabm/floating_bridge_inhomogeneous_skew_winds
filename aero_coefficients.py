"""
This script collects experimental data_in on drag, lift, moment and axial coefficients and interpolates and extrapolates it

The available data_in from the SOH tests is limited (30 points, 6 betas(uncorrected) * 5 alphas).
Different methods to interpolate and extrapolate were studied in the "aero_coefficients_study_different_extrapolations".

The main conclusion is:
It is reasonable to use a 2-variable (2D) 2nd order polynomial to fit all the data_in points.
Another source of data_in (either data_in from a similar bridge girder, or artificial data_in from the cosine rule) could be
partially used for the parts of the domain far from the existing data_in points, using a weight function.

Other conclusions are:
The data_in in SOH coordinates (betas_uncorrected, alphas) is in a regular grid.
The data_in in Zhu coordinates (betas, thetas) is in an irregular grid.
It is also reasonable to use a 1-variable (1D) 2nd order polynomial to fit strips of the data_in in one direction (alphas),
and then repeating for the other direction (betas), and finally connecting all the points into one surface. However,
some unexpected fit-curvatures appear, given that this method is more sensitive to bad outliers in the
(even more limited) data_in of each strip. On the other hand, it can be more easily tuned locally as desired.
The 1D fitting needs to be done on a regular grid. This can be overcome by 1) using SOH coordinates for all inter- and
extrapolations and changing to Zhu coordinates in the end. Or 2) interpolating the irregular grid data_in into a
pseudo-regular grid (the convex-hull domain will not be a rectangle) and then doing 1D operations on this interpolated
data_in in a similar way.

created: 12/2019
author: Bernardo Costa
email: bernamdc@gmail.com
"""

import sys
import numpy as np
import pandas as pd
from aerodynamic_coefficients.polynomial_fit import cons_poly_fit
from transformations import T_LnwLs_func, theta_yz_bar_func
import copy

def rad(deg):
    return deg * np.pi / 180

def deg(rad):
    return rad * 180 / np.pi

# Converting to L.D. Zhu "beta" and "theta" definition. Points are no longer in a regular grid. Angles are converted to a [-180,180] deg interval
def from_SOH_to_Zhu_angles(betas_uncorrected, alphas):
    betas = np.arctan(np.tan(betas_uncorrected) / np.cos(alphas))  # [Rad]. Correcting the beta values (when alpha != 0). See the "Angles of the skewed wind" document in the _basis folder for explanation.
    thetas = -np.arcsin(np.cos(betas_uncorrected) * np.sin(alphas))  # [Rad]. thetas as in L.D.Zhu definition. See the "Angles of the skewed wind" document in the _basis folder for explanation.
    return betas, thetas

def aero_coef(betas_extrap, thetas_extrap, method, coor_system, constr_fit_degree_list=[3,4,4,4,4,4], constr_fit_2_degree_list=[3,4,4,4,4,4], free_fit_degree_list=[2,2,1,1,3,4]):
    """
    betas: 1D-array
    thetas: 1D-array (same size as betas)

    plot = True or False. Plots and saves pictures of the generated surface for Cd

    method = '2D_fit_free', '2D_fit_cons', 'cos_rule', 'hybrid'.

    coor_system = 'Lnw', 'Ls', 'Lnw&Ls'

    Returns the interpolated / extrapolated values of the aerodynamic coefficients in Local-normal-wind "Lnw" at each given
    coordinate (betas[i], thetas[i]), using the results from the SOH wind tunnel tests. The results are
    in "Local normal wind coordinates" (the same as the Lwbar system from L.D.Zhu when rotated by beta back to non-skew position) or in Ls.
    Regardless, in the process, Local Structural coordinates are used for correctness in symmetry transformations and in constraints.
    """
    betas_extrap = copy.deepcopy(betas_extrap)
    thetas_extrap = copy.deepcopy(thetas_extrap)
    # Data error checks
    if any(abs(thetas_extrap) > rad(90)):
        raise ValueError('At least one theta is outside [-90,90] deg (in radians)')
    if any(abs(betas_extrap) > rad(180)):
        raise ValueError('At least one beta is outside [-180,180] deg (in radians)')
    if len(betas_extrap.flatten()) != len(thetas_extrap.flatten()):
        raise ValueError('Both arrays need to have same size')
    if len(np.shape(betas_extrap)) != 1 or len(np.shape(thetas_extrap)) != 1:
        raise TypeError('Input should be 1D array')

    size = len(betas_extrap)
    try:  # works only when "Run"
        import os
        project_path = os.path.dirname(os.path.abspath(__file__))
    except:  # works when running directly in console
        project_path = sys.path[-2]  # Path of the project directory. To be used in the Python Console! When a console is opened in Pycharm, the current project path should be automatically added to sys.path.


    # Importing input file
    df = pd.read_csv(
        project_path + r'\\aerodynamic_coefficients\\aero_coef_experimental_data.csv')  # raw original values

    # Importing the angles
    betas_SOH = rad(df['beta[deg]'].to_numpy())
    thetas_SOH = rad(df['theta[deg]'].to_numpy())
    alphas_SOH = rad(df['alpha[deg]'].to_numpy())  # torsional rotation of the bridge girder (different from thetas)

    Cx_Ls = df['Cx_Ls'].to_numpy()
    Cy_Ls = df['Cy_Ls'].to_numpy()
    Cz_Ls = df['Cz_Ls'].to_numpy()
    Cxx_Ls = df['Cxx_Ls'].to_numpy()
    Cyy_Ls = df['Cyy_Ls'].to_numpy()
    Czz_Ls = df['Czz_Ls'].to_numpy()

    # Converting all [-180,180] angles into equivalent [0,90] angles. The sign information outside [0,90] is lost and stored manually for each coefficient. Assumes symmetric cross-section.
    Cx_sign, Cy_sign, Cz_sign, Cxx_sign, Cyy_sign, Czz_sign = np.zeros((6, size))
    # Signs for axes in Ls.
    for b in range(size):
        if rad(0) <= betas_extrap[b] <= rad(90):  # all other intervals will be transformations to this one
            Cx_sign[b] = 1
            Cy_sign[b] = 1
            Cz_sign[b] = 1
            Cxx_sign[b] = 1
            Cyy_sign[b] = 1
            Czz_sign[b] = 1
        elif rad(90) < betas_extrap[b] <= rad(180):
            betas_extrap[b] = rad(180) - betas_extrap[b]  # if beta = 110, then becomes 180-110=70
            # the following signs will conserve the fact that beta was in another quadrant.
            Cx_sign[b] = 1
            Cy_sign[b] = -1
            Cz_sign[b] = 1
            Cxx_sign[b] = -1
            Cyy_sign[b] = 1
            Czz_sign[b] = -1
        elif -rad(90) <= betas_extrap[b] < 0:
            betas_extrap[b] = -betas_extrap[b]  # if beta = -60, then becomes 60
            # the following signs will conserve the fact that beta was in another quadrant.
            Cx_sign[b] = -1
            Cy_sign[b] = 1
            Cz_sign[b] = 1
            Cxx_sign[b] = 1
            Cyy_sign[b] = -1
            Czz_sign[b] = -1
        elif -rad(180) <= betas_extrap[b] < -rad(90):
            betas_extrap[b] = rad(180) + betas_extrap[b]  # if beta = -160, then becomes 180+(-160)=20
            # the following signs will conserve the fact that beta was in another quadrant.
            Cx_sign[b] = -1
            Cy_sign[b] = -1
            Cz_sign[b] = 1
            Cxx_sign[b] = -1
            Cyy_sign[b] = -1
            Czz_sign[b] = 1

    # Input data and desired output coordinates (betas and thetas)
    data_in_Cx_Ls = np.array([betas_SOH, thetas_SOH, Cx_Ls])
    data_in_Cy_Ls = np.array([betas_SOH, thetas_SOH, Cy_Ls])
    data_in_Cz_Ls = np.array([betas_SOH, thetas_SOH, Cz_Ls])
    data_in_Cxx_Ls = np.array([betas_SOH, thetas_SOH, Cxx_Ls])
    data_in_Cyy_Ls = np.array([betas_SOH, thetas_SOH, Cyy_Ls])
    data_in_Czz_Ls = np.array([betas_SOH, thetas_SOH, Czz_Ls])

    data_coor_out = np.array([betas_extrap.flatten(), thetas_extrap.flatten()])
    data_bounds = np.array([[0, np.pi / 2], [-np.pi / 2, np.pi / 2]])  # [[beta bounds], [theta bounds]]
    data_bounds_Cy = np.array([[0, np.pi / 2], [-rad(30), rad(30)]])  # [[beta bounds], [theta bounds]]

    # Transforming the coefficients to Local normal wind "Lnw" coordinates, whose axes are defined as:
    # x-axis <=> along-normal-wind (i.e. a "cos-rule-drag"), aligned with the (U+u)*cos(beta) that lies in a 2D plane normal to the bridge girder.
    # y-axis <=> along bridge girder but respecting a M rotation in SOH report where wind is from left and leading edge goes up.
    # z-axis <=> cross product of x and y (i.e. a "cos-rule-lift"), in the same 2D normal plane as x-axis

    if 'Lnw' in coor_system:
        print('WARNING: Avoid using coor_system=Lnw. This has not been carefully looked at')
        theta_yz = theta_yz_bar_func(betas_extrap, thetas_extrap)
        T_LnwLs = T_LnwLs_func(betas=betas_extrap, theta_yz=theta_yz, dim='6x6')

    # 2D polynomial fitting. Note: wrong signs if outside [0,90]
    if method == '2D_fit_free' or method == 'hybrid':
        Cx_Ls_2D_fit_free = cons_poly_fit(data_in_Cx_Ls, data_coor_out, data_bounds, degree=free_fit_degree_list[0], ineq_constraint=False,
                                          other_constraint=False, degree_type='max')[1] * Cx_sign
        Cy_Ls_2D_fit_free = cons_poly_fit(data_in_Cy_Ls, data_coor_out, data_bounds, degree=free_fit_degree_list[1], ineq_constraint=False,
                                          other_constraint=False, degree_type='max')[1] * Cy_sign
        Cz_Ls_2D_fit_free = cons_poly_fit(data_in_Cz_Ls, data_coor_out, data_bounds, degree=free_fit_degree_list[2], ineq_constraint=False,
                                          other_constraint=False, degree_type='max')[1] * Cz_sign
        Cxx_Ls_2D_fit_free = cons_poly_fit(data_in_Cxx_Ls, data_coor_out, data_bounds, degree=free_fit_degree_list[3], ineq_constraint=False,
                                           other_constraint=False, degree_type='max')[1] * Cxx_sign
        Cyy_Ls_2D_fit_free = cons_poly_fit(data_in_Cyy_Ls, data_coor_out, data_bounds, degree=free_fit_degree_list[4], ineq_constraint=False,
                                           other_constraint=False, degree_type='max')[1] * Cyy_sign
        Czz_Ls_2D_fit_free = cons_poly_fit(data_in_Czz_Ls, data_coor_out, data_bounds, degree=free_fit_degree_list[5], ineq_constraint=False,
                                           other_constraint=False, degree_type='max')[1] * Czz_sign
        C_Ci_Ls_2D_fit_free = np.array(
            [Cx_Ls_2D_fit_free, Cy_Ls_2D_fit_free, Cz_Ls_2D_fit_free, Cxx_Ls_2D_fit_free, Cyy_Ls_2D_fit_free,
             Czz_Ls_2D_fit_free])
        if 'Lnw' in coor_system:
            C_Ci_Lnw_2D_fit_free = np.einsum('icd,di->ci', T_LnwLs, C_Ci_Ls_2D_fit_free, optimize=True)

    # Cosine rule: Coefficients(beta,theta) = Coefficients(0,theta)*cos(beta)**2. See LDZhu PhD Thesis, Chapter 6.7. Note: wrong signs if outside [0,90]
    if method in ['cos_rule', 'hybrid', '2D']:
        if method == '2D':
            theta_yz = theta_yz_bar_func(betas_extrap, thetas_extrap)
            # First step: find the C(0,theta) values for all thetas (even if repeated), using the 2D fit on all SOH data_in.
            data_coor_out_beta_0_theta_all = np.array([np.zeros(size), theta_yz.flatten()])
            Cx_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            # Cx_Ls_2D_fit_beta_0_theta_all = cons_poly_fit( data_in_Cx_Ls , data_coor_out_beta_0_theta_all, data_bounds, degree=2, ineq_constraint=False, other_constraint=False, degree_type='total')[1] * Cx_sign
            Cy_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cy_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=free_fit_degree_list[1], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cy_sign
            Cz_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cz_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=free_fit_degree_list[2], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cz_sign
            Cxx_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cxx_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=free_fit_degree_list[3], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cxx_sign
            Cyy_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            Czz_Ls_2D_fit_beta_0_theta_all = np.zeros(size)

            factor = np.sin(thetas_extrap)** 2 + np.cos(betas_extrap)** 2 * np.cos(thetas_extrap)** 2

        elif method == 'cos_rule':
            # First step: find the C(0,theta) values for all thetas (even if repeated), using the 2D fit on all SOH data_in.
            data_coor_out_beta_0_theta_all = np.array([np.zeros(size), thetas_extrap.flatten()])
            Cx_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            # Cx_Ls_2D_fit_beta_0_theta_all = cons_poly_fit( data_in_Cx_Ls , data_coor_out_beta_0_theta_all, data_bounds, degree=2, ineq_constraint=False, other_constraint=False, degree_type='total')[1] * Cx_sign
            Cy_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cy_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=free_fit_degree_list[1], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cy_sign
            Cz_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cz_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=free_fit_degree_list[2], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cz_sign
            Cxx_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cxx_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=free_fit_degree_list[3], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cxx_sign
            Cyy_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            Czz_Ls_2D_fit_beta_0_theta_all = np.zeros(size)

            factor = np.cos(betas_extrap) ** 2

        Cx_Ls_cos = Cx_Ls_2D_fit_beta_0_theta_all * factor
        Cy_Ls_cos = Cy_Ls_2D_fit_beta_0_theta_all * factor
        Cz_Ls_cos = Cz_Ls_2D_fit_beta_0_theta_all * factor
        Cxx_Ls_cos = Cxx_Ls_2D_fit_beta_0_theta_all * factor
        Cyy_Ls_cos = Cyy_Ls_2D_fit_beta_0_theta_all * factor
        Czz_Ls_cos = Czz_Ls_2D_fit_beta_0_theta_all * factor
        C_Ci_Ls_cos = np.array([Cx_Ls_cos, Cy_Ls_cos, Cz_Ls_cos, Cxx_Ls_cos, Cyy_Ls_cos, Czz_Ls_cos])

        if 'Lnw' in coor_system:
            C_Ci_Lnw_cos = np.einsum('icd,di->ci', T_LnwLs, C_Ci_Ls_cos, optimize=True)
        # Note: The cos^2 rule, from L.D.Zhu eq. (6-10), is to be performed on structural xyz coordinates.

    # Hybrid Model: If inside SOH domain = 2D fit, if outside: Cosine. Smooth function between them. Different for Ca.
    if method == 'hybrid':
        # NEW FORMULATION (with cos**2)
        # Cd_hyb = Cd_2D_fit_free * np.cos(betas_extrap)**2 + Cd_cos * np.sin(betas_extrap)**2
        # Cl_hyb = Cl_2D_fit_free * np.cos(betas_extrap)**2 + Cl_cos * np.sin(betas_extrap)**2
        # Cm_hyb = Cm_2D_fit_free * np.cos(betas_extrap)**2 + Cm_cos * np.sin(betas_extrap)**2
        # Ca_hyb = -0.011  * np.sin(betas_extrap)**2  # Alternative: Ca_hyb = -0.011 * np.sin( (betas_extrap/rad(90))**(1/1.5)*rad(90) )**2
        pass

    if method == '2D_fit_cons':
        # Ls coordinates.
        # The constraints are reasoned for a 0-90 deg interval, but applicable to a -180 to 180 deg interval when the symmetry signs (defined above) are also used.
        # Cx
        ineq_constraint_Cx = False  # False or 'positivity' or 'negativity'
        other_constraint_Cx = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end']
        # Cy
        ineq_constraint_Cy = False  # False or 'positivity' or 'negativity'
        other_constraint_Cy = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_start', 'dF/dx0_is_0_at_x0_end_at_x1_middle']   # ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end','dF/dx0_is_0_at_x0_start'] # ] # 'dF/dx0_is_0_at_x0_start']  #  is a cheap way to replace the malfunctioning "positivity" constraint.
        # Cz
        ineq_constraint_Cz = False  # False or 'positivity' or 'negativity'. we could have: dF/dx1_is_positive_at_x0_end', but difficult to implement with little gain.
        other_constraint_Cz = [ 'F_is_0_at_x0_end_at_x1_middle', 'F_is_-2_at_x1_start', 'F_is_2_at_x1_end', 'dF/dx0_is_0_at_x0_start', 'dF/dx0_is_0_at_x0_end']  # 'dF/dx0_is_0_at_x0_start']  # can eventually remove derivative constraint
        # Cxx
        ineq_constraint_Cxx = False  # False or 'positivity' or 'negativity'
        other_constraint_Cxx = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_start', 'dF/dx0_is_0_at_x0_end_at_x1_middle']   # 'dF/dx0_is_0_at_x0_start'
        # Cyy
        ineq_constraint_Cyy = False  # False or 'positivity' or 'negativity'
        other_constraint_Cyy = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end']
        # Czz
        ineq_constraint_Czz = False  # False or 'positivity' or 'negativity'
        other_constraint_Czz = ['F_is_0_at_x0_start', 'F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end']
        Cx_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cx_Ls, data_coor_out, data_bounds, constr_fit_degree_list[0], ineq_constraint_Cx, other_constraint_Cx,
                      degree_type='max')[1] * Cx_sign
        Cy_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cy_Ls, data_coor_out, data_bounds, constr_fit_degree_list[1], ineq_constraint_Cy, other_constraint_Cy,
                      degree_type='max')[1] * Cy_sign  #,, minimize_method='trust-constr', init_guess=[3.71264795e-22, -8.86505000e+00, 4.57056472e+01, -7.39911989e+01, 3.71506016e+01, -6.12248467e-22, -8.75830974e+00,  5.74817737e+01, -1.10425715e+02, 6.17022514e+01, -1.09522498e-21, -2.46382690e+01, 7.14658962e+01, -4.41460857e+01, -2.68154157e+00,  0.00000000e+00, 4.21168758e+01, -1.42475723e+02,  1.30059436e+02, -2.97005883e+01, 0.00000000e+00,  1.44752923e-01, -3.21775938e+01,  9.85035640e+01, -6.64707231e+01])[1] * Cy_sign  # minimize_method='trust-constr'
        Cz_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cz_Ls, data_coor_out, data_bounds, constr_fit_degree_list[2], ineq_constraint_Cz, other_constraint_Cz,
                      degree_type='max', minimize_method='SLSQP')[1] * Cz_sign
        Cxx_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cxx_Ls, data_coor_out, data_bounds, constr_fit_degree_list[3], ineq_constraint_Cxx, other_constraint_Cxx,
                      degree_type='max')[1] * Cxx_sign
        Cyy_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cyy_Ls, data_coor_out, data_bounds, constr_fit_degree_list[4], ineq_constraint_Cyy, other_constraint_Cyy,
                      degree_type='max')[1] * Cyy_sign
        Czz_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Czz_Ls, data_coor_out, data_bounds, constr_fit_degree_list[5], ineq_constraint_Czz, other_constraint_Czz,
                      degree_type='max')[1] * Czz_sign
        C_Ci_Ls_2D_fit_cons = np.array(
            [Cx_Ls_2D_fit_cons, Cy_Ls_2D_fit_cons, Cz_Ls_2D_fit_cons, Cxx_Ls_2D_fit_cons, Cyy_Ls_2D_fit_cons,
             Czz_Ls_2D_fit_cons])

    # if method == '2D_fit_cons_2':
    #     # Ls coordinates.
    #     # The constraints are reasoned for a 0-90 deg interval, but applicable to a -180 to 180 deg interval when the symmetry signs (defined above) are also used.
    #     # Cx
    #     ineq_constraint_Cx = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cx = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']
    #     # Cy
    #     ineq_constraint_Cy = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cy = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']  # ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end','dF/dx0_is_0_at_x0_start'] # ] # 'dF/dx0_is_0_at_x0_start']  #  is a cheap way to replace the malfunctioning "positivity" constraint.
    #     # Cz
    #     ineq_constraint_Cz = False  # False or 'positivity' or 'negativity'. we could have: dF/dx1_is_positive_at_x0_end', but difficult to implement with little gain.
    #     other_constraint_Cz = ['F_is_-2_at_x1_start', 'F_is_2_at_x1_end', 'F_is_0_at_x0_end_at_x1_middle', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']  # 'dF/dx0_is_0_at_x0_start']  # can eventually remove derivative constraint
    #     # Cxx
    #     ineq_constraint_Cxx = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cxx = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']  # 'dF/dx0_is_0_at_x0_start'
    #     # Cyy
    #     ineq_constraint_Cyy = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cyy = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']
    #     # Czz
    #     ineq_constraint_Czz = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Czz = ['F_is_0_at_x0_start', 'F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']
    #
    #     Cx_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cx_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[0], ineq_constraint_Cx, other_constraint_Cx,
    #                   degree_type='max')[1] * Cx_sign
    #     Cy_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cy_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[1], ineq_constraint_Cy, other_constraint_Cy,
    #                   degree_type='max')[1] * Cy_sign  #,, minimize_method='trust-constr', init_guess=[3.71264795e-22, -8.86505000e+00, 4.57056472e+01, -7.39911989e+01, 3.71506016e+01, -6.12248467e-22, -8.75830974e+00,  5.74817737e+01, -1.10425715e+02, 6.17022514e+01, -1.09522498e-21, -2.46382690e+01, 7.14658962e+01, -4.41460857e+01, -2.68154157e+00,  0.00000000e+00, 4.21168758e+01, -1.42475723e+02,  1.30059436e+02, -2.97005883e+01, 0.00000000e+00,  1.44752923e-01, -3.21775938e+01,  9.85035640e+01, -6.64707231e+01])[1] * Cy_sign  # minimize_method='trust-constr'
    #     Cz_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cz_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[2], ineq_constraint_Cz, other_constraint_Cz,
    #                   degree_type='max', minimize_method='SLSQP')[1] * Cz_sign
    #     Cxx_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cxx_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[3], ineq_constraint_Cxx, other_constraint_Cxx,
    #                   degree_type='max')[1] * Cxx_sign
    #     Cyy_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cyy_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[4], ineq_constraint_Cyy, other_constraint_Cyy,
    #                   degree_type='max')[1] * Cyy_sign
    #     Czz_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Czz_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[5], ineq_constraint_Czz, other_constraint_Czz,
    #                   degree_type='max')[1] * Czz_sign
    #     C_Ci_Ls_2D_fit_cons = np.array(
    #         [Cx_Ls_2D_fit_cons, Cy_Ls_2D_fit_cons, Cz_Ls_2D_fit_cons, Cxx_Ls_2D_fit_cons, Cyy_Ls_2D_fit_cons,
    #          Czz_Ls_2D_fit_cons])

    if 'Lnw' in coor_system:
        C_Ci_Lnw_2D_fit_cons = np.einsum('icd,di->ci', T_LnwLs, C_Ci_Ls_2D_fit_cons, optimize=True)

    if coor_system == 'Ls':
        if method == '2D_fit_free':
            return C_Ci_Ls_2D_fit_free
        elif method in ['2D_fit_cons', '2D_fit_cons_2']:
            return C_Ci_Ls_2D_fit_cons
        elif method in ['cos_rule','2D']:
            return C_Ci_Ls_cos
        elif method == 'hybrid':
            return None
    elif coor_system == 'Lnw':
        if method == '2D_fit_free':
            return C_Ci_Lnw_2D_fit_free
        elif method in ['2D_fit_cons', '2D_fit_cons_2']:
            return C_Ci_Lnw_2D_fit_cons
        elif method in ['cos_rule','2D']:
            return C_Ci_Lnw_cos
        elif method == 'hybrid':
            return None
    elif coor_system == 'Lnw&Ls':
        if method == '2D_fit_free':
            return C_Ci_Lnw_2D_fit_free, C_Ci_Ls_2D_fit_free
        elif method in ['2D_fit_cons', '2D_fit_cons_2']:
            return C_Ci_Lnw_2D_fit_cons, C_Ci_Ls_2D_fit_cons
        elif method in ['cos_rule','2D']:
            return C_Ci_Lnw_cos, C_Ci_Ls_cos
        elif method == 'hybrid':
            return None


def aero_coef_derivatives(betas, thetas, method, coor_system):
    betas = copy.deepcopy(betas)
    thetas = copy.deepcopy(thetas)
    # Attention: The Lnw will produce wrong errors since Lnw adapts to all Ci(theta), Ci(theta_prev) and Ci(theta_next) and then the gradient is wrong, and very different for beta -180 and 0 deg,
    # since Lnw is only physical in the [0,90] beta-interval.
    if coor_system == 'Lnw': print('WARNING: coor_system should be "Ls" otherwise the dtheta derivatives will be WRONG!')

    delta_angle = rad(0.001)  # rad. Small variation in beta and theta to calculate the gradients.

    # Attention: if some beta are super close to the boundaries -180,-90,0,90,180 since aero_coef function mirrors f.ex: Ci_before = -0.1 back to 0.1 and then the derivative is wrong and huge (an example gave 10**5 bigger value)! Solution: decrease delta_angle.
    # Correting the error when a beta is exactly at the boundary, by deliberatelly changing problematic betas to very close values.
    angle_correction = delta_angle * 2.1  # rad.
    for i, beta in enumerate(betas):
        if abs(beta - rad(-180)) < delta_angle:
            betas[i] += angle_correction
        if abs(beta - rad(-90)) < delta_angle:
            betas[i] += angle_correction
        if abs(beta - rad(0)) < delta_angle:
            betas[i] += angle_correction
        if abs(beta - rad(90)) < delta_angle:
            betas[i] -= angle_correction
        if abs(beta - rad(180)) < delta_angle:
            betas[i] -= angle_correction

    # Check if previous correction worked:
    if any(abs(rad(180) - abs(betas)) <= delta_angle) or any(abs(rad(90) - abs(betas)) <= delta_angle) or any(
            abs(rad(0) - abs(betas)) <= delta_angle):
        print("WARNING !!! : at least one aero coef derivative could be wrong.")

    # Values "previous" and "next" meaning negative and positive infinitesimal variation of the respective angles.
    beta_prev = betas - delta_angle
    beta_next = betas + delta_angle
    theta_prev = thetas - delta_angle
    theta_next = thetas + delta_angle

    # The centered value of the coefficients
    Cx, Cy, Cz, Cxx, Cyy, Czz = aero_coef(copy.deepcopy(betas), copy.deepcopy(thetas), method=method, coor_system=coor_system)

    # The immediately before and after values of the coefficients
    Cx_beta_prev, Cy_beta_prev, Cz_beta_prev, Cxx_beta_prev, Cyy_beta_prev, Czz_beta_prev = aero_coef(copy.deepcopy(beta_prev), copy.deepcopy(thetas), method=method, coor_system=coor_system)
    Cx_beta_next, Cy_beta_next, Cz_beta_next, Cxx_beta_next, Cyy_beta_next, Czz_beta_next = aero_coef(copy.deepcopy(beta_next), copy.deepcopy(thetas), method=method, coor_system=coor_system)
    Cx_theta_prev, Cy_theta_prev, Cz_theta_prev, Cxx_theta_prev, Cyy_theta_prev, Czz_theta_prev = aero_coef(copy.deepcopy(betas), copy.deepcopy(theta_prev), method=method, coor_system=coor_system)
    Cx_theta_next, Cy_theta_next, Cz_theta_next, Cxx_theta_next, Cyy_theta_next, Czz_theta_next = aero_coef(copy.deepcopy(betas), copy.deepcopy(theta_next), method=method, coor_system=coor_system)

    # Calculating the derivatives = delta(Coef)/delta(angle)
    Cx_dbeta = np.gradient(np.array([Cx_beta_prev, Cx, Cx_beta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cy_dbeta = np.gradient(np.array([Cy_beta_prev, Cy, Cy_beta_next]), axis=0)[1] / delta_angle
    Cz_dbeta = np.gradient(np.array([Cz_beta_prev, Cz, Cz_beta_next]), axis=0)[1] / delta_angle
    Cxx_dbeta = np.gradient(np.array([Cxx_beta_prev, Cxx, Cxx_beta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cyy_dbeta = np.gradient(np.array([Cyy_beta_prev, Cyy, Cyy_beta_next]), axis=0)[1] / delta_angle
    Czz_dbeta = np.gradient(np.array([Czz_beta_prev, Czz, Czz_beta_next]), axis=0)[1] / delta_angle

    Cx_dtheta = np.gradient(np.array([Cx_theta_prev, Cx, Cx_theta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cy_dtheta = np.gradient(np.array([Cy_theta_prev, Cy, Cy_theta_next]), axis=0)[1] / delta_angle
    Cz_dtheta = np.gradient(np.array([Cz_theta_prev, Cz, Cz_theta_next]), axis=0)[1] / delta_angle
    Cxx_dtheta = np.gradient(np.array([Cxx_theta_prev, Cxx, Cxx_theta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cyy_dtheta = np.gradient(np.array([Cyy_theta_prev, Cyy, Cyy_theta_next]), axis=0)[1] / delta_angle
    Czz_dtheta = np.gradient(np.array([Czz_theta_prev, Czz, Czz_theta_next]), axis=0)[1] / delta_angle

    return np.array([[Cx_dbeta, Cy_dbeta, Cz_dbeta, Cxx_dbeta, Cyy_dbeta, Czz_dbeta],
                     [Cx_dtheta, Cy_dtheta, Cz_dtheta, Cxx_dtheta, Cyy_dtheta, Czz_dtheta]])


def from_SOH_to_Zhu_coef_normalization(Cd, Cl, Cm, Ca):
    """
    All in bridge local reference frame "Ls" (not wind reference frame "Lw")
    Converting the experimental coefficients normalized according to SOH, to Zhu's normalization.
    This function could have only 4 lines, but the following is for understanding (de-normalizing and normalizing again)
    """
    rho = 1.25  # [kg/m3]. air density. Not important since it cancels itself.
    U_model = 10  # [m/s]. model wind speed. Not important since it cancels itself.

    # SOH model in the wind tunnel.
    h_model = 0.043  # [m]. model height
    b_model = 0.386  # [m]. model width
    L_model = 2.4  # [m]. model length
    P_model = 62.4 / 80  # [m]. model cross-section perimeter (real scale perimeter divided by scale factor)

    # From SOH's coefficients, to Aerodynamic forces in the model. See SOH report, eq.(C.1)-(C.3)
    Fd_model = 1 / 2 * rho * U_model ** 2 * L_model * h_model * Cd  # [N]
    Fl_model = 1 / 2 * rho * U_model ** 2 * L_model * b_model * Cl  # [N]
    Fm_model = 1 / 2 * rho * U_model ** 2 * L_model * b_model ** 2 * Cm  # [Nm]
    Fa_model = 1 / 2 * rho * U_model ** 2 * L_model * P_model * Ca  # [N]

    # Normalizing according to L.D.Zhu. Note that these are still in bridge ref. frame, not in the wind ref. frame.
    # See You-Lin Xu book, eq. (10.13).
    Cd_Zhu = Fd_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model)  # [-]
    Cl_Zhu = Fl_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model)  # [-]
    Cm_Zhu = Fm_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model ** 2)  # [-]
    Ca_Zhu = Fa_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model)  # [-]

    return Cd_Zhu, Cl_Zhu, Cm_Zhu, Ca_Zhu
