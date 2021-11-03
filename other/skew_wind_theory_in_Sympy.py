"""
bb means beta_bar which means "mean beta"
tb means theta_bar which means "mean theta"

bt means beta_tilde which means "instantaneous beta"
tt means theta_tilde which means "instantaneous theta"

Sometimes b or t are included after other variables, such as Gwb ('bar' == mean Gw) and Gwt ('tilde' == instantaneous Gw)

delta means displacements except for delta_b and delta_t which are delta_beta and delta_theta
_d means derivative. So delta_x is displacement along x-axis and delta_x_d is velocity.

uu, vv, ww is the same as ru, rv, rw, to indicate rotation axis.
"""

from sympy import symbols, Matrix, simplify, cos, sin, tan, acos, asin, atan, sqrt, eye, diag, expand, solve, linsolve, Eq, O, Poly, pi, Dummy, S, apart, sign, diff, Derivative, Abs, pprint, Q, factorial, posify, atan2, sec, expand_trig, collect, factor, invert, BlockMatrix, limit, nsimplify, csc, cot, sec, atan
import numpy as np

def check_sympy_version():
    import sympy
    return sympy.__version__
# assert '1.6' in check_sympy_version(), "Newer versions (1.7.1) of sympy convert some expressions (e.g. simplify(sign(x)/Abs(x))) into Piecewise functions, which are incompatible with the present code"

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

def compare_symbolically(expr1, expr2):
    return simplify((expr1) - (expr2))

def matrix_3dof_to_6dof(M3):
    """Converts a [3x3] to a [6x6] matrix"""
    M6 = Matrix([[M3[0,0], M3[0,1], M3[0,2], 0, 0, 0],
                 [M3[1,0], M3[1,1], M3[1,2], 0, 0, 0],
                 [M3[2,0], M3[2,1], M3[2,2], 0, 0, 0],
                 [0, 0, 0, M3[0,0], M3[0,1], M3[0,2]],
                 [0, 0, 0, M3[1,0], M3[1,1], M3[1,2]],
                 [0, 0, 0, M3[2,0], M3[2,1], M3[2,2]]])
    return M6

def Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree):
    """
    Mathematical formulation reference:
    https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Multivariable_Calculus/3%3A_Topics_in_Partial_Derivatives/Taylor__Polynomials_of_Functions_of_Two_Variables
    See the validation and my online reviews in:
    https://stackoverflow.com/questions/23803320/computing-taylor-series-of-multivariate-function-with-sympy/63850955#63850955
    :param function_expression: Sympy expression of the function
    :param variable_list: list. All variables to be approximated (to be "Taylorized")
    :param evaluation_point: list. Coordinates, where the function will be expressed
    :param degree: int. Total degree of the Taylor polynomial
    :return: Returns a Sympy expression of the Taylor series up to a given degree, of a given multivariate expression, approximated as a multivariate polynomial evaluated at the evaluation_point
    """
    from sympy import factorial, Matrix, prod
    import itertools
    n_var = len(variable_list)
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]  # list of tuples with variables and their evaluation_point coordinates, to later perform substitution
    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))  # list with exponentials of the partial derivatives
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  # Discarding some higher-order terms
    n_terms = len(deriv_orders)
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]  # Individual degree of each partial derivative, of each term
    terms = []
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)  # e.g. df/(dx*dy**2)
        denominator = prod([factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = prod([(Matrix(variable_list) - Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        terms.append(partial_derivatives_at_point / denominator * distances_powered)
    return sum(terms)

# Symbols
rho = symbols('rho', real=True)
B = symbols('B', real=True)
H = symbols('H', real=True)

# To keep equations symbolic use: (_b is used for 'bar' for mean values; _t is used for 'tilde' for instantaneous values, as in L.D. Zhu notation)
U = symbols('U', real=True, positive=True)
u = symbols('u', real=True)
v = symbols('v', real=True)
w = symbols('w', real=True)
bb = symbols(r' bb', real=True, positive=True)  # CHANGE THIS to prove it works for both positive and negative
tb = symbols(r'tb', real=True)
costb = Dummy('costb', real=True, positive=True)  # Trick: cos(tb) as a variable named 'costb', created to enable simplifications that are only possible for cos(tb) > 0. (by enforcing costb -> positive=True).
bt = symbols('bt', real=True)
tt = symbols('tt', real=True)
Cq = symbols('Cq', real=True)
Cp = symbols('Cp', real=True)
Ch = symbols('Ch', real=True)
Cqq = symbols('Cqq', real=True)
Cpp = symbols('Cpp', real=True)
Chh = symbols('Chh', real=True)
Cq_db = symbols('Cq_db', real=True)
Cp_db = symbols('Cp_db', real=True)
Ch_db = symbols('Ch_db', real=True)
Cqq_db = symbols('Cqq_db', real=True)
Cpp_db = symbols('Cpp_db', real=True)
Chh_db = symbols('Chh_db', real=True)
Cq_dt = symbols('Cq_dt', real=True)
Cp_dt = symbols('Cp_dt', real=True)
Ch_dt = symbols('Ch_dt', real=True)
Cqq_dt = symbols('Cqq_dt', real=True)
Cpp_dt = symbols('Cpp_dt', real=True)
Chh_dt = symbols('Chh_dt', real=True)
Cu = symbols('Cu', real=True)
Cv = symbols('Cv', real=True)
Cw = symbols('Cw', real=True)
Cuu = symbols('Cuu', real=True)
Cvv = symbols('Cvv', real=True)
Cww = symbols('Cww', real=True)
Cu_db = symbols('Cu_db', real=True)
Cv_db = symbols('Cv_db', real=True)
Cw_db = symbols('Cw_db', real=True)
Cuu_db = symbols('Cuu_db', real=True)
Cvv_db = symbols('Cvv_db', real=True)
Cww_db = symbols('Cww_db', real=True)
Cu_dt = symbols('Cu_dt', real=True)
Cv_dt = symbols('Cv_dt', real=True)
Cw_dt = symbols('Cw_dt', real=True)
Cuu_dt = symbols('Cuu_dt', real=True)
Cvv_dt = symbols('Cvv_dt', real=True)
Cww_dt = symbols('Cww_dt', real=True)
Cx = symbols('Cx', real=True)
Cy = symbols('Cy', real=True)
Cz = symbols('Cz', real=True)
Crx = symbols('Crx', real=True)
Cry = symbols('Cry', real=True)
Crz = symbols('Crz', real=True)
Cx_db = symbols('Cx_db', real=True)
Cy_db = symbols('Cy_db', real=True)
Cz_db = symbols('Cz_db', real=True)
Crx_db = symbols('Crx_db', real=True)
Cry_db = symbols('Cry_db', real=True)
Crz_db = symbols('Crz_db', real=True)
Cx_dt = symbols('Cx_dt', real=True)
Cy_dt = symbols('Cy_dt', real=True)
Cz_dt = symbols('Cz_dt', real=True)
Crx_dt = symbols('Crx_dt', real=True)
Cry_dt = symbols('Cry_dt', real=True)
Crz_dt = symbols('Crz_dt', real=True)
Cy_dtn = symbols("Cy_dtn", real=True)
Cz_dtn = symbols("Cz_dtn", real=True)
Crx_dtn = symbols("Crx_dtn", real=True)
delta_Xu = symbols('delta_Xu', real=True)
delta_Yv = symbols('delta_Yv', real=True)
delta_Zw = symbols('delta_Zw', real=True)
delta_rXu = symbols('delta_rXu', real=True)
delta_rYv = symbols('delta_rYv', real=True)
delta_rZw = symbols('delta_rZw', real=True)
delta_Xu_d = symbols('delta_Xu_d', real=True)
delta_Yv_d = symbols('delta_Yv_d', real=True)
delta_Zw_d = symbols('delta_Zw_d', real=True)
delta_rXu_d = symbols('delta_rXu_d', real=True)
delta_rYv_d = symbols('delta_rYv_d', real=True)
delta_rZw_d = symbols('delta_rZw_d', real=True)
delta_x = symbols('delta_x', real=True)
delta_y = symbols('delta_y', real=True)
delta_z = symbols('delta_z', real=True)
delta_rx = symbols('delta_rx', real=True)  # alpha, rx rotation.
delta_ry = symbols('delta_ry', real=True)
delta_rz = symbols('delta_rz', real=True)
delta_x_d = symbols('delta_x_d', real=True)
delta_y_d = symbols('delta_y_d', real=True)
delta_z_d = symbols('delta_z_d', real=True)
delta_rx_d = symbols('delta_rx_d', real=True)  # alpha, rx rotation.
delta_ry_d = symbols('delta_ry_d', real=True)
delta_rz_d = symbols('delta_rz_d', real=True)
Un = symbols('Un', real=True, positive=True) # Projection of U onto the yz-plane
un = symbols('un', real=True) # Projection of u onto the yz-plane
vn = symbols('vn', real=True)
wn = symbols('wn', real=True) # Component perpendicular to un, in the yz-plane (it is not the projection of w)
ux = symbols('un', real=True) # (Vx-Ux)
vy = symbols('vn', real=True) # (Vy-Uy)
wz = symbols('wn', real=True) # (Vz-Uz)
tbn = symbols('tbn', real=True)  # theta_bar normal (in the yz-plane)
Cd = symbols('Cd', real=True)
Cl = symbols('Cl', real=True)
Cm = symbols('Cm', real=True)
Cd_dtn = symbols('Cd_dtn', real=True)
Cl_dtn = symbols('Cl_dtn', real=True)
Cm_dtn = symbols('Cm_dtn', real=True)
Ca = symbols('Ca', real=True)
delta_D = symbols('delta_D', real=True)
delta_A = symbols('delta_A', real=True)
delta_L = symbols('delta_L', real=True)
delta_rD = symbols('delta_rD', real=True)
delta_rA = symbols('delta_rA', real=True)
delta_rL = symbols('delta_rL', real=True)
delta_D_d = symbols('delta_D_d', real=True)
delta_A_d = symbols('delta_A_d', real=True)
delta_L_d = symbols('delta_L_d', real=True)
delta_rD_d = symbols('delta_rD_d', real=True)
delta_rA_d = symbols('delta_rA_d', real=True)
delta_rL_d = symbols('delta_rL_d', real=True)

signcosbb = "any"   #  [1,-1,"any"] TODO: CHOOSE HERE THE DESIRED SIGN! Relevant for the normal wind buffeting theory with self-excited forces

if signcosbb == 1:
    cosbb = symbols('cosbb', real=True, positive=True)
elif signcosbb == -1:
    cosbb = symbols('cosbb', real=True, negative=True)
else:
    assert signcosbb == 'any'
    cosbb = symbols('cosbb', real=True)

def R_x(alpha):
    return Matrix([[1, 0, 0],
                   [0, cos(alpha), -sin(alpha)],
                   [0, sin(alpha), cos(alpha)]])

def R_y(alpha):
    return Matrix([[cos(alpha), 0, sin(alpha)],
                   [0, 1, 0],
                   [-sin(alpha), 0, cos(alpha)]])

def R_z(alpha):
    return Matrix([[cos(alpha), -sin(alpha), 0],
                   [sin(alpha), cos(alpha), 0],
                   [0, 0, 1]])

T_LsGwb = ( R_y(tb) @ R_z(-bb-pi/2) ).T
T_LsGwb_6 = matrix_3dof_to_6dof(T_LsGwb)

Cqph = Matrix([Cq, Cp, Ch, Cqq, Cpp, Chh])
Cqph_db = Matrix([Cq_db, Cp_db, Ch_db, Cqq_db, Cpp_db, Chh_db])
Cqph_dt = Matrix([Cq_dt, Cp_dt, Ch_dt, Cqq_dt, Cpp_dt, Chh_dt])

Cuvw = Matrix([Cu, Cv, Cw, Cuu, Cvv, Cww])
Cuvw_db = Matrix([Cu_db, Cv_db, Cw_db, Cuu_db, Cvv_db, Cww_db])
Cuvw_dt = Matrix([Cu_dt, Cv_dt, Cw_dt, Cuu_dt, Cvv_dt, Cww_dt])

Cxyz = Matrix([Cx, Cy, Cz, Crx, Cry, Crz])
Cxyz_db = Matrix([Cx_db, Cy_db, Cz_db, Crx_db, Cry_db, Crz_db])
Cxyz_dt = Matrix([Cx_dt, Cy_dt, Cz_dt, Crx_dt, Cry_dt, Crz_dt])

# Finding the relation between Cxyz and each of its partial derivatives to the Cuvw counterparts
T_GwbLs_6 = T_LsGwb_6.T
Cuvw_ECxyz = T_GwbLs_6 @ Cxyz
Cuvw_db_ECxyz = diff(T_GwbLs_6, bb) @ Cxyz + T_GwbLs_6 @ Cxyz_db
Cuvw_dt_ECxyz = diff(T_GwbLs_6, tb) @ Cxyz + T_GwbLs_6 @ Cxyz_dt
Cxyz_ECuvw = T_LsGwb_6 @ Cuvw
Cxyz_db_ECuvw = diff(T_LsGwb_6, bb) @ Cuvw + T_LsGwb_6 @ Cuvw_db
Cxyz_dt_ECuvw = diff(T_LsGwb_6, tb) @ Cuvw + T_LsGwb_6 @ Cuvw_dt
list_Cuvw_to_Cxyz = list(zip(Cuvw,Cuvw_ECxyz))
list_Cuvw_db_to_Cxyz = list(zip(Cuvw_db,Cuvw_db_ECxyz))
list_Cuvw_dt_to_Cxyz = list(zip(Cuvw_dt,Cuvw_db_ECxyz))
list_Cxyz_to_Cuvw = list(zip(Cxyz,Cxyz_ECuvw))
list_Cxyz_db_to_Cuvw = list(zip(Cxyz_db,Cxyz_db_ECuvw))
list_Cxyz_dt_to_Cuvw = list(zip(Cxyz_dt,Cxyz_dt_ECuvw))

delta_Ls = Matrix([delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz])
delta_Ls_d = Matrix([delta_x_d, delta_y_d, delta_z_d, delta_rx_d, delta_ry_d, delta_rz_d])
delta_Gw = Matrix([delta_Xu, delta_Yv, delta_Zw, delta_rXu, delta_rYv, delta_rZw])
delta_Gw_d = Matrix([delta_Xu_d, delta_Yv_d, delta_Zw_d, delta_rXu_d, delta_rYv_d, delta_rZw_d])
delta_Lnw = Matrix([delta_D, delta_A, delta_L, delta_rD, delta_rA, delta_rL])
delta_Lnw_d = Matrix([delta_D_d, delta_A_d, delta_L_d, delta_rD_d, delta_rA_d, delta_rL_d])

def replace_sign(expression):
    return expression.subs(sign(cos(bb)),cos(bb)/Abs(cos(bb))).subs(sign(sin(bb)*cos(tb)),sin(bb)*cos(tb)/Abs(sin(bb)*cos(tb))).subs(sign(sin(bb)*cos(bb)*cos(tb)/Abs(cos(bb))), (sin(bb)*cos(bb)*cos(tb)/Abs(cos(bb)))/Abs((sin(bb)*cos(bb)*cos(tb)/Abs(cos(bb))))).subs(sign(cos(bb)*cos(tb)), (cos(bb)*cos(tb))/Abs(cos(bb)*cos(tb)))

def testing_and_comparing_numerical_random_values(expression_1, expression_2, n_tests=10, rel_tol=1e-6, abs_tol=1e-10):
    import random
    from math import isclose
    all_symbols_list_1 = list(expression_1.free_symbols)
    all_symbols_list_2 = list(expression_2.free_symbols)
    C_symbols_list_1 = sorted([str(x) for x in all_symbols_list_1 if "C" in str(x)])
    C_symbols_list_2 = sorted([str(x) for x in all_symbols_list_2 if "C" in str(x)])
    delta_symbols_list_1 = sorted([str(x) for x in all_symbols_list_1 if "delta" in str(x)])
    delta_symbols_list_2 = sorted([str(x) for x in all_symbols_list_2 if "delta" in str(x)])
    assert (all(elem in delta_symbols_list_1 for elem in delta_symbols_list_2) or all(elem in delta_symbols_list_2 for elem in delta_symbols_list_1)), "Transform the structural motions of both expressions to the same system before using this function, OR UPGRADE THIS FUNCTION TO WORK"
    for i in range(n_tests):
        B_test = random.uniform(0,50)
        H_test = random.uniform(0,5)
        U_test = random.uniform(20,40)
        # bb_test = random.uniform(-pi,pi)
        if signcosbb == 1:
            bb_test = random.uniform(-pi/2,pi/2)
        elif signcosbb == -1:
            bb_test = random.choice([random.uniform(-pi,-pi/2),random.uniform(pi/2,pi)])
        elif signcosbb == 'any':
            bb_test = random.uniform(-pi, pi)
            # bb_test = random.uniform(rad(89.9), rad(90.1))  # test. delete
        tb_test = random.uniform(-pi/2,pi/2)
        rho_test = 0.125

        Cx_test = random.uniform(-30,30)
        Cx_db_test = random.uniform(-30,30)
        Cx_dt_test = random.uniform(-30,30)
        Cy_test = random.uniform(-30,30)
        Cy_db_test = random.uniform(-30,30)
        Cy_dt_test = random.uniform(-30,30)
        Cz_test = random.uniform(-30,30)
        Cz_db_test = random.uniform(-30,30)
        Cz_dt_test = random.uniform(-30,30)
        Crx_test = random.uniform(-30,30)
        Crx_db_test = random.uniform(-30,30)
        Crx_dt_test = random.uniform(-30,30)
        Cry_test = random.uniform(-30,30)
        Cry_db_test = random.uniform(-30,30)
        Cry_dt_test = random.uniform(-30,30)
        Crz_test = random.uniform(-30,30)
        Crz_db_test = random.uniform(-30,30)
        Crz_dt_test = random.uniform(-30,30)

        T_GwbLs_test = (T_LsGwb.T).subs({bb:bb_test, tb:tb_test}).evalf()
        T_GwbLs_6_test = matrix_3dof_to_6dof(T_GwbLs_test)

        Cuvw_test = Cuvw_ECxyz.subs({bb:bb_test, tb:tb_test}).subs([(Cx, Cx_test), (Cy, Cy_test), (Cz, Cz_test), (Crx, Crx_test), (Cry, Cry_test), (Crz, Crz_test),
                                      (Cx_db, Cx_db_test), (Cy_db, Cy_db_test), (Cz_db, Cz_db_test), (Crx_db, Crx_db_test), (Cry_db, Cry_db_test), (Crz_db, Crz_db_test),
                                      (Cx_dt, Cx_dt_test), (Cy_dt, Cy_dt_test), (Cz_dt, Cz_dt_test), (Crx_dt, Crx_dt_test), (Cry_dt, Cry_dt_test), (Crz_dt, Crz_dt_test)]).evalf()
        Cuvw_db_test = Cuvw_db_ECxyz.subs({bb:bb_test, tb:tb_test}).subs([(Cx, Cx_test), (Cy, Cy_test), (Cz, Cz_test), (Crx, Crx_test), (Cry, Cry_test), (Crz, Crz_test),
                                      (Cx_db, Cx_db_test), (Cy_db, Cy_db_test), (Cz_db, Cz_db_test), (Crx_db, Crx_db_test), (Cry_db, Cry_db_test), (Crz_db, Crz_db_test),
                                      (Cx_dt, Cx_dt_test), (Cy_dt, Cy_dt_test), (Cz_dt, Cz_dt_test), (Crx_dt, Crx_dt_test), (Cry_dt, Cry_dt_test), (Crz_dt, Crz_dt_test)]).evalf()
        Cuvw_dt_test = Cuvw_dt_ECxyz.subs({bb:bb_test, tb:tb_test}).subs([(Cx, Cx_test), (Cy, Cy_test), (Cz, Cz_test), (Crx, Crx_test), (Cry, Cry_test), (Crz, Crz_test),
                                      (Cx_db, Cx_db_test), (Cy_db, Cy_db_test), (Cz_db, Cz_db_test), (Crx_db, Crx_db_test), (Cry_db, Cry_db_test), (Crz_db, Crz_db_test),
                                      (Cx_dt, Cx_dt_test), (Cy_dt, Cy_dt_test), (Cz_dt, Cz_dt_test), (Crx_dt, Crx_dt_test), (Cry_dt, Cry_dt_test), (Crz_dt, Crz_dt_test)]).evalf()

        Cu_test = Cuvw_test[0]
        Cu_db_test = Cuvw_db_test[0]
        Cu_dt_test = Cuvw_dt_test[0]
        Cv_test =  Cuvw_test[1]
        Cv_db_test = Cuvw_db_test[1]
        Cv_dt_test = Cuvw_dt_test[1]
        Cw_test =  Cuvw_test[2]
        Cw_db_test = Cuvw_db_test[2]
        Cw_dt_test = Cuvw_dt_test[2]
        Cuu_test = Cuvw_test[3]
        Cuu_db_test = Cuvw_db_test[3]
        Cuu_dt_test = Cuvw_dt_test[3]
        Cvv_test = Cuvw_test[4]
        Cvv_db_test = Cuvw_db_test[4]
        Cvv_dt_test = Cuvw_dt_test[4]
        Cww_test = Cuvw_test[5]
        Cww_db_test = Cuvw_db_test[5]
        Cww_dt_test = Cuvw_dt_test[5]

        U_Ls_test = T_GwbLs_test.T @ Matrix([U_test,0,0])
        U_y_test = U_Ls_test[1]
        U_z_test = U_Ls_test[2]
        U_yz_test = sqrt(U_y_test ** 2 + U_z_test ** 2)
        tbn_test = simplify(asin(U_z_test / U_yz_test))
        T_LsLnwb_test = (R_y(tbn_test) * R_z(-pi / 2 * sign(cos(bb_test)))).T
        T_LsLnwb_6_test = matrix_3dof_to_6dof(T_LsLnwb_test)

        Cy_dtn_test = random.uniform(-30,30)
        Cz_dtn_test = random.uniform(-30,30)
        Crx_dtn_test = random.uniform(-30,30)

        Cd_test, _, Cl_test, _, Cm_test, _ =  T_LsLnwb_6_test.T @ Matrix([0, Cy_test, Cz_test, Crx_test, 0, 0])
        Cd_dtn_test, _, Cl_dtn_test, _, Cm_dtn_test, _ = T_LsLnwb_6_test.T @ Matrix([0, Cy_dtn_test, Cz_dtn_test, Crx_dtn_test, 0, 0])

        Ca_test = random.uniform(-30,30)

        u_test = random.uniform(-10,10)
        v_test = random.uniform(-10,10)
        w_test = random.uniform(-10,10)

        T_LsGwb_test = (R_y(tb_test) @ R_z(-bb_test - pi / 2)).T
        T_LnwbGwb_test = T_LsLnwb_test.T @ T_LsGwb_test

        un_test, vn_test, wn_test = T_LnwbGwb_test @ Matrix([u_test,v_test,w_test])
        delta_rXu_test = random.uniform(-rad(1),rad(1))
        delta_rYv_test = random.uniform(-rad(1),rad(1))
        delta_rZw_test = random.uniform(-rad(1),rad(1))
        delta_Xu_d_test = random.uniform(-2,2)
        delta_Yv_d_test = random.uniform(-2,2)
        delta_Zw_d_test = random.uniform(-2,2)
        delta_rx_test = random.uniform(-rad(1),rad(1))
        delta_ry_test = random.uniform(-rad(1),rad(1))
        delta_rz_test = random.uniform(-rad(1),rad(1))
        delta_x_d_test = random.uniform(-2,2)
        delta_y_d_test = random.uniform(-2,2)
        delta_z_d_test = random.uniform(-2,2)
        delta_rD_test = random.uniform(-rad(1),rad(1))
        delta_rA_test = random.uniform(-rad(1),rad(1))
        delta_rL_test = random.uniform(-rad(1),rad(1))
        delta_D_d_test = random.uniform(-2,2)
        delta_A_d_test = random.uniform(-2,2)
        delta_L_d_test = random.uniform(-2,2)

        result_1 = expression_1.subs([(un,un_test),(vn,vn_test),(wn,wn_test),(u,u_test),(v,v_test),(w,w_test),(B,B_test),(H,H_test),(U,U_test),(bb,bb_test),(tb,tb_test),(rho,rho_test),(Cd,Cd_test),(Cl,Cl_test),(Cm,Cm_test),(Cd_dtn,Cd_dtn_test),(Cl_dtn,Cl_dtn_test),(Cm_dtn,Cm_dtn_test),(Ca, Ca_test),
                                      (Cx, Cx_test), (Cy, Cy_test), (Cz, Cz_test), (Crx, Crx_test), (Cry, Cry_test), (Crz, Crz_test),
                                      (Cx_db, Cx_db_test), (Cy_db, Cy_db_test), (Cz_db, Cz_db_test), (Crx_db, Crx_db_test), (Cry_db, Cry_db_test), (Crz_db, Crz_db_test),
                                      (Cx_dt, Cx_dt_test), (Cy_dt, Cy_dt_test), (Cz_dt, Cz_dt_test), (Crx_dt, Crx_dt_test), (Cry_dt, Cry_dt_test), (Crz_dt, Crz_dt_test),
                                      (Cy_dtn, Cy_dtn_test), (Cz_dtn, Cz_dtn_test), (Crx_dtn, Crx_dtn_test),
                                      (Cu, Cu_test), (Cv, Cv_test), (Cw, Cw_test), (Cuu, Cuu_test), (Cvv, Cvv_test), (Cww, Cww_test),
                                      (Cu_db, Cu_db_test), (Cv_db, Cv_db_test), (Cw_db, Cw_db_test), (Cuu_db, Cuu_db_test), (Cvv_db, Cvv_db_test), (Cww_db, Cww_db_test),
                                      (Cu_dt, Cu_dt_test), (Cv_dt, Cv_dt_test), (Cw_dt, Cw_dt_test), (Cuu_dt, Cuu_dt_test), (Cvv_dt, Cvv_dt_test), (Cww_dt, Cww_dt_test),
                                      (delta_rXu,delta_rXu_test), (delta_rYv, delta_rYv_test), (delta_rZw, delta_rZw_test), (delta_Xu_d, delta_Xu_d_test), (delta_Yv_d,delta_Yv_d_test),(delta_Zw_d,delta_Zw_d_test),
                                      (delta_rx, delta_rx_test), (delta_ry, delta_ry_test), (delta_rz, delta_rz_test), (delta_x_d, delta_x_d_test), (delta_y_d, delta_y_d_test), (delta_z_d, delta_z_d_test),
                                      (un,un_test),(delta_rD, delta_rD_test), (delta_rA, delta_rA_test), (delta_rL, delta_rL_test), (delta_D_d, delta_D_d_test), (delta_A_d, delta_A_d_test), (delta_L_d, delta_L_d_test)]).evalf()
        result_2 = expression_2.subs([(un,un_test),(vn,vn_test),(wn,wn_test),(u,u_test),(v,v_test),(w,w_test),(B,B_test),(H,H_test),(U,U_test),(bb,bb_test),(tb,tb_test),(rho,rho_test),(Cd,Cd_test),(Cl,Cl_test),(Cm,Cm_test),(Cd_dtn,Cd_dtn_test),(Cl_dtn,Cl_dtn_test),(Cm_dtn,Cm_dtn_test),(Ca, Ca_test),
                                      (Cx, Cx_test), (Cy, Cy_test), (Cz, Cz_test), (Crx, Crx_test), (Cry, Cry_test), (Crz, Crz_test),
                                      (Cx_db, Cx_db_test), (Cy_db, Cy_db_test), (Cz_db, Cz_db_test), (Crx_db, Crx_db_test), (Cry_db, Cry_db_test), (Crz_db, Crz_db_test),
                                      (Cx_dt, Cx_dt_test), (Cy_dt, Cy_dt_test), (Cz_dt, Cz_dt_test), (Crx_dt, Crx_dt_test), (Cry_dt, Cry_dt_test), (Crz_dt, Crz_dt_test),
                                      (Cu, Cu_test), (Cv, Cv_test), (Cw, Cw_test), (Cuu, Cuu_test), (Cvv, Cvv_test), (Cww, Cww_test),
                                      (Cu_db, Cu_db_test), (Cv_db, Cv_db_test), (Cw_db, Cw_db_test), (Cuu_db, Cuu_db_test), (Cvv_db, Cvv_db_test), (Cww_db, Cww_db_test),
                                      (Cu_dt, Cu_dt_test), (Cv_dt, Cv_dt_test), (Cw_dt, Cw_dt_test), (Cuu_dt, Cuu_dt_test), (Cvv_dt, Cvv_dt_test), (Cww_dt, Cww_dt_test),
                                      (Cy_dtn, Cy_dtn_test), (Cz_dtn, Cz_dtn_test), (Crx_dtn, Crx_dtn_test),
                                      (delta_rXu,delta_rXu_test), (delta_rYv, delta_rYv_test), (delta_rZw, delta_rZw_test), (delta_Xu_d, delta_Xu_d_test), (delta_Yv_d,delta_Yv_d_test),(delta_Zw_d,delta_Zw_d_test),
                                      (delta_rx, delta_rx_test), (delta_ry, delta_ry_test), (delta_rz, delta_rz_test), (delta_x_d, delta_x_d_test), (delta_y_d, delta_y_d_test), (delta_z_d, delta_z_d_test),
                                      (delta_rD, delta_rD_test), (delta_rA, delta_rA_test), (delta_rL, delta_rL_test), (delta_D_d, delta_D_d_test), (delta_A_d, delta_A_d_test), (delta_L_d, delta_L_d_test)]).evalf()
        try:  # if it is a matrix (or vector)
            return_error = sum(result_1)  # needed, otherwise the operation below might return error even though we are inside a "try:" ... ...
            return_error = sum(result_2)  # needed, otherwise the operation below might return error even though we are inside a "try:" ... ...
            result_1 = result_1._mat  # flatten
            result_2 = result_2._mat  # flatten
            assert len(result_1) == len(result_2), "The two expressions have different shapes"
            print('1st expression predicts: ' + str(result_1))
            print('2nd expression predicts: ' + str(result_2))
            print('Similarity is... '+str(all([isclose(result_1[i], result_2[i], rel_tol=rel_tol, abs_tol=abs_tol) for i in range(len(result_1))])))
        except TypeError:  # it is just a scalar
            print('1st expression predicts: ' + str(result_1))
            print('2nd expression predicts: ' + str(result_2))
            print('Similarity is... '+str(isclose(result_1, result_2, rel_tol=rel_tol, abs_tol=abs_tol)))
    return None

# SIMPLIFICATION IN WOLFRAM MATHEMATICA
def wolfram_mathematica_equation_simplification(expression, form_try='mathml', n_tests=5):
    """
    :param expression:
    :param form_try: Try between: "mathml", "traditional" and "latex". None is perfect.
    :return:
    """
    # Expression to be simplified in Mathematica:
    from sympy import mathml, factor, latex
    from sympy.parsing.sympy_parser import parse_expr
    import random
    symbols_list = list(expression.free_symbols)
    coefficients_in = []
    if any([s in [Cy,Cz,Crx,Cy_dtn,Cz_dtn,Crx_dtn] for s in symbols_list]):
        coefficients_in.append('xyz')
    if any([s in [Cd,Cl,Cm,Cd_dtn,Cl_dtn,Cm_dtn,Ca] for s in symbols_list]):
        coefficients_in.append('dal')
    assert len(coefficients_in) <= 1, "Make sure there are not Cdal and Cxyz coefficients at the same time!"
    if coefficients_in:
        coefficients_in = coefficients_in[0]

    # To avoid problems in Mathematica and the mathml (or latex) expression the variables need to be one lower case letter
    B_M = symbols(r'c')  # closest letter to B, but m was already taken to Cm.
    H_M = symbols(r'h')
    U_M = symbols(r's')  # speed
    bb_M = symbols(r'b')
    tb_M = symbols(r't')
    rho_M = symbols(r'r')

    Cy_M = symbols(r'd')
    Cz_M = symbols(r'l')
    Crx_M = symbols(r'm')
    Cy_dtn_M = symbols(r'e')
    Cz_dtn_M = symbols(r'k')  # closest letter to l, but m was already taken to Cm.
    Crx_dtn_M = symbols(r'n')

    Cd_M = symbols(r'd')
    Cl_M = symbols(r'l')
    Cm_M = symbols(r'm')
    Cd_dtn_M = symbols(r'e')
    Cl_dtn_M = symbols(r'k')  # closest letter to l, but m was already taken to Cm.
    Cm_dtn_M = symbols(r'n')
    Ca_M = symbols(r'z')

    if any(x in symbols_list for x in list(delta_Ls)+list(delta_Ls_d)):
        assert all(x not in symbols_list for x in list(delta_Gw)+list(delta_Gw_d)+list(delta_Lnw)+list(delta_Lnw_d)), 'Make sure all motion-related variables are in 1 system only'
        variables_expressed_in = 'Ls'
        delta_rx_M = symbols(r'a')
        delta_ry_M = symbols(r'f')
        delta_rz_M = symbols(r'g')
        delta_x_d_M = symbols(r'i')
        delta_y_d_M = symbols(r'j')
        delta_z_d_M = symbols(r'o')
        expression_to_M = expression.subs(
            [(B, B_M), (H, H_M), (U, U_M), (bb, bb_M), (tb, tb_M), (rho, rho_M), (Cd, Cd_M), (Cl, Cl_M), (Cm, Cm_M), (Cd_dtn, Cd_dtn_M), (Cl_dtn, Cl_dtn_M), (Cm_dtn, Cm_dtn_M), (Ca, Ca_M),
             (Cy, Cy_M), (Cz, Cz_M), (Crx, Crx_M), (Cy_dtn, Cy_dtn_M), (Cz_dtn, Cz_dtn_M), (Crx_dtn, Crx_dtn_M),
             (delta_rx, delta_rx_M), (delta_ry, delta_ry_M), (delta_rz, delta_rz_M), (delta_x_d, delta_x_d_M), (delta_y_d, delta_y_d_M), (delta_z_d, delta_z_d_M)])
    elif any(x in symbols_list for x in list(delta_Gw)+list(delta_Gw_d)):
        assert all(x not in symbols_list for x in list(delta_Ls)+list(delta_Ls_d)+list(delta_Lnw)+list(delta_Lnw_d)), 'Make sure all motion-related variables are in 1 system only'
        variables_expressed_in = 'Gw'
        delta_rXu_M = symbols(r'a')
        delta_rYv_M = symbols(r'f')
        delta_rZw_M = symbols(r'g')
        delta_Xu_d_M = symbols(r'i')
        delta_Yv_d_M = symbols(r'j')
        delta_Zw_d_M = symbols(r'o')
        expression_to_M = expression.subs(
            [(B, B_M), (H, H_M), (U, U_M), (bb, bb_M), (tb, tb_M), (rho, rho_M), (Cd, Cd_M), (Cl, Cl_M), (Cm, Cm_M), (Cd_dtn, Cd_dtn_M), (Cl_dtn, Cl_dtn_M), (Cm_dtn, Cm_dtn_M), (Ca, Ca_M),
             (Cy, Cy_M), (Cz, Cz_M), (Crx, Crx_M), (Cy_dtn, Cy_dtn_M), (Cz_dtn, Cz_dtn_M), (Crx_dtn, Crx_dtn_M),
             (delta_rXu, delta_rXu_M), (delta_rYv, delta_rYv_M), (delta_rZw, delta_rZw_M), (delta_Xu_d, delta_Xu_d_M), (delta_Yv_d, delta_Yv_d_M), (delta_Zw_d, delta_Zw_d_M)])
    elif any(x in symbols_list for x in list(delta_Lnw)+list(delta_Lnw_d)):
        assert all(x not in symbols_list for x in list(delta_Gw)+list(delta_Gw_d)+list(delta_Ls)+list(delta_Ls_d)), 'Make sure all motion-related variables are in 1 system only'
        variables_expressed_in = 'Lnw'
        delta_rD_M = symbols(r'a')
        delta_rA_M = symbols(r'f')
        delta_rL_M = symbols(r'g')
        delta_D_d_M = symbols(r'i')
        delta_A_d_M = symbols(r'j')
        delta_L_d_M = symbols(r'o')
        expression_to_M = expression.subs(
            [(B, B_M), (H, H_M), (U, U_M), (bb, bb_M), (tb, tb_M), (rho, rho_M), (Cd, Cd_M), (Cl, Cl_M), (Cm, Cm_M), (Cd_dtn, Cd_dtn_M), (Cl_dtn, Cl_dtn_M), (Cm_dtn, Cm_dtn_M), (Ca, Ca_M),
             (Cy, Cy_M), (Cz, Cz_M), (Crx, Crx_M), (Cy_dtn, Cy_dtn_M), (Cz_dtn, Cz_dtn_M), (Crx_dtn, Crx_dtn_M),
             (delta_rD, delta_rD_M), (delta_rA, delta_rA_M), (delta_rL, delta_rL_M), (delta_D_d, delta_D_d_M), (delta_A_d, delta_A_d_M), (delta_L_d, delta_L_d_M)])
    else: # no motion-dependent variables
        variables_expressed_in = None
        expression_to_M = expression.subs(
            [(B, B_M), (H, H_M), (U, U_M), (bb, bb_M), (tb, tb_M), (rho, rho_M), (Cd, Cd_M), (Cl, Cl_M), (Cm, Cm_M), (Cd_dtn, Cd_dtn_M), (Cl_dtn, Cl_dtn_M), (Cm_dtn, Cm_dtn_M), (Ca, Ca_M),
             (Cy, Cy_M), (Cz, Cz_M), (Crx, Crx_M), (Cy_dtn, Cy_dtn_M), (Cz_dtn, Cz_dtn_M), (Crx_dtn, Crx_dtn_M)])

    print("Copy the following line and run it in Wolfram Mathematica v12.1:")
    # Paste this in Mathematica:
    if signcosbb == 1:
        str_cosbb = 'Cos[b]>0,'
    elif signcosbb == -1:
        str_cosbb = 'Cos[b]<0,'
    elif signcosbb == 'any':
        str_cosbb = ''
    if form_try == 'traditional':
        print("FortranForm[Collect[FullSimplify[Collect[Simplify[Rationalize[FullSimplify[ToExpression["+'"'+str(expression_to_M)+'"'+", TraditionalForm],{s\[Element]PositiveReals,r\[Element]PositiveReals,d\[Element]Reals,e\[Element]Reals,c\[Element]PositiveReals,h\[Element]PositiveReals,t\[Element]Reals,-Pi/2<=t<=Pi/2,b\[Element]Reals,-Pi<=b<=Pi,"+str_cosbb+"l\[Element]Reals,m\[Element]Reals,k\[Element]Reals,n\[Element]Reals,a\[Element]Reals,f\[Element]Reals,g\[Element]Reals,i\[Element]Reals,j\[Element]Reals,o\[Element]Reals,z\[Element]Reals}]]],{d,l,m,e,k,n,z}]],{d,l,m,e,k,n,z}]]")
    elif form_try == 'mathml':
        print("FortranForm[Collect[FullSimplify[Collect[Simplify[Rationalize[FullSimplify[ToExpression["+'"'+mathml(expression_to_M)+'"'+", MathMLForm],{s\[Element]PositiveReals,r\[Element]PositiveReals,d\[Element]Reals,e\[Element]Reals,c\[Element]PositiveReals,h\[Element]PositiveReals,t\[Element]Reals,-Pi/2<=t<=Pi/2,b\[Element]Reals,-Pi<=b<=Pi,"+str_cosbb+"l\[Element]Reals,m\[Element]Reals,k\[Element]Reals,n\[Element]Reals,a\[Element]Reals,f\[Element]Reals,g\[Element]Reals,i\[Element]Reals,j\[Element]Reals,o\[Element]Reals,z\[Element]Reals}]]],{d,l,m,e,k,n,z}]],{d,l,m,e,k,n,z}]]")
    elif form_try == 'latex':
        print("FortranForm[Collect[FullSimplify[Collect[Simplify[Rationalize[FullSimplify[ToExpression["+'"'+latex(expression_to_M)+'"'+", TeXForm],{s\[Element]PositiveReals,r\[Element]PositiveReals,d\[Element]Reals,e\[Element]Reals,c\[Element]PositiveReals,h\[Element]PositiveReals,t\[Element]Reals,-Pi/2<=t<=Pi/2,b\[Element]Reals,-Pi<=b<=Pi,"+str_cosbb+"l\[Element]Reals,m\[Element]Reals,k\[Element]Reals,n\[Element]Reals,a\[Element]Reals,f\[Element]Reals,g\[Element]Reals,i\[Element]Reals,j\[Element]Reals,o\[Element]Reals,z\[Element]Reals}]]],{d,l,m,e,k,n,z}]],{d,l,m,e,k,n,z}]]")

    str_M = str(input("Enter the result from Wolfram Mathematica, as simple text:"))
    is_matrix = True if 'List(' in str_M else False
    str_M_1 = str_M.replace('List(','(').replace('Cot','cot').replace('Csc','csc').replace('ArcSin','asin').replace('ArcCos','acos').replace('ArcTan','atan').replace('Cos','cos').replace('Sin','sin').replace('Tan','tan').replace('Sign','sign').replace('Sec','sec').replace('Sqrt','sqrt')  # this is not an exaustive list! Make sure all functions are Python recognizable

    if variables_expressed_in == 'Gw':
        if coefficients_in == 'dal':
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c':B,'h':H,'s':U,'b':bb,'t':tb,'r':rho,'d':Cd,'l':Cl,'m':Cm,'e':Cd_dtn,'k':Cl_dtn,'n':Cm_dtn, 'z':Ca,
                                                           'a':delta_rXu, 'f':delta_rYv, 'g':delta_rZw, 'i':delta_Xu_d, 'j':delta_Yv_d, 'o':delta_Zw_d})
        else:
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c':B,'h':H,'s':U,'b':bb,'t':tb,'r':rho,'d':Cy,'l':Cz,'m':Crx,'e':Cy_dtn,'k':Cz_dtn,'n':Crx_dtn,
                                                           'a':delta_rXu, 'f':delta_rYv, 'g':delta_rZw, 'i':delta_Xu_d, 'j':delta_Yv_d, 'o':delta_Zw_d})
    elif variables_expressed_in == 'Ls':
        if coefficients_in == 'dal':
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c': B, 'h': H, 's': U, 'b': bb, 't': tb, 'r': rho, 'd': Cd, 'l': Cl, 'm': Cm, 'e': Cd_dtn, 'k': Cl_dtn, 'n': Cm_dtn, 'z':Ca,
                                                               'a': delta_rx, 'f': delta_ry, 'g': delta_rz, 'i': delta_x_d, 'j': delta_y_d, 'o': delta_z_d})
        else:
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c': B, 'h': H, 's': U, 'b': bb, 't': tb, 'r': rho, 'd':Cy, 'l':Cz, 'm':Crx, 'e':Cy_dtn, 'k':Cz_dtn, 'n':Crx_dtn,
                                                               'a': delta_rx, 'f': delta_ry, 'g': delta_rz, 'i': delta_x_d, 'j': delta_y_d, 'o': delta_z_d})
    elif variables_expressed_in == 'Lnw':
        if coefficients_in == 'dal':
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c': B, 'h': H, 's': U, 'b': bb, 't': tb, 'r': rho, 'd': Cd, 'l': Cl, 'm': Cm, 'e': Cd_dtn, 'k': Cl_dtn, 'n': Cm_dtn, 'z':Ca,
                                                               'a': delta_rD, 'f': delta_rA, 'g': delta_rL, 'i': delta_D_d, 'j': delta_A_d, 'o': delta_L_d})
        else:
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c': B, 'h': H, 's': U, 'b': bb, 't': tb, 'r': rho, 'd':Cy, 'l':Cz, 'm':Crx, 'e':Cy_dtn, 'k':Cz_dtn, 'n':Crx_dtn,
                                                               'a': delta_rD, 'f': delta_rA, 'g': delta_rL, 'i': delta_D_d, 'j': delta_A_d, 'o': delta_L_d})
    else:
        if coefficients_in == 'dal':
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c': B, 'h': H, 's': U, 'b': bb, 't': tb, 'r': rho, 'd': Cd, 'l': Cl, 'm': Cm, 'e': Cd_dtn, 'k': Cl_dtn, 'n': Cm_dtn, 'z':Ca})
        else:
            expression_from_M = parse_expr(str_M_1,local_dict={'un':un, 'vn':vn, 'wn':wn, 'u':u, 'v':v, 'w':w, 'c': B, 'h': H, 's': U, 'b': bb, 't': tb, 'r': rho, 'd':Cy, 'l':Cz, 'm':Crx, 'e':Cy_dtn, 'k':Cz_dtn, 'n':Crx_dtn})
    if is_matrix:
        expression_from_M = Matrix(expression_from_M)
    testing_and_comparing_numerical_random_values(expression_1=expression_from_M, expression_2=expression, n_tests=n_tests)
    return expression_from_M

# Mean system
U_Gw = Matrix([U, 0, 0])
U_Ls = T_LsGwb @ U_Gw
U_x = U_Ls[0]
U_y = U_Ls[1]
U_z = U_Ls[2]

# Instantaneous system

V_Gwb = Matrix([U+u, v, w])
V = sqrt((U+u)**2 + v**2 + w**2)
V_linear = sqrt(U**2 + 2*U*u)
V_Gwt = Matrix([V, 0, 0])

V_Ls = T_LsGwb @ V_Gwb
V_x = V_Ls[0]
V_y = V_Ls[1]
V_z = V_Ls[2]
V_xy_nonlinear = sqrt(V_x ** 2 + V_y ** 2)
V_xy = sqrt(simplify(expand(V_xy_nonlinear**2) + O(u**2) + O(v**2) + O(w**2) + O(u*w)).removeO())
# Confirmation:
# T_LsGwt = ( R_y(tt) @ R_z(-bt-pi/2) ).T
# V_Ls_confirm = T_LsGwt @ V_Gwt

# Easy-to-prove version. Testing two possible cases: sign(bb) = -1, 1. The sign must be changed above, where bb is defined, before all equations.
bt = acos(V_y / V_xy) * -sign(U_x)  # Correct version would be: bt = acos(V_y / V_xy)  * -sign(V_x)
# bt = atan2(-V_x,V_y)  # possible alternative. Not sure simplifications are equally effective then.

# Note: By testing numerically the 4 possible cases: (sign(bb), sign(V_x)) = (-1,-1), (-1,1), (1,-1), (1,1) the result 'bt = bb +v/U/cos(tb)' is shown to be general!
# This proof is hard to do fully symbolically. Note that the taylor approximation is around a point [u,v,w]=[0,0,0] such that -sign(U_x) == -sign(V_x)

# Taylor expansion, using my function:
bt_Taylor_general = simplify(Taylor_polynomial_sympy(function_expression=bt, variable_list=[u,v,w], evaluation_point=[0,0,0], degree=1))  # this implies a linearization around a point where sign(V_x) == sign(bb)!
bt_Taylor = simplify(bt_Taylor_general.subs(Abs(sin(bb)), sin(bb)*sign(bb)).subs(cos(tb),costb).subs(acos(cos(bb)), bb*sign(bb)).subs(costb,cos(tb)).subs(sign(sin(bb)),sign(bb)))
# # Taylor expansion confirmation:
# bt_000 = simplify(simplify(bt.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))
# bt_du = Derivative(bt, u)
# bt_dv = Derivative(bt, v)
# bt_dw = Derivative(bt, w)
# bt_du_000 = simplify(simplify(bt_du.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))  # becomes zero!!
# bt_dv_000 = simplify(simplify(bt_dv.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))
# bt_dw_000 = simplify(simplify(bt_dw.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))
# bt_Taylor_general = simplify(bt_000 + bt_du_000 * u + bt_dv_000 * v + bt_dw_000 * w)
# bt_Taylor_general = simplify(simplify(bt_Taylor_general.subs(cos(tb), costb)).subs(costb, cos(tb)))  # artifact to enforce that cos(tb) >= 0, so simplifications can be made
# if bb < 0:
#     bt_Taylor_general = bt_Taylor_general.subs(acos(cos(bb)), -bb).subs(sign(sin(bb)),sign(bb))  # acos(cos(bb)) = -bb
#     bt_Taylor = simplify(bt_Taylor_general.subs(Abs(sin(bb)), -sin(bb)))  # abs(sin(bb)) = -sin(bb)
# if bb > 0:
#     bt_Taylor_general = bt_Taylor_general.subs(acos(cos(bb)), bb).subs(sign(sin(bb)),sign(bb))  # acos(cos(bb)) = bb
#     bt_Taylor = simplify(bt_Taylor_general.subs(Abs(sin(bb)), sin(bb)))  # abs(sin(bb)) = sin(bb)
# print(bt_Taylor)

tt_nonlinear = asin(V_z / V)
tt = asin(V_z / V_linear)
# Taylor expansion:
tt_Taylor_general = simplify(Taylor_polynomial_sympy(function_expression=tt, variable_list=[u,v,w], evaluation_point=[0,0,0], degree=1))  # this implies a linearization around a point where sign(V_x) == sign(bb)!
tt_Taylor = simplify(tt_Taylor_general.subs(Abs(sin(bb)), sin(bb)*sign(bb)).subs(cos(tb),costb).subs(asin(sin(tb)),tb).subs(acos(cos(bb)), bb*sign(bb)).subs(costb,cos(tb)).subs(sign(sin(bb)),sign(bb))) # valid for a given beta [0,180]
# # Taylor expansion confirmation:
# tt_000 = simplify(simplify(tt.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))
# tt_du = simplify(Derivative(tt, u))
# tt_dv = simplify(Derivative(tt, v))
# tt_dw = simplify(Derivative(tt, w))
# tt_du_000 = simplify(simplify(tt_du.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))  # becomes zero!!
# tt_dv_000 = simplify(simplify(tt_dv.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))
# tt_dw_000 = simplify(simplify(tt_dw.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb)))
# tt_Taylor_general = simplify(tt_000 + tt_du_000 * u + tt_dv_000 * v + tt_dw_000 * w)
# tt_Taylor_general = simplify(simplify(tt_Taylor_general.subs(cos(tb), costb)).subs(costb, cos(tb)))  # artifact to enforce that cos(tb) >= 0, so simplifications can be made
# tt_Taylor = tt_Taylor_general.subs(asin(sin(tb)), tb)  # asin(sin(tb)) = tb, for tb in [-pi/2, pi/2)

T_LsGwt = ( R_y(tt_Taylor) @ R_z(-bt_Taylor-pi/2) ).T
T_GwbGwt = T_LsGwb.T @ T_LsGwt
# Taylor expansion, using my function:
T_GwbGwt_Taylor = simplify(Matrix([[Taylor_polynomial_sympy(function_expression=T_GwbGwt[i,j], variable_list=[u,v,w], evaluation_point=[0,0,0], degree=1) for j in range(3)] for i in range(3)]))
T_GwbGwt_Taylor_6 = matrix_3dof_to_6dof(T_GwbGwt_Taylor)
# # Taylor expansion confirmation:
# T_GwbGwt_000 = T_GwbGwt.subs({u:0, v:0, w:0})
# T_GwbGwt_du = Derivative(T_GwbGwt, u)
# T_GwbGwt_dv = Derivative(T_GwbGwt, v)
# T_GwbGwt_dw = Derivative(T_GwbGwt, w)
# T_GwbGwt_du_000 = simplify(T_GwbGwt_du.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb))  # becomes zero!!
# T_GwbGwt_dv_000 = simplify(T_GwbGwt_dv.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb))
# T_GwbGwt_dw_000 = simplify(T_GwbGwt_dw.subs({u:0, v:0, w:0}).subs(cos(tb),costb)).subs(costb, cos(tb))
# T_GwbGwt_Taylor = simplify(simplify(T_GwbGwt_000 + T_GwbGwt_du_000 * u + T_GwbGwt_dv_000 * v + T_GwbGwt_dw_000 * w).subs(cos(tb),costb)).subs(costb, cos(tb))

########################################################################################################################################################################################
# BUFFETING THEORY - STATIC BRIDGE
########################################################################################################################################################################################
B_mat = diag([B, B, B, B**2, B**2, B**2], unpack=True)

delta_b = bt_Taylor - bb
delta_t = tt_Taylor - tb

# In Gw:
Cuvw_Taylor = Cuvw + Cuvw_db*delta_b + Cuvw_dt*delta_t
fb_Gw =  S.Half * rho * V_linear**2 * T_GwbGwt_Taylor_6 * B_mat * Cuvw_Taylor - S.Half*rho*U**2*B_mat*Cuvw
A_Gw = simplify(Matrix([[Poly(fb_Gw[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint

# In Ls (ELs - 'Expressed with Ls coefficients'):
Cxyz_Taylor = Cxyz + Cxyz_db*delta_b + Cxyz_dt*delta_t
fad_Ls =  S.Half * rho * V_linear**2 * B_mat * Cxyz_Taylor
A_Ls = simplify(Matrix([[Poly(fad_Ls[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint

# # Confirmation:
# Cuvw_Taylor_ECxyz = Cuvw_ECxyz + Cuvw_db_ECxyz*delta_b + Cuvw_dt_ECxyz*delta_t # Expressed in the Ls system
# fad_Ls_2 =  T_LsGwb_6 * S.Half * rho * V_linear**2 * T_GwbGwt_Taylor_6 * B_mat * Cuvw_Taylor_ECxyz
# A_Ls_2 = simplify(Matrix([[Poly(fad_Ls_2[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# # Matching Strommen's formulation:
# fad_Ls_2_ECuvw =  fad_Ls_2.subs(list_Cxyz_to_Cuvw+list_Cxyz_db_to_Cuvw+list_Cxyz_dt_to_Cuvw)
# A_Ls_2_ECuvw = simplify(Matrix([[Poly(fad_Ls_2_ECuvw[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# A_Ls_2.subs(bb,0).subs(tb,0)
# A_Ls_2_ECuvw.subs(bb,0).subs(tb,0)
# testing_and_comparing_numerical_random_values(expression_1=T_LsGwb_6 @ A_Gw @ Matrix([u,v,w]), expression_2=A_Ls @ Matrix([u,v,w]))

########################################################################################################################################################################################
# BUFFETING THEORY - WITH SELF EXCITED FORCES (BRIDGE MOTIONS)
########################################################################################################################################################################################
# Expressing Ls quantities as function of Gw quantities
delta_Ls_EGw = T_LsGwb_6 @ delta_Gw  # expressed as a function of Gw coefficients
delta_x_EGw, delta_y_EGw, delta_z_EGw, delta_rx_EGw, delta_ry_EGw, delta_rz_EGw = delta_Ls_EGw
delta_Ls_d_EGw = T_LsGwb_6 @ delta_Gw_d  # expressed as a function of Gw coefficients
delta_x_d_EGw, delta_y_d_EGw, delta_z_d_EGw, delta_rx_d_EGw, delta_ry_d_EGw, delta_rz_d_EGw = delta_Ls_d_EGw

# Expressing Gw quantities as function of Ls quantities
delta_Gw_ELs = T_LsGwb_6.T @ delta_Ls  # expressed as a function of Gw coefficients
delta_Xu_ELs, delta_Yv_ELs, delta_Zw_ELs, delta_rXu_ELs, delta_rYv_ELs, delta_rZw_ELs = delta_Gw_ELs
delta_Gw_d_ELs = T_LsGwb_6.T @ delta_Ls_d  # expressed as a function of Gw coefficients
delta_Xu_d_ELs, delta_Yv_d_ELs, delta_Zw_d_ELs, delta_rXu_d_ELs, delta_rYv_d_ELs, delta_rZw_d_ELs = delta_Gw_d_ELs

u_rel = u - delta_Xu_d
v_rel = v - delta_Yv_d
w_rel = w - delta_Zw_d
V_rel = sqrt((U + u_rel)**2 + v_rel**2 + w_rel**2)

# Auxiliary lists, for an easy change of variables.
list_delta_Gw_to_ELs = [(delta_Xu, delta_Xu_ELs), (delta_Yv, delta_Yv_ELs), (delta_Zw, delta_Zw_ELs), (delta_rXu, delta_rXu_ELs), (delta_rYv, delta_rYv_ELs), (delta_rZw, delta_rZw_ELs),
                        (delta_Xu_d, delta_Xu_d_ELs), (delta_Yv_d, delta_Yv_d_ELs), (delta_Zw_d, delta_Zw_d_ELs), (delta_rXu_d, delta_rXu_d_ELs), (delta_rYv_d, delta_rYv_d_ELs), (delta_rZw_d, delta_rZw_d_ELs)]
list_delta_Ls_to_EGw = [(delta_x, delta_x_EGw), (delta_y, delta_y_EGw), (delta_z, delta_z_EGw), (delta_rx, delta_rx_EGw), (delta_ry, delta_ry_EGw), (delta_rz, delta_rz_EGw),
                        (delta_x_d, delta_x_d_EGw), (delta_y_d, delta_y_d_EGw), (delta_z_d, delta_z_d_EGw), (delta_rx_d, delta_rx_d_EGw), (delta_ry_d, delta_ry_d_EGw), (delta_rz_d, delta_rz_d_EGw)]

######################################################### DERIVATION SECTION BELOW - HOW TO OBTAIN bt_Taylor_rel & tt_Taylor_rel ##################################################
V_x_rel = V_x - delta_x_d
V_y_rel = V_y - delta_y_d
V_z_rel = V_z - delta_z_d
V_Ls_rel = Matrix([V_x_rel, V_y_rel, V_z_rel])
# # Equivalent alternative:
# V_Gwb_rel = Matrix([U+u_rel, v_rel, w_rel])
# V_Ls_rel_2 = (T_LsGwb @ V_Gwb_rel).subs(list_delta_Gw_to_ELs)
# print(simplify(V_Ls_rel - V_Ls_rel_2))

V_xy_nonlinear_rel = sqrt(V_x_rel**2 + V_y_rel**2)
V_xy_rel = simplify(Taylor_polynomial_sympy(V_xy_nonlinear_rel, [u,v,w]+list(delta_Ls)+list(delta_Ls_d), [0]*15, 1).subs(cos(tb),costb)).subs(costb,cos(tb))
# # Equivalent alternative:
# V_xy_rel_2 = simplify(simplify(Taylor_polynomial_sympy(V_xy_nonlinear_rel.subs(list_delta_Ls_to_EGw), [u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1).subs(list_delta_Gw_to_ELs).subs(cos(tb),costb)).subs(costb,cos(tb)))

# Now including rotational motions (using "relrel" notation when including both translational velocities ("rel") and rotational displacements (+"rel")
T_3rotations_ELs = (R_x(delta_rx) @ R_y(delta_ry) @ R_z(delta_rz)).T
T_LsttLs_ELs = simplify(Matrix([[Taylor_polynomial_sympy(T_3rotations_ELs[i, j], [delta_rx, delta_ry, delta_rz], [0, 0, 0], 1) for j in range(3)] for i in range(3)]))  # beautiful! infinitesimal angular displacements are commutative!
T_LsttLs_EGw = T_LsttLs_ELs.subs(list_delta_Ls_to_EGw)
# # Normalizing the linearized matrix. But, if it is linearized again, it goes back to the same.
# from sympy import det
# T_LsttLs_ELs_norm = simplify(T_LsttLs_ELs / det(T_LsttLs_ELs))
# T_LsttLs_ELs_norm_simple = Matrix([[Taylor_polynomial_sympy(T_LsttLs_ELs_norm[i,j], list(delta_Ls), [0]*6, 1) for j in range(3)] for i in range(3)])
V_Ls_relrel = T_LsttLs_ELs @ V_Ls_rel
# Equivalent alternative:
# V_Gwb_rel = Matrix([U+u_rel, v_rel, w_rel])
# V_Ls_relrel_2 = (T_LsttLs_ELs @ T_LsGwb @ V_Gwb_rel).subs(list_delta_Gw_to_ELs)
# testing_and_comparing_numerical_random_values(V_Ls_relrel, V_Ls_relrel_2)
V_x_relrel = V_Ls_relrel[0]
V_y_relrel = V_Ls_relrel[1]
V_z_relrel = V_Ls_relrel[2]
V_xy_relrel = sqrt(V_x_relrel**2 + V_y_relrel**2)
V_xy_Taylor_relrel = simplify(Taylor_polynomial_sympy(V_xy_relrel, [u,v,w]+list(delta_Ls)+list(delta_Ls_d), [0]*15, 1).subs(cos(tb),costb)).subs(costb,cos(tb))


# # Equivalent alternative:
# V_xy_relrel_2 = V_xy_relrel.subs(list_delta_Ls_to_EGw)
# V_xy_Taylor_relrel_2 = simplify(simplify(Taylor_polynomial_sympy(V_xy_relrel_2, [u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1).subs(cos(tb),costb)).subs(costb,cos(tb)).subs(list_delta_Gw_to_ELs))
V_y_Taylor_relrel = simplify(Taylor_polynomial_sympy(V_y_relrel, [u,v,w]+list(delta_Ls)+list(delta_Ls_d), [0]*15, 1).subs(cos(tb),costb)).subs(costb,cos(tb))

if bb < 0:
    bt_relrel = -acos(V_y_Taylor_relrel / V_xy_Taylor_relrel) # - delta_rz_EGw
if bb > 0:
    bt_relrel = acos(V_y_Taylor_relrel / V_xy_Taylor_relrel) # - delta_rz_EGw

bt_Taylor_relrel_general = simplify(Taylor_polynomial_sympy(bt_relrel, [u,v,w]+list(delta_Ls)+list(delta_Ls_d), [0]*15, 1))
bt_Taylor_relrel_general = simplify(bt_Taylor_relrel_general.subs(cos(tb), costb)).subs(costb, cos(tb))
if bb < 0:
    bt_Taylor_relrel_general = bt_Taylor_relrel_general.subs(acos(cos(bb)), -bb)  # acos(cos(bb)) = -bb
    bt_Taylor_relrel = bt_Taylor_relrel_general.subs(Abs(sin(bb)), -sin(bb))  # abs(sin(bb)) = -sin(bb)
if bb > 0:
    bt_Taylor_relrel_general = bt_Taylor_relrel_general.subs(acos(cos(bb)), bb)  # acos(cos(bb)) = bb
    bt_Taylor_relrel = bt_Taylor_relrel_general.subs(Abs(sin(bb)), sin(bb))  # abs(sin(bb)) = sin(bb)
# Changing to Gw coordinates
bt_Taylor_relrel = expand(simplify(bt_Taylor_relrel.subs(list_delta_Ls_to_EGw)))

# And now the theta:
V_relrel = sqrt(V_x_relrel**2 + V_y_relrel**2 + V_z_relrel**2)  # the magnitude of this should be the same as the V_rel
# testing_and_comparing_numerical_random_values(expression_1=V_rel, expression_2=V_relrel.subs(list_delta_Ls_to_EGw))  # this is supposed to be only approximate. Note that the det(T_LsttLs) is slightly larger than 1, hence the difference
tt_relrel = asin(V_z_relrel / V_relrel)
tt_Taylor_relrel_general = simplify(Taylor_polynomial_sympy(tt_relrel, [u,v,w]+list(delta_Ls)+list(delta_Ls_d), [0]*15, 1))
tt_Taylor_relrel = simplify(simplify(tt_Taylor_relrel_general.subs(cos(tb), costb)).subs(costb, cos(tb)).subs(asin(sin(tb)), tb).subs(list_delta_Ls_to_EGw))
######################################################### DERIVATION SECTION ABOVE - HOW TO OBTAIN bt_Taylor_rel & tt_Taylor_rel ##################################################

# Relative bt and tt (including all structural motions and rotations):
bt_Taylor_rel = bb + v_rel/(U*cos(tb)) - delta_rZw/cos(tb)  # proof in the section above.
tt_Taylor_rel = tb + w_rel/U + delta_rYv  # proof in the section above.
delta_b_rel = bt_Taylor_rel - bb
delta_t_rel = tt_Taylor_rel - tb
Cuvw_Taylor_rel = Cuvw + Cuvw_db*delta_b_rel + Cuvw_dt*delta_t_rel
# Transformation matrix, including the effects of both the turbulences, and the motions! "Lsdelta" = Ls displaced
T_LsdeltaGwtrel = (R_y(tt_Taylor_rel) @ R_z(-bt_Taylor_rel-pi/2)).T # By definition, the Yv_tilde_rel is in the x_rel-y_rel plane!! No R_x is necessary. todo: is there something missing here...?

T_3rotations_EGw = (R_x(delta_rx_EGw) @ R_y(delta_ry_EGw) @ R_z(delta_rz_EGw)).T  # 3 generic rotations around the Ls axes. After the Taylorization, for small angles, rotations become commutative!
T_LsttLs_EGw = simplify(Matrix([[Taylor_polynomial_sympy(T_3rotations_EGw[i, j], [delta_rXu, delta_rYv, delta_rZw], [0, 0, 0], 1) for j in range(3)] for i in range(3)]))  # beautiful! infinitesimal angular displacements are commutative!
T_LsttLs_ELs = simplify(T_LsttLs_EGw.subs(list_delta_Gw_to_ELs))

# T_LsttLs_EGw = diag(1,1,1)
T_GwbGwt_rel = T_LsGwb.T @ T_LsttLs_EGw.T @ T_LsdeltaGwtrel  # todo: Is this redundant?! the 3rotations are accounted for in the second and third T matrices.
T_GwbGwt_Taylor_rel = simplify(Matrix([[Taylor_polynomial_sympy(T_GwbGwt_rel[i,j], [u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1) for j in range(3)] for i in range(3)]))
T_GwbGwt_Taylor_6_rel = matrix_3dof_to_6dof(T_GwbGwt_Taylor_rel)
V_rel_Taylor_square = Taylor_polynomial_sympy(V_rel**2, [u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1)
# Confirmation that V_rel_Taylor_square == V_relrel_Taylor_square
# V_relrel_Taylor_square = Taylor_polynomial_sympy((V_relrel**2).subs(list_delta_Ls_to_EGw), [u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1)
# V_relrel_Taylor_square_M = U*(U - 2*delta_Xu_d + 2*u)
# Buffeting forces with self-excited forces:
fb_Gw_rel =  T_GwbGwt_Taylor_6_rel * S.Half * rho * V_rel_Taylor_square * B_mat * Cuvw_Taylor_rel - S.Half*rho*U**2*B_mat*Cuvw
A_Gw_rel = simplify(Matrix([[Poly(fb_Gw_rel[i],[u,v,w]+list(delta_Gw)+list(delta_Gw_d)).coeff_monomial(([u,v,w]+list(delta_Gw)+list(delta_Gw_d))[j]) for j in range(15)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# These results are only simplified if bb is either positive = True or negative = True
Auvw_Gw_rel = A_Gw_rel[:,:3]
Kse_Gw = A_Gw_rel[:,3:9]
Cse_Gw = A_Gw_rel[:,9:]
Cse_Ls = T_LsGwb_6 @ Cse_Gw @ T_LsGwb_6.T
Kse_Ls = T_LsGwb_6 @ Kse_Gw @ T_LsGwb_6.T
Kse_LsGw = T_LsGwb_6 @ Kse_Gw
Auvw_Gw_rel  / (S.Half*rho*U) # <-------------------------------------------------- Enter in console to print nicely. Same as the one without SE forces.
Kse_Gw  / (S.Half*rho*U**2)  # <-------------------------------------------------- Enter in console to print nicely
Cse_Gw  / (S.Half*rho*U)  # <-------------------------------------------------- Enter in console to print nicely

# Special cases:
Kse_Ls = simplify(Kse_Ls)
Kse_LsGw = simplify(Kse_LsGw)
Kse_Ls.subs(bb,0).subs(tb,0)  # todo: MAKES SENSE
Kse_LsGw.subs(bb,0).subs(tb,0)  # todo: MAKES SENSE
Kse_Gw.subs(bb,0).subs(tb,0)  # todo: MAKES SENSE

# And now in LsGw coordinates, with Cxyz coefficients:
Cxyz_Taylor_rel = Cxyz + Cxyz_db*delta_b_rel + Cxyz_dt*delta_t_rel
T_LsLstt_6_EGw = matrix_3dof_to_6dof(T_LsttLs_EGw.T)
fb_Ls_rel =  T_LsLstt_6_EGw * S.Half * rho * V_rel_Taylor_square * B_mat * Cxyz_Taylor_rel - S.Half*rho*U**2*B_mat*Cuvw
A_LsGw_rel = simplify(Matrix([[Poly(fb_Ls_rel[i],[u,v,w]+list(delta_Gw)+list(delta_Gw_d)).coeff_monomial(([u,v,w]+list(delta_Gw)+list(delta_Gw_d))[j]) for j in range(15)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
A_LsGw_rel = Matrix([[simplify(expand_trig(collect(A_LsGw_rel[i,j], list(Cxyz)+list(Cxyz_db)+list(Cxyz_dt)))) for j in range(15)] for i in range(6)])
Auvw_LsGw_rel = A_LsGw_rel[:,:3]
Kse_LsGw = A_LsGw_rel[:,3:9]
Cse_LsGw = A_LsGw_rel[:,9:]
Kse_Ls_ECxyz = simplify(Kse_LsGw @ T_LsGwb_6.T)
testing_and_comparing_numerical_random_values(expression_1=Kse_Ls_ECxyz, expression_2= T_LsGwb_6 @ Kse_Gw @ T_LsGwb_6.T)
# And now in purely Ls coordinates (function of delta_Ls), with Cxyz coefficients:
A_Ls_rel = simplify(Matrix([[Poly(fb_Ls_rel[i],[u,v,w]+list(delta_Ls)+list(delta_Ls_d)).coeff_monomial(([u,v,w]+list(delta_Ls)+list(delta_Ls_d))[j]) for j in range(15)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint



########################################################################################################################################################################################
# SCANLAN FLUTTER DERIVATIVES, AND THE RESPECTIVE KSE AND CSE
########################################################################################################################################################################################
# Following the notation in You-Lin Xu book, eq. 10.37a (same notation as by L.D. Zhu)
K = symbols('K', real=True)
A1 = symbols('A1', real=True)
A2 = symbols('A2', real=True)
A3 = symbols('A3', real=True)
A4 = symbols('A4', real=True)
A5 = symbols('A5', real=True)
A6 = symbols('A6', real=True)
P1 = symbols('P1', real=True)
P2 = symbols('P2', real=True)
P3 = symbols('P3', real=True)
P4 = symbols('P4', real=True)
P5 = symbols('P5', real=True)
P6 = symbols('P6', real=True)
H1 = symbols('H1', real=True)
H2 = symbols('H2', real=True)
H3 = symbols('H3', real=True)
H4 = symbols('H4', real=True)
H5 = symbols('H5', real=True)
H6 = symbols('H6', real=True)
f_se_x_Scanlan = 0
f_se_y_Scanlan  = S.Half * rho * U**2 * B * (K*P1*delta_y_d/U + K*P2*B*delta_rx_d/U + K**2*P3*delta_rx + K**2*P4*delta_y/B + K*P5*delta_z_d/U + K**2*P6*delta_z/B)
f_se_z_Scanlan  = S.Half * rho * U**2 * B * (K*H1*delta_z_d/U + K*H2*B*delta_rx_d/U + K**2*H3*delta_rx + K**2*H4*delta_z/B + K*H5*delta_y_d/U + K**2*H6*delta_y/B)
f_se_rx_Scanlan = S.Half * rho * U**2 * B**2 * (K*A1*delta_z_d/U + K*A2*B*delta_rx_d/U + K**2*A3*delta_rx + K**2*A4*delta_z/B + K*A5*delta_y_d/U + K**2*A6*delta_y/B)
f_se_ry_Scanlan = 0
f_se_rz_Scanlan = 0
f_se_mat_Ls_Scanlan = Matrix([f_se_x_Scanlan, f_se_y_Scanlan, f_se_z_Scanlan, f_se_rx_Scanlan, f_se_ry_Scanlan, f_se_rz_Scanlan])
A_Ls_rel_Scanlan = simplify(Matrix([[Poly(f_se_mat_Ls_Scanlan[i], [u,v,w]+list(delta_Ls)+list(delta_Ls_d)).coeff_monomial(([u,v,w]+list(delta_Ls)+list(delta_Ls_d))[j]) for j in range(15)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
Auvw_Ls_rel_Scanlan = A_Ls_rel_Scanlan[:,:3]
Kse_Ls_Scanlan = A_Ls_rel_Scanlan[:,3:9]
Cse_Ls_Scanlan = A_Ls_rel_Scanlan[:,9:]
Auvw_Ls_rel_Scanlan  # <-------------------------------------------------- Enter in console to print nicely. Same as the one without SE forces.
Kse_Ls_Scanlan  # <-------------------------------------------------- Enter in console to print nicely
Cse_Ls_Scanlan  # <-------------------------------------------------- Enter in console to print nicely
# Now considering that Cse_Gw == T_LsGwb_6.T @ Cse_Ls_Scanlan
# And considering that Kse_Gw == T_LsGwb_6.T @ Kse_Ls_Scanlan
# ...the quasy steady-state flutter derivatives can be obtained:
eqn1 = Cse_Ls - Cse_Ls_Scanlan  # equals to 0.
eqn2 = Kse_Ls - Kse_Ls_Scanlan  # equals to 0.
FD_Scanlan = simplify(solve(eqn1[:] + eqn2[:], P1, P2, P3, P4, P5, P6, H1, H2, H3, H4, H5, H6, A1, A2, A3, A4, A5, A6))  # the "+" is joining the eqn1 and eqn2 tuples
FD_Scanlan  # <-------------------------------------------------- Enter in console to print nicely

A_FD_Scanlan = simplify(Matrix([[Poly(FD_Scanlan[i],list(Cuvw)+list(Cuvw_db)+list(Cuvw_dt)).coeff_monomial((list(Cuvw)+list(Cuvw_db)+list(Cuvw_dt))[j]) for j in range(18)] for i in [P1, P3, P5, H1, H3, H5, A1, A3, A5]]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint


# wolfram_mathematica_equation_simplification(expression=diag(K,K**2,K,K,K**2,K,K,K**2,K) @ A_FD_Scanlan, form_try='mathml')
A_FD_Scanlan_M = diag(K,K**2,K,K,K**2,K,K,K**2,K)**-1 @ Matrix([

[-cos(bb)**2*cos(tb)**2 - 1, 0.5*sin(2*bb)*cos(2*tb)*sec(tb), (-sin(tb)**2*cos(bb)**2 + 1)*tan(tb),                          0,                                         0,                                  0,          sin(bb)*cos(bb), -sin(bb)**2*sec(tb),        -sin(bb)*cos(bb)*tan(tb),                  0,                        0,                           0, sin(tb)*cos(bb)**2*cos(tb), -sin(bb)*sin(tb)*cos(bb),  -sin(tb)**2*cos(bb)**2,                                0,                   0,                          0],
[                         0,         sin(bb)*cos(bb)*tan(tb),                  -sin(bb)**2*sec(tb),                          0,                                         0,                                  0, -sin(bb)*sin(tb)*cos(bb),  sin(bb)**2*tan(tb), sin(bb)*sin(tb)*cos(bb)*tan(tb),                  0,                        0,                           0,        -cos(bb)**2*cos(tb),          sin(bb)*cos(bb),      sin(tb)*cos(bb)**2,                                0,                   0,                          0],
[  -sin(tb)*cos(bb)*cos(tb),               2*sin(bb)*sin(tb),             (sin(tb)**2 + 1)*cos(bb),                          0,                                         0,                                  0,                        0,                   0,                               0,                  0,                        0,                           0,        -cos(bb)*cos(tb)**2,          sin(bb)*cos(tb), sin(tb)*cos(bb)*cos(tb),                                0,                   0,                          0],
[       0.5*cos(2*tb) - 1.5,                               0,                     -sin(tb)*cos(tb),                          0,                                         0,                                  0,                        0,                   0,                               0,                  0,                        0,                           0,           -sin(tb)*cos(tb),                        0,             -cos(tb)**2,                                0,                   0,                          0],
[                         0,                        -sin(bb),                                    0,                          0,                                         0,                                  0, -sin(bb)*sin(tb)*tan(tb),                   0,                -sin(bb)*sin(tb),                  0,                        0,                           0,           -sin(tb)*cos(bb),                        0,        -cos(bb)*cos(tb),                                0,                   0,                          0],
[  -sin(tb)*cos(bb)*cos(tb),                               0,             (sin(tb)**2 - 2)*cos(bb),                          0,                                         0,                                  0,          sin(bb)*tan(tb),                   0,                         sin(bb),                  0,                        0,                           0,         sin(tb)**2*cos(bb),                        0, sin(tb)*cos(bb)*cos(tb),                                0,                   0,                          0],
[                         0,                               0,                                    0,    sin(bb)*sin(tb)*cos(tb),                         2*sin(tb)*cos(bb),           (cos(tb)**2 - 2)*sin(bb),                        0,                   0,                               0,                  0,                        0,                           0,                          0,                        0,                       0,               sin(bb)*cos(tb)**2,     cos(bb)*cos(tb),   -sin(bb)*sin(tb)*cos(tb)],
[                         0,                               0,                                    0,                          0,                       -sin(bb)**2*tan(tb),           -sin(bb)*cos(bb)*sec(tb),                        0,                   0,                               0, sin(bb)**2*sin(tb),  sin(bb)*cos(bb)*tan(tb), -sin(bb)**2*sin(tb)*tan(tb),                          0,                        0,                       0,          sin(bb)*cos(bb)*cos(tb),          cos(bb)**2,   -sin(bb)*sin(tb)*cos(bb)],
[                         0,                               0,                                    0, sin(bb)*cos(bb)*cos(tb)**2, sin(bb)**2*sec(tb) + 2*cos(bb)**2*cos(tb), sin(bb)*sin(tb)**2*cos(bb)*tan(tb),                        0,                   0,                               0,        -sin(bb)**2, -sin(bb)*cos(bb)*sec(tb),          sin(bb)**2*tan(tb),                          0,                        0,                       0, -sin(bb)*sin(tb)*cos(bb)*cos(tb), -sin(tb)*cos(bb)**2, sin(bb)*sin(tb)**2*cos(bb)]])
# Confirmation:
# for i in list(range(9)):
#     j = [P1, P3, P5, H1, H3, H5, A1, A3, A5][i]
#     print(compare_symbolically(FD_Scanlan[j], (A_FD_Scanlan_M @ Matrix(list(Cuvw)+list(Cuvw_db)+list(Cuvw_dt)))[i]))


# Copy of L.D. Zhu flutter derivatives, for comparison. Replacing coefficients accordingly, e.g.: C_Cq = -Cv
P1_star_Zhu = -1 / K * (((2 * cos(tb)**2 - 1) * sin(bb) * cos(bb) / cos(tb)) * (-Cv)
                    + (1 + cos(bb)**2 * cos(tb)**2) * Cu
                    - ((sin(bb)**2 + cos(bb)**2 * cos(tb)**2) * tan(tb)) * Cw
                    - (sin(bb)**2 / cos(tb)) * (-Cv_db) - (sin(bb) * cos(bb) * sin(tb)) * (-Cv_dt)
                    - (sin(bb) * cos(bb)) * Cu_db - (cos(bb)**2 * sin(tb) * cos(tb)) * Cu_dt
                    + (sin(bb) * cos(bb) * tan(tb)) * Cw_db + (cos(bb)**2 * sin(tb)**2) * Cw_dt)
P2_star_Zhu = 0
P3_star_Zhu = -1 / (K ** 2) * ((sin(bb) * cos(bb)) * (-Cv_dt) + (cos(bb)**2 * cos(tb)) * Cu_dt
                           - (cos(bb)**2 * sin(tb)) * Cw_dt)
P4_star_Zhu = 0
P5_star_Zhu = -1 / K * ((2 * sin(bb) * sin(tb)) * (-Cv) + (cos(bb) * sin(tb) * cos(tb)) * Cu
                    - ((2 - cos(tb)**2) * cos(bb)) * Cw + (sin(bb) * cos(tb)) * (-Cv_dt)  # this line from Zhu probably includes a typo! replaced cos(bb) by cos(tb) and we have perfect match.
                    + (cos(bb) * cos(tb)**2) * Cu_dt - (cos(bb) * sin(tb) * cos(tb)) * Cw_dt)
P6_star_Zhu = 0
H1_star_Zhu = -1 / K * ((2 - cos(tb)**2) * Cu + cos(tb)**2 * Cw_dt + (sin(tb) * cos(tb)) * (Cw + Cu_dt))
H2_star_Zhu = 0
H3_star_Zhu = -1 / (K ** 2) * ((cos(bb) * sin(tb)) * Cu_dt + (cos(bb) * cos(tb)) * Cw_dt)
H4_star_Zhu = 0
H5_star_Zhu = -1 / K * ((cos(bb) * sin(tb) * cos(tb)) * Cu + (cos(bb) * (1 + cos(tb)**2)) * Cw
                    - (sin(bb) * tan(tb)) * Cu_db - (cos(bb) * sin(tb)**2) * Cu_dt
                    - (sin(bb)) * Cw_db - (cos(bb) * sin(tb) * cos(tb)) * Cw_dt)
H6_star_Zhu = 0
A1_star_Zhu = -1 / K * ((2 * cos(bb) * sin(tb)) * (-Cvv) - (sin(bb) * sin(tb) * cos(tb)) * Cuu
                    + ((2 - cos(tb)**2) * sin(bb)) * Cww + (cos(bb) * cos(tb)) * (-Cvv_dt)
                    - (sin(bb) * cos(tb)**2) * Cuu_dt + (sin(bb) * sin(tb) * cos(tb)) * Cww_dt)
A2_star_Zhu = 0
A3_star_Zhu = -1 / (K ** 2) * ((cos(bb)**2) * (-Cvv_dt) - (sin(bb) * cos(bb) * cos(tb)) * Cuu_dt
                           + (sin(bb) * cos(bb) * sin(tb)) * Cww_dt)
A4_star_Zhu = 0
A5_star_Zhu = - 1 / K * (((sin(bb)**2 + 2 * cos(bb)**2 * cos(tb)**2) / cos(tb)) * (-Cvv)
                     - (sin(bb) * cos(bb) * cos(tb)**2) * Cuu - (sin(bb) * cos(bb) * sin(tb)**2 * tan(tb)) * Cww
                     - (sin(bb) * cos(bb) / cos(tb)) * (-Cvv_db) - (cos(bb)**2 * sin(tb)) * (-Cvv_dt)
                     + (sin(bb)**2) * Cuu_db + (sin(bb) * cos(bb) * sin(tb) * cos(tb)) * Cuu_dt
                     - (sin(bb)**2 * tan(tb)) * Cww_db - (sin(bb) * cos(bb) * sin(tb)**2) * Cww_dt)
A6_star_Zhu = 0
# Printing the differences:
print('My difference in FDs: ' + str(simplify(FD_Scanlan[P1] - P1_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[P3] - P3_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[P5] - P5_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[H1] - H1_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[H3] - H3_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[H5] - H5_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[A1] - A1_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[A3] - A3_star_Zhu)))
print('My difference in FDs: ' + str(simplify(FD_Scanlan[A5] - A5_star_Zhu)))

########################################################################################################################################################################################
# LE-DONG ZHU - Scanlan flutter derivatives extactly according to Zhu (which I believe are incomplete and inaccurate)
########################################################################################################################################################################################
# First way of obtaining Zhu's flutter derivatives, using his reasoning in eq. (5-12) and (5-13):
Kse_Ls_Zhu_times_deltas = - S.Half * rho * T_LsGwb_6 * U**2 * B_mat @ Matrix(Cuvw_dt).col_insert(1, Matrix(Cuvw_db)) @ Matrix([(-delta_rYv_ELs), delta_rz])
Kse_Ls_Zhu = simplify(Matrix([[Poly(Kse_Ls_Zhu_times_deltas[i], list(delta_Ls)).coeff_monomial(list(delta_Ls)[j]) for j in range(6)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# Second way of obtaining Zhu's flutter derivatives, in the same fashion as our paper and reasoning:
bt_Taylor_rel_Zhu = bb + v/(U*cos(tb)) - delta_rz_EGw  # this lacks v_rel, and the contribution from other rotations: ry is important to change bt when e.g. bb = 0 & tb = 45 deg.
tt_Taylor_rel_Zhu = tb + w/U + delta_rYv  # this lacks w_rel
delta_b_rel_Zhu = bt_Taylor_rel_Zhu - bb
delta_t_rel_Zhu = tt_Taylor_rel_Zhu - tb
Cuvw_Taylor_rel_Zhu = Cuvw + Cuvw_db*delta_b_rel_Zhu + Cuvw_dt*delta_t_rel_Zhu
fb_Gw_rel_Zhu =  S.Half * rho * V_rel**2 * 1 * B_mat * Cuvw_Taylor_rel_Zhu - S.Half*rho*U**2*B_mat*Cuvw  # This equation, equivalent to Zhu's reasoning, lacks a transformation matrix!
A_Gw_rel_Zhu = simplify(Matrix([[Poly(fb_Gw_rel_Zhu[i], u, v, w, delta_Xu, delta_Yv, delta_Zw, delta_rXu, delta_rYv, delta_rZw, delta_Xu_d, delta_Yv_d, delta_Zw_d, delta_rXu_d, delta_rYv_d, delta_rZw_d).coeff_monomial([u, v, w, delta_Xu, delta_Yv, delta_Zw, delta_rXu, delta_rYv, delta_rZw, delta_Xu_d, delta_Yv_d, delta_Zw_d, delta_rXu_d, delta_rYv_d, delta_rZw_d][j]) for j in range(15)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
Kse_Gw_Zhu = A_Gw_rel_Zhu[:,3:9]
# Confirming the flutter derivatives shown in eq. (5-16). A typo was found in his Thesis in eq. (5-16c) where the last cos(bb) of the second line shoud be cos(tb) instead.
eqn1 = T_LsGwb_6 @ Cse_Gw @ T_LsGwb_6.T - Cse_Ls_Scanlan  # Note that the Cse_Gw is not from the Self-Excited forces section, but from the linear buffeting theory
eqn2_Zhu = Kse_Ls_Zhu - Kse_Ls_Scanlan
FD_Scanlan_Zhu = simplify(solve(eqn1[:] + eqn2_Zhu[:], P1, P2, P3, P4, P5, P6, H1, H2, H3, H4, H5, H6, A1, A2, A3, A4, A5, A6))  # the "+" is joining the eqn1 and eqn2 tuples
FD_Scanlan_Zhu  # <-------------------------------------------------- Enter in console to print nicely
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[P1] - P1_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[P3] - P3_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[P5] - P5_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[H1] - H1_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[H3] - H3_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[H5] - H5_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[A1] - A1_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[A3] - A3_star_Zhu)))
print('Zhu (confirmation) difference: ' + str(simplify(FD_Scanlan_Zhu[A5] - A5_star_Zhu)))
# # Comparison with Strømmen, whose forces are presented in a static Gs system (the y and z do not follow the motions):
# print(Kse_Ls.subs(bb,0).subs(tb,0))
# print(Kse_Ls_Zhu.subs(bb,0).subs(tb,0))

########################################################################################################################################################################################
# ONLY NORMAL WIND APPROXIMATION - STATIC BRIDGE - (Only the wind in the yz-plane is considered)
########################################################################################################################################################################################
# 'n' letter is used to denote 'normal', meaning in the yz-plane
T_LsLnwb = (R_y(tbn)*R_z(-pi/2*sign(cosbb))).T

Vn = sqrt((Un+un)**2 + wn**2)  # Un and un - along Drag; wn along Lift
Vn_Lnwb = Matrix([Un+un, -sign(cos(bb))*U_x+vn, wn])
Vn_Ls = T_LsLnwb @ Vn_Lnwb
Vn_z = Vn_Ls[2]  # This is correct. Confirmed that: Vn_Ls[2] == V_Ls[2].
ttn = asin(Vn_z / Vn)

ttn_Taylor = simplify(Taylor_polynomial_sympy(ttn, [un,vn,wn], [0,0,0], 1)).subs(cos(tbn), costb).subs(costb,cos(tbn)).subs(asin(sin(tbn)),tbn)
Vn_Taylor_square = simplify(Taylor_polynomial_sympy(Vn**2, [un,vn,wn], [0,0,0], 1)).subs(cos(tbn), costb).subs(costb,cos(tbn)).subs(asin(sin(tbn)),tbn)

delta_tn_Taylor = ttn_Taylor - tbn

T_LnwbLnwt = (R_y(ttn_Taylor)*R_y(-tbn)).T
# I thought that this assumes un and wn are smaller than Vn, which is only true for small beta. But, when linearizing for u,v,w and formulating w.r.t U, the same results were obtained!!
T_LnwbLnwt_Taylor = simplify(Matrix([[Taylor_polynomial_sympy(T_LnwbLnwt[i,j], [un, vn, wn], [0, 0, 0], 1) for j in range(3)] for i in range(3)]))  # ONE MORE TAYLOR.
T_LnwbLnwt_Taylor_6 = matrix_3dof_to_6dof(T_LnwbLnwt_Taylor)

Cdal = Matrix([Cd, 0, Cl, 0, Cm, 0])
Cdal_dtn = Matrix([Cd_dtn, 0, Cl_dtn, 0, Cm_dtn, 0])

Cdal_Taylor = Cdal + Cdal_dtn * delta_tn_Taylor


B_mat_Lnw = diag([H, B, B, B**2, B**2, B**2], unpack=True)
fad_Lnw_cosr1 = T_LnwbLnwt_Taylor_6 * S.Half * rho * Vn**2 * B_mat_Lnw * Cdal_Taylor
A_Lnw_cos1 = simplify(Matrix([[Poly(fad_Lnw_cosr1[i], un, vn, wn).coeff_monomial([un, vn, wn][j]) for j in range(3)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint

# # Confirmation - If Taylorization is only performed at the end, on the forces, the same result is obtained:
# delta_tn = ttn - tbn
# Cdal_tilde = Cdal + Cdal_dtn * delta_tn
# T_LnwbLnwt_6 = matrix_3dof_to_6dof(T_LnwbLnwt)
# B_mat_Lnw = diag([H, B, B, B**2, B**2, B**2], unpack=True)
# fad_Lnw_cosr1_NLatEnd = S.Half * rho * T_LnwbLnwt_6 * Vn**2  * B_mat_Lnw * Cdal_tilde
# fad_Lnw_cosr1_NLatEnd_Taylor = Matrix([Taylor_polynomial_sympy(fad_Lnw_cosr1[i], [un,vn,wn], [0,0,0],1) for i in range(6)])
# A_Lnw_cos1_NLatEnd = simplify(Matrix([[Poly(fad_Lnw_cosr1_NLatEnd_Taylor[i], un, vn, wn).coeff_monomial([un, vn, wn][j]) for j in range(3)] for i in range(6)])).subs(asin(sin(tbn)), tbn).subs(sqrt(cos(tbn)**2), cos(tbn))  # <-------------------------------------------------- Enter in console to print nicely or do pprint



############################################################################################################################################
# THE FOLLOWING SECTION IS A MATHEMATICAL CONFIRMATION. It proves that using Lnw variables and linearizing w.r.t. un, vn, wn is the same
# as using Gw variables and linearizing w.r.t. u,v,w!
############################################################################################################################################
# Now in EGV - "Expressed with Global Variables" (they are: tb, tt, bb, bt, U, u, v, w. tbn is perhaps necessary for compact presentation):
# The Taylorization will be performed on u,v,w instead. This is more correct since these quantities are always smaller than U, as opposed to un, vn, wn respective to Un.
########################################################################################################################################################################################
U_yz = sqrt(U_y**2 + U_z**2)
V_yz = sqrt(V_y**2 + V_z**2)
V_yz_Taylor = Taylor_polynomial_sympy(V_yz, [u,v,w], [0,0,0],1)
V_yz_Taylor_square = collect(expand_trig(simplify(Taylor_polynomial_sympy(V_yz**2, [u,v,w], [0,0,0],1))), U)
V_yz_Taylor_M = (U*(sin(tb)**2 + cos(bb)**2*cos(tb)**2) + u*(sin(tb)**2 + cos(bb)**2*cos(tb)**2) - v*sin(bb)*cos(bb)*cos(tb) + w*sin(bb)**2*sin(tb)*cos(tb))/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2)  # from Mathematica, with some extra operations in Python
testing_and_comparing_numerical_random_values(V_yz_Taylor, V_yz_Taylor_M)
V_yz_Taylor = V_yz_Taylor_M

tbn_EGV = simplify(asin(U_z / U_yz))
ttn_EGV = asin(V_z / V_yz)

ttn_EGV_Taylor = Taylor_polynomial_sympy(ttn_EGV, [u,v,w], [0,0,0],1)
ttn_EGV_Taylor_M = tbn_EGV + (v*sin(tb)*tan(bb) + w)*Abs(cos(bb))/(U*(sin(tb)**2 + cos(bb)**2*cos(tb)**2))  # from Mathematica
testing_and_comparing_numerical_random_values(ttn_EGV_Taylor, ttn_EGV_Taylor_M)
ttn_EGV_Taylor = ttn_EGV_Taylor_M


delta_tn_EGV = ttn_EGV_Taylor - tbn_EGV

T_LnwbLnwt_EGV = (R_y(ttn_EGV_Taylor)*R_y(-tbn_EGV)).T   # This is wrong if bb = 85deg and bt = 95deg because Lnwt would then invert. Linear theory shall not be used av the vicinity bb ~ +-pi/2.
T_LnwbLnwt_Taylor_EGV = simplify(Matrix([[Taylor_polynomial_sympy(T_LnwbLnwt_EGV[i,j], [u,v,w], [0,0,0],1) for j in range(3)] for i in range(3)]))
T_LnwbLnwt_6_Taylor_EGV = matrix_3dof_to_6dof(T_LnwbLnwt_Taylor_EGV)
Cdal_tilde_Taylor_EGV = Cdal + Cdal_dtn * delta_tn_EGV

fad_Lnw_cosr1_EGV = S.Half * rho * T_LnwbLnwt_6_Taylor_EGV * V_yz_Taylor_square  * B_mat_Lnw * Cdal_tilde_Taylor_EGV
fad_Lnw_cosr1_EGV_Taylor = Matrix([Taylor_polynomial_sympy(fad_Lnw_cosr1_EGV[i], [u,v,w], [0,0,0],1) for i in range(6)])
A_Lnw_cos1_EGV = Matrix([[Poly(fad_Lnw_cosr1_EGV_Taylor[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)])  # <-------------------------------------------------- Enter in console to print nicely or do pprint
A_Lnw_cos1_EGV_simple = simplify(simplify(simplify(A_Lnw_cos1_EGV.subs(cos(tb),costb)).subs(-cos(tb)**2+1, sin(tb)**2)).subs(costb,cos(tb)))  # It was very hard to find out that this could be simplified this much!!!
# testing_and_comparing_numerical_random_values(A_Lnw_cos1_EGV_simple[0,1], 0.5*rho*U*(-2*H*Cd*cos(bb)**2*cos(tb)+(H*Cd_dtn-B*Cl)*Abs(cos(bb))*sin(tb))*tan(bb))
# testing_and_comparing_numerical_random_values(A_Lnw_cos1_EGV_simple[2,1], 0.5*rho*U*(-2*B*Cl*cos(bb)**2*cos(tb)+(B*Cl_dtn+H*Cd)*Abs(cos(bb))*sin(tb))*tan(bb))

# Special case: When theta = 0:
T_LnwbLnwt_6_Taylor_EGV.subs(tb,0)
fad_Lnw_cosr1_EGV_Taylor.subs(tb,0)
A_Lnw_cos1_EGV_simple.subs(tb,0)

# # # # Just making easier to copy paste to Word:
# A_Lnw_cos1_EGV_simple[0,1] / (1/2 * rho * U)
# 2.0*(-0.5*B*Cl*sin(tb)*Abs(cos(bb)) - 1.0*Cd*H*cos(bb)**2*cos(tb) + 0.5*Cd_dtn*H*sin(tb)*Abs(cos(bb)))*tan(bb)
# A_Lnw_cos1_EGV_simple[2,1] / (1/2 * rho * U)
# 2.0*(-1.0*B*Cl*cos(bb)**2*cos(tb) + 0.5*B*Cl_dtn*sin(tb)*Abs(cos(bb)) + 0.5*Cd*H*sin(tb)*Abs(cos(bb)))*tan(bb)
# A_Lnw_cos1_EGV_simple[4,1] / (1/2 * rho * U)
# -2.0*B**2*(1.0*Cm*cos(tb) - 0.5*Cm_dtn*sin(tb)/Abs(cos(bb)))*sin(bb)*cos(bb)
#
# A_Lnw_cos1_EGV_simple[0,2] / (1/2 * rho * U)
# -1.0*B*Cl*Abs(cos(bb)) + 2.0*Cd*H*sin(bb)**2*sin(tb)*cos(tb) + 1.0*Cd_dtn*H*Abs(cos(bb))
# A_Lnw_cos1_EGV_simple[2,2] / (1/2 * rho * U)
# 2.0*B*Cl*sin(bb)**2*sin(tb)*cos(tb) + 1.0*B*Cl_dtn*Abs(cos(bb)) + 1.0*Cd*H*Abs(cos(bb))
# A_Lnw_cos1_EGV_simple[4,2] / (1/2 * rho * U)
# 2.0*B**2*(1.0*Cm*sin(bb)**2*sin(tb)*cos(tb) + 0.5*Cm_dtn*Abs(cos(bb)))

######################
# Other variables
T_LnwbGwb = simplify((T_LsLnwb.T * T_LsGwb).subs(cosbb, cos(bb)).subs(tbn,tbn_EGV))
uvwn_Lnwb_EGV = T_LnwbGwb @ Matrix([u, v, w])  # Expressed only in global variables
un_EGV = uvwn_Lnwb_EGV[0]
vn_EGV = uvwn_Lnwb_EGV[1]
wn_EGV = uvwn_Lnwb_EGV[2]

# #######################################
# # THE WORST ALTERNATIVE - DID NOT WORK:
# # Trying with more intermediate Taylors, and keeping "tbn", so EpGV - "Expressed partially in global coordinates":
# V_yz_Taylor = simplify(Taylor_polynomial_sympy(V_yz, [u,v,w], [0,0,0], 1))
# ttn_EpGV = simplify(ttn.subs([(Un, U_yz), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV)]))
# delta_tn_EpGV = ttn_EpGV - tbn
# delta_tn_Taylor_EpGV = simplify(Taylor_polynomial_sympy(delta_tn_EpGV, [u,v,w], [0,0,0], 1))
# delta_tn_Taylor_simple_EpGV = simplify(delta_tn_Taylor_EpGV.subs(asin(sin(tbn)),tbn).subs(sqrt(cos(tbn)**2),cos(tbn)))
# T_LnwbLnwt_EpGV = (R_y(ttn_EpGV)*R_y(-tbn)).T
# T_LnwbLnwt_Taylor_EpGV = Matrix([[Taylor_polynomial_sympy(T_LnwbLnwt_EpGV[i,j], [u, v, w], [0,0,0], 1) for j in range(3)] for i in range(3)])
# T_LnwbLnwt_Taylor_6_EpGV = matrix_3dof_to_6dof(T_LnwbLnwt_Taylor_EpGV)
# Cdal_tilde_Taylor_EGV = simplify(Matrix([Taylor_polynomial_sympy(Cdal_tilde_Taylor_EGV[i], [u,v,w], [0,0,0],1) for i in range(6)]))
# fad_Lnw_cosr1_EpGV = 0.5 * rho * T_LnwbLnwt_Taylor_6_EpGV * V_yz_Taylor**2  * B_mat_Lnw * Cdal_tilde_Taylor_EGV
# fad_Lnw_cosr1_Taylor_EpGV = Matrix([Taylor_polynomial_sympy(fad_Lnw_cosr1_EpGV[i], [u,v,w], [0,0,0],1) for i in range(6)])
# A_Lnw_cos1_EpGV = Matrix([[Poly(fad_Lnw_cosr1_Taylor_EpGV[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)])  # <-------------------------------------------------- Enter in console to print nicely or do pprint

# It is the same to linearize w.r.t un,vn,wn or u,v,w:
testing_and_comparing_numerical_random_values((tbn_EGV + wn_EGV/U_yz).subs(cosbb,cos(bb)).subs(tbn,tbn_EGV), ttn_EGV_Taylor, n_tests=5)

################################################################
# expression = T_GwbGwt_Taylor_rel[0,1]
# wolfram_mathematica_equation_simplification(expression, form_try='latex')
################################################################

#################################################################
# LOTS OF TRASH - MANY ATTEMPTS WITHOUT SUCCESS OF SIMPLIFICATION

# # Confirmation
# T_LnwbGwb = (R_y(tb) @ R_z(-bb) @ R_y(-tbn)).T  # this is WRONG for cosbb<0

# # When theta = 0:
# simplify(Un_EGV.subs([(tbn,tbn_EGV), (tb,0)]))
# simplify(Vn_EGV.subs([(tbn,tbn_EGV), (tb,0)]))
# simplify(T_LnwbGwb.subs([(tbn,tbn_EGV), (tb,0)]))
# simplify(ttn_Taylor.subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn,tbn_EGV), (tb,0)]))
# Vn_t0 = simplify(Vn_nonlinear_EGV.subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn,tbn_EGV)]).subs(tb,0))
# Vn_t0_2 = simplify(Vn.subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn,tbn_EGV)]).subs(tb,0))
# Vn_t0_3 = simplify(sqrt(V_y**2+V_z**2).subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn,tbn_EGV)]).subs(tb,0))
# Vn_t0_3_Taylor = Taylor_polynomial_sympy(Vn_t0_3, [u,v,w], [0,0,0], 1)
#
# # Testing:
# tb_test = rad(0)
# bb_test = rad(85)
# U_test = 0.0001
# u_test = 0
# v_test = 0
# w_test = 5
# print('theta_bar [deg]: ', deg(tb.subs([(tbn,tbn_EGV), (tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf()))
# print('theta_tilde [deg]: ', deg(tt.subs([(tbn,tbn_EGV), (tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf()))
# print('delta_tn [deg]: ', deg((ttn-tbn).subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn,tbn_EGV), (tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf()))
# print('delta_tn_Taylor [deg]: ', deg(ttn_Taylor.subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn,tbn_EGV), (tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf()))

#
# print(Vn_t0.subs([(tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf())
# print(Vn_t0_2.subs([(tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf())
# print(Vn_t0_3.subs([(tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf())
# print(Vn_t0_3_Taylor.subs([(tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf())
# print(wn_EGV.subs([(tbn,tbn_EGV), (tb,tb_test),(bb,bb_test),(U,U_test),(u,u_test),(v,v_test),(w,w_test)]).evalf())

# # Dead end (Trash):
# T_LnwtGwt = simplify(T_LnwbLnwt.T * T_LnwbGwb * T_GwbGwt_Taylor)
# T_LnwtGwt_Taylor = simplify(Matrix([[Taylor_polynomial_sympy(T_LnwtGwt[i,j], [u,v,w,un,vn,wn], [0,0,0,0,0,0], 1) for j in range(3)] for i in range(3)]).subs(cos(tb),costb)).subs(costb,cos(tb))
# T_LnwtGwt_EGV = T_LnwtGwt.subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn, tbn_EGV)])
# T_LnwtGwt_Taylor_EGV = simplify(Matrix([[Taylor_polynomial_sympy(T_LnwtGwt_EGV[i,j], [u,v,w], [0,0,0], 1) for j in range(3)] for i in range(3)]).subs(cos(tb),costb)).subs(costb,cos(tb))
# Vyz_nonlinear = sqrt(V_y**2+V_z**2)
# Vyz = sqrt(simplify((expand(Vyz_nonlinear**2) + O(u**2) + O(v**2) + O(w**2) + O(u*v) + O(u*w) + O(v*w)).removeO()))
# Vyz_Taylor = simplify(Taylor_polynomial_sympy(Vyz, [u,v,w], [0,0,0],1))
# Uyz = sqrt(U_y**2+U_z**2)
# tbn_EGV = simplify(asin(U_z / Uyz))
# ttn_EGV = simplify(asin(V_z / Vyz))
# ttn_Taylor_EGV = simplify(Taylor_polynomial_sympy(ttn_EGV, [u,v,w], [0,0,0],1).subs(cos(tb),costb)).subs(costb,cos(tb))
# delta_tn = simplify((ttn_Taylor_EGV - tbn_EGV).subs(cos(tb),costb)).subs(costb,cos(tb))
# delta_tn_Taylor = simplify(Taylor_polynomial_sympy(delta_tn, [u,v,w], [0,0,0],1).subs(cos(tb),costb)).subs(costb,cos(tb))

# # Testing values of u,v,w,tb,bb and observing the corresponding values of un, vn, wn, tbn:
u_test = 0
v_test = -3
w_test = 0
tb_test = rad(20)
bb_test = rad(45)
# print('tbn[deg] = ', deg(tbn_EGV.subs([(tb, tb_test), (bb, bb_test)])))
# print('un = ', un_EGV.subs([(u,u_test), (v,v_test), (w, w_test), (tb, tb_test), (bb, bb_test), (tbn, tbn_EGV.subs([(tb, tb_test), (bb, bb_test)]))]))
# print('vn = ', vn_EGV.subs([(u,u_test), (v,v_test), (w, w_test), (tb, tb_test), (bb, bb_test), (tbn, tbn_EGV.subs([(tb, tb_test), (bb, bb_test)]))]))
# print('wn = ', wn_EGV.subs([(u,u_test), (v,v_test), (w, w_test), (tb, tb_test), (bb, bb_test), (tbn, tbn_EGV.subs([(tb, tb_test), (bb, bb_test)]))]))

# # Plotting relation between tb and tbn for different bb:
# import matplotlib.pyplot as plt
# bb_arr = np.append(np.arange(rad(0),rad(90.000),rad(10)), rad(90))
# tb_arr = np.arange(rad(0),rad(45.0001),rad(0.1))
# tbn_mat = np.array([[tbn_EGV.subs([(bb, bb_arr[i]),(tb, tb_arr[j])]).as_real_imag()[0] for j in range(len(tb_arr))] for i in range(len(bb_arr))])
# plt.figure(figsize=(6,3), dpi=300)
# for i in range(len(bb_arr)):
#     plt.plot(deg(tb_arr), deg(np.real(tbn_mat[-i-1])), label=r"$\beta = %s \degree$" % str(round(np.real(deg(bb_arr[-i-1])))))
# plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
# plt.xlabel(r"$\theta\/\/[\degree]$")
# plt.ylabel(r"$\theta_{normal}\/\/[\degree]$")
# plt.tight_layout()

# # VERY SLOW PART BELLOW ############################
# # And in Ls coordinates
# T_LsLnwb_6 = matrix_3dof_to_6dof(T_LsLnwb)
# Cxyz_Taylor_cosr1 = Matrix([0, Cxyz_Taylor[1], Cxyz_Taylor[2], Cxyz_Taylor[3], 0, 0])  # Keeping only the components in the yz-plane
# Cdal_Taylor_cosr1_ELs = T_LsLnwb_6.T * Cxyz_Taylor_cosr1
# Cdal_Taylor_cosr1_ELs[0] = Cdal_Taylor_cosr1_ELs[0] / H * B  # todo: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# fad_Ls_cosr1 = T_LsLnwb_6 * 0.5 * rho * T_LnwbLnwt_Taylor_6 * Vn**2 * B_mat_Lnw * Cdal_Taylor_cosr1_ELs
# fad_Ls_cosr1_Els = fad_Ls_cosr1.subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV), (tbn, tbn_EGV)])
# A_Ls_cos1 = simplify(Matrix([[Poly(fad_Ls_cosr1_Els[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)]))
#
# # IF theta_b is 0, then this becomes simply:
# A_Ls_cos1_tb0 = simplify(A_Ls_cos1.subs(tb, 0))

# # keeping the tbn variable:
# fad_Ls_cosr1_Els_keep_tbn = fad_Ls_cosr1.subs([(Un, Un_EGV), (un, un_EGV), (vn, vn_EGV), (wn, wn_EGV)])
# A_Ls_cos1_keep_tbn = simplify(Matrix([[Poly(fad_Ls_cosr1_Els_keep_tbn[i], u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)] for i in range(6)]))
# # keeping all Lnw variables:
# fad_Ls_cosr1_Els_keep_all = fad_Ls_cosr1
# A_Ls_cos1_keep_all = simplify(Matrix([[Poly(fad_Ls_cosr1_Els_keep_all[i], un, vn, wn).coeff_monomial([un, vn, wn][j]) for j in range(3)] for i in range(6)]))
# VERY SLOW PART ABOVE ############################

############################################################################################################################################
# THE FOLLOWING SECTION IS A MATHEMATICAL CONFIRMATION. It proves that using Lnw variables and linearizing w.r.t. un, vn, wn is the same
# as using Gw variables and linearizing w.r.t. u,v,w!
########################################################################################################################################################################################
# ONLY NORMAL WIND APPROXIMATION - WITH SELF-EXCITED FORCES - (Only the wind in the yz-plane is considered)
########################################################################################################################################################################################
from sympy import trigsimp, collect, factor

zeta = 1-sin(bb)**2*cos(tb)**2
SS = signcosbb
if signcosbb == 'any':
    SS = sign(cos(bb))

# Adopting the relative velocities already in the global reference Gw u,v,w, and not in the projected velocities
V_yz_relrel = sqrt(V_y_relrel**2 + V_z_relrel**2).subs(list_delta_Ls_to_EGw)
V_yz_relrel_Taylor = Taylor_polynomial_sympy(V_yz_relrel,[u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1)
# Same expressions, simplified in Wolfram Mathematica:
V_yz_relrel_power2_Taylor = simplify(Taylor_polynomial_sympy(V_yz_relrel**2,[u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1))
V_yz_relrel_power2_Taylor_M = (U**2+2*U*u_rel)*zeta + 2*U*((w_rel+U*delta_rYv)*sin(bb)*sin(tb) - (v_rel-U*delta_rZw)*cos(bb))*sin(bb)*cos(tb)
testing_and_comparing_numerical_random_values(V_yz_relrel_power2_Taylor, V_yz_relrel_power2_Taylor_M, n_tests=5)
V_yz_relrel_power2_Taylor = V_yz_relrel_power2_Taylor_M

ttn_EGV_rel = asin(V_z_relrel / V_yz_relrel).subs(list_delta_Ls_to_EGw)  # puting a minus delta_rxprime_ELs here is redundant since V_i_relrel already includes the structural motions!
ttn_EGV_rel_Taylor = Taylor_polynomial_sympy(ttn_EGV_rel, [u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15, 1)
ttn_EGV_rel_Taylor_simplest = tbn_EGV + (w_rel+ U*delta_rYv +(v_rel-U*delta_rZw)*sin(tb)*tan(bb))*Abs(cos(bb))/(U*zeta)

testing_and_comparing_numerical_random_values(ttn_EGV_rel_Taylor, ttn_EGV_rel_Taylor_simplest,n_tests=5)
ttn_EGV_rel_Taylor = ttn_EGV_rel_Taylor_simplest

# Special cases
# simplify(ttn_EGV_rel_Taylor.subs(bb, 0))
# simplify(ttn_EGV_rel_Taylor.subs(tb, 0))

delta_tn_EGV_rel = simplify(ttn_EGV_rel_Taylor - tbn_EGV)  # doesn't matter the coordinate system, only difference in theta angles?

T_LsttLnwtt = (R_y(ttn_EGV_rel_Taylor)*R_z(-pi/2*sign(cosbb))).T  # todo: should be sign(cos(btt)), but if too complicated, try both cases of sign(cos(bb)) positive and negative
T_LsttLnwtt_Taylor = simplify(Matrix([[Taylor_polynomial_sympy(T_LsttLnwtt[i,j],[u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15,1) for j in range(3)] for i in range(3)]))

T_LsLnwb_EGw = T_LsLnwb.subs(tbn, tbn_EGV)
T_LnwbLnwt_EGw_rel = T_LsLnwb_EGw.T @ T_LsttLs_EGw.T @ T_LsttLnwtt
T_LnwbLnwt_EGw_rel_Taylor = Matrix([[Taylor_polynomial_sympy(T_LnwbLnwt_EGw_rel[i,j],[u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15,1) for j in range(3)] for i in range(3)])


# T_LnwbLnwt_EGw_rel_Taylor_simple = simplify(T_LnwbLnwt_EGw_rel_Taylor)  # This actually simplifies it considerably

T_LnwbLnwt_EGw_rel_Taylor_simple_01 = -sign(cos(bb))*(delta_rYv*sin(bb)*sin(tb)+delta_rZw*cos(bb))/sqrt(zeta)
T_LnwbLnwt_EGw_rel_Taylor_simple_02 = sign(cos(bb))*(delta_rXu*sin(bb)*cos(tb) + delta_rYv*cos(bb) - delta_rZw*sin(bb)*sin(tb)) - delta_tn_EGV_rel
T_LnwbLnwt_EGw_rel_Taylor_simple_12 = (-delta_rXu*zeta + delta_rYv*sin(bb)*cos(bb)*cos(tb) - delta_rZw*sin(bb)**2*sin(tb)*cos(tb) ) / sqrt(zeta)
T_LnwbLnwt_EGw_rel_Taylor_simple = Matrix([[                                   1, T_LnwbLnwt_EGw_rel_Taylor_simple_01, T_LnwbLnwt_EGw_rel_Taylor_simple_02],
                                           [-T_LnwbLnwt_EGw_rel_Taylor_simple_01,                                   1, T_LnwbLnwt_EGw_rel_Taylor_simple_12],
                                           [-T_LnwbLnwt_EGw_rel_Taylor_simple_02,-T_LnwbLnwt_EGw_rel_Taylor_simple_12,                                   1]])
testing_and_comparing_numerical_random_values(T_LnwbLnwt_EGw_rel_Taylor.subs(cosbb,cos(bb)), T_LnwbLnwt_EGw_rel_Taylor_simple)


# wolfram_mathematica_equation_simplification(T_LnwbLnwt_EGw_rel_Taylor_simple)
T_LnwbLnwt_rel_Taylor_simplest_EGw = T_LnwbLnwt_EGw_rel_Taylor_simple  # todo: make new one from


# todo: the one below is wrong, missing sign(cos(bb)) in the T matrix.
# T_LnwbLnwt_rel_Taylor_simplest_ELs = Matrix([
# [                                                                                                                                                                                                                                                                  1, (delta_ry*sin(tb) - delta_rz*cos(tb)*Abs(cos(bb)))/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2), -delta_rx + 2*(U*delta_ry*sin(bb)*cos(tb)**2 + delta_z_d*cos(tb) - w + (U*delta_rx - delta_y_d*sin(tb))*cos(bb) + (U*delta_rz*cos(tb) - v + (U*delta_rx*sin(tb) - delta_y_d)*sin(bb))*sin(tb)*tan(bb))*Abs(cos(bb))/(U*(-sin(bb)**2*cos(2*tb) + cos(bb)**2 + 1))],
# [                                                                                                                                                                       (-delta_ry*sin(tb) + delta_rz*cos(tb)*Abs(cos(bb)))/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2),                                                                                           1,                                                                                                                                                                     -(delta_ry*cos(tb)*Abs(cos(bb)) + delta_rz*sin(tb))/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2)],
# [delta_rx + 2*(-U*delta_ry*sin(bb)*cos(tb)**2 - delta_z_d*cos(tb) + w + (-U*delta_rx + delta_y_d*sin(tb))*cos(bb) + (-U*delta_rz*cos(tb) + v + (-U*delta_rx*sin(tb) + delta_y_d)*sin(bb))*sin(tb)*tan(bb))*Abs(cos(bb))/(U*(-sin(bb)**2*cos(2*tb) + cos(bb)**2 + 1)), (delta_ry*cos(tb)*Abs(cos(bb)) + delta_rz*sin(tb))/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2),                                                                                                                                                                                                                                                                1]])
# testing_and_comparing_numerical_random_values(expression_1=T_LnwbLnwt_rel_Taylor_simplest_ELs,expression_2=T_LnwbLnwt_ELs_rel_Taylor)

T_LnwbLnwt_EGw_rel_Taylor = T_LnwbLnwt_rel_Taylor_simplest_EGw
T_LnwbLnwt_6_EGw_rel = matrix_3dof_to_6dof(T_LnwbLnwt_EGw_rel_Taylor)


T_LsLnwb_EGw_6 = matrix_3dof_to_6dof(T_LsLnwb_EGw)
# Cdal_ELs = simplify(T_LsLnwb_EGw_6.T @ Matrix([0,Cy,Cz,Crx,0,0]))
# Cdal_dtn_ELs = simplify(T_LsLnwb_EGw_6.T @ Matrix([0,Cy_dtn,Cz_dtn,Crx_dtn,0,0]))
# list_Cdal_to_Cxyz = list(zip(Cdal,Cdal_ELs))
# list_Cdal_dtn_to_Cxyz_dtn = list(zip(Cdal_dtn,Cdal_dtn_ELs))

Cxyz_dtn = Matrix([0, Cy_dtn, Cz_dtn, Crx_dtn, 0, 0])
Cxyz_ELnw = simplify(T_LsLnwb_EGw_6 @ Matrix([Cd,0,Cl,0,Cm,0]))
Cxyz_dtn_ELnw = simplify(T_LsLnwb_EGw_6 @ Matrix([Cd_dtn,0,Cl_dtn,0,Cm_dtn,0]))
list_Cxyz_to_Cdal = list(zip(Cxyz,Cxyz_ELnw))
list_Cxyz_dtn_to_Cdal_dtn = list(zip(Cxyz_dtn,Cxyz_dtn_ELnw))

Cdal_tilde_EGV_rel = Cdal + Cdal_dtn * delta_tn_EGV_rel
Cdal_tilde_EGV_rel_Taylor = Matrix([Taylor_polynomial_sympy(Cdal_tilde_EGV_rel[i],[u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15,1) for i in range(6)])
# Cdal_tilde_ELs_rel = Cdal_ELs + Cdal_dtn_ELs * delta_tn_ELs_rel
# Cdal_tilde_ELs_rel_Taylor = Matrix([Taylor_polynomial_sympy(Cdal_tilde_ELs_rel[i],[u,v,w]+list(delta_Ls)+list(delta_Ls_d), [0]*15,1) for i in range(6)])

fad_Lnw_cosr1_EGV_rel = S.Half * rho * T_LnwbLnwt_6_EGw_rel * V_yz_relrel_power2_Taylor * B_mat_Lnw * Cdal_tilde_EGV_rel_Taylor
fad_Lnw_cosr1_EGV_Taylor_rel = Matrix([Taylor_polynomial_sympy(fad_Lnw_cosr1_EGV_rel[i],[u,v,w]+list(delta_Gw)+list(delta_Gw_d), [0]*15,1) for i in range(6)])
fad_Lnw_cosr1_EGV_Taylor_rel_simple = simplify(fad_Lnw_cosr1_EGV_Taylor_rel).subs(cosbb,cos(bb))

# fad_Ls_cosr1_ELs_rel = S.Half * rho * T_LsLnwb_EGw_6 * T_LnwbLnwt_6_ELs_rel * V_yz_relrel_power2_Taylor * B_mat * Cdal_tilde_ELs_rel_Taylor
# fad_Ls_cosr1_ELs_Taylor_rel = Matrix([Taylor_polynomial_sympy(fad_Ls_cosr1_ELs_rel[i],[u,v,w]+list(delta_Ls)+list(delta_Ls_d), [0]*15,1) for i in range(6)])
# fad_Ls_cosr1_ELs_Taylor_rel_simple = simplify(fad_Ls_cosr1_ELs_Taylor_rel)

# A_Ls_cos1_ELs_rel = simplify(Matrix([[Poly(fad_Ls_cosr1_ELs_Taylor_rel_simple[i],[u,v,w]+list(delta_Ls)[3:]+list(delta_Ls_d)[:3]).coeff_monomial(([u,v,w]+list(delta_Ls)[3:]+list(delta_Ls_d)[:3])[j]) for j in range(9)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
A_LnwGw_cos1_rel = simplify(Matrix([[Poly(fad_Lnw_cosr1_EGV_Taylor_rel_simple[i],[u,v,w]+list(delta_Gw)[3:]+list(delta_Gw_d)[:3]).coeff_monomial(([u,v,w]+list(delta_Gw)[3:]+list(delta_Gw_d)[:3])[j]) for j in range(9)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint

normalization = diag(diag(1,1,1) * (S.Half*rho*U), diag(1,1,1) * (S.Half*rho*U**2), diag(1,1,1) * (S.Half*rho*U))

A_LnwGw_cos1_rel_simple = Matrix([
[   2*Cd*H*zeta, (-SS*B*Cl*sin(tb) - 2*Cd*H*cos(bb)*cos(tb) + SS*Cd_dtn*H*sin(tb))*sin(bb), -SS*B*Cl*cos(bb) + Cd*H*sin(bb)**2*sin(2*tb) + SS*Cd_dtn*H*cos(bb),  SS*B*Cl*sin(bb)*cos(tb)*zeta, -SS*B*Cl*sin(bb)**2*cos(bb)*cos(tb)**2 + Cd*H*sin(bb)**2*sin(2*tb) + SS*Cd_dtn*H*cos(bb), (SS*B*Cl*sin(bb)**2*sin(tb)*cos(tb)**2 + 2*Cd*H*cos(bb)*cos(tb) - SS*Cd_dtn*H*sin(tb))*sin(bb),    -2*Cd*H*zeta, (SS*B*Cl*sin(tb) + 2*Cd*H*cos(bb)*cos(tb) - SS*Cd_dtn*H*sin(tb))*sin(bb),  SS*B*Cl*cos(bb) - Cd*H*sin(bb)**2*sin(2*tb) - SS*Cd_dtn*H*cos(bb)],
[             0,                                                                         0,                                                                  0,               -B*Cl*zeta**1.5,                              (B*Cl*cos(bb)*cos(tb) + SS*Cd*H*sin(tb))*sqrt(zeta)*sin(bb),                                (-B*Cl*sin(bb)**2*sin(tb)*cos(tb) + SS*Cd*H*cos(bb))*sqrt(zeta),               0,                                                                        0,                                                                  0],
[   2*B*Cl*zeta, (-2*B*Cl*cos(bb)*cos(tb) + SS*B*Cl_dtn*sin(tb) + SS*Cd*H*sin(tb))*sin(bb),  B*Cl*sin(bb)**2*sin(2*tb) + SS*B*Cl_dtn*cos(bb) + SS*Cd*H*cos(bb), -SS*Cd*H*sin(bb)*cos(tb)*zeta,  B*Cl*sin(bb)**2*sin(2*tb) + SS*B*Cl_dtn*cos(bb) + SS*Cd*H*sin(bb)**2*cos(bb)*cos(tb)**2, (2*B*Cl*cos(bb)*cos(tb) - SS*B*Cl_dtn*sin(tb) - SS*Cd*H*sin(bb)**2*sin(tb)*cos(tb)**2)*sin(bb),   - 2*B*Cl*zeta, (2*B*Cl*cos(bb)*cos(tb) - SS*B*Cl_dtn*sin(tb) - SS*Cd*H*sin(tb))*sin(bb), -B*Cl*sin(bb)**2*sin(2*tb) - SS*B*Cl_dtn*cos(bb) - SS*Cd*H*cos(bb)],
[             0,                                                                         0,                                                                  0,                             0,                                                   -SS*B**2*Cm*sqrt(zeta)*sin(bb)*sin(tb),                                                                 -SS*B**2*Cm*sqrt(zeta)*cos(bb),               0,                                                                        0,                                                                  0],
[2*B**2*Cm*zeta,                  B**2*(-2*Cm*cos(bb)*cos(tb) + SS*Cm_dtn*sin(tb))*sin(bb),                 B**2*(Cm*sin(bb)**2*sin(2*tb) + SS*Cm_dtn*cos(bb)),                             0,                                       B**2*(Cm*sin(bb)**2*sin(2*tb) + SS*Cm_dtn*cos(bb)),                                        B**2*(2*Cm*cos(bb)*cos(tb) - SS*Cm_dtn*sin(tb))*sin(bb), -2*B**2*Cm*zeta,                  B**2*(2*Cm*cos(bb)*cos(tb) - SS*Cm_dtn*sin(tb))*sin(bb),                -B**2*(Cm*sin(bb)**2*sin(2*tb) + SS*Cm_dtn*cos(bb))],
[             0,                                                                         0,                                                                  0,             B**2*Cm*zeta**1.5,                                              -B**2*Cm*sqrt(zeta)*sin(bb)*cos(bb)*cos(tb),                                                  B**2*Cm*sqrt(zeta)*sin(bb)**2*sin(tb)*cos(tb),               0,                                                                        0,                                                                  0]]) @ normalization
testing_and_comparing_numerical_random_values(A_LnwGw_cos1_rel_simple, A_LnwGw_cos1_rel, n_tests=5)
###################################################################################################
# And now the beautiful final convertion from LnwGw (linearized for u,v,w), to Lnw (same as LnwLnw)
###################################################################################################
A_LnwLnw = A_LnwGw_cos1_rel_simple[:,:3]  @ T_LnwbGwb.T
A_LnwLnw = A_LnwLnw.subs(tbn, tbn_EGV).subs(cosbb,cos(bb)).subs(sign(cos(bb)), Abs(cos(bb))/cos(bb))
# wolfram_mathematica_equation_simplification(A_LnwLnw)
A_LnwLnw_simplest = S.Half * rho * U_yz * Matrix([
                           [   2*Cd*H, 0, -B*Cl + Cd_dtn*H],
                           [        0, 0,                0],
                           [   2*B*Cl, 0,  B*Cl_dtn + Cd*H],
                           [        0, 0,                0],
                           [2*B**2*Cm, 0,      B**2*Cm_dtn],
                           [        0, 0,                0]])
testing_and_comparing_numerical_random_values(A_LnwLnw, A_LnwLnw_simplest)
A_LnwLnw = A_LnwLnw_simplest


A_delta_LnwLnw = (A_LnwGw_cos1_rel_simple[:,3:6] / (S.Half*rho*U**2)) @ T_LnwbGwb.T
A_delta_LnwLnw = A_delta_LnwLnw.subs(tbn, tbn_EGV).subs(cosbb,cos(bb)).subs(sign(cos(bb)), Abs(cos(bb))/cos(bb))
# wolfram_mathematica_equation_simplification(A_delta_LnwLnw)  # this needs to be done separately for cosbb<0 and >0, and some extra manual simplification work is needed.


A_delta_LnwLnw_simplest = S.Half*rho*U**2 * Matrix([
[ SS*(B*Cl - Cd_dtn*H)*sqrt(zeta)*sin(bb)*cos(tb),    Cd_dtn*H*zeta,    SS*2*Cd*H*sqrt(zeta)*sin(bb)*cos(tb)],
[                                      -B*Cl*zeta,                0,                               Cd*H*zeta],
[-SS*(B*Cl_dtn + Cd*H)*sqrt(zeta)*sin(bb)*cos(tb),    B*Cl_dtn*zeta,    SS*2*B*Cl*sqrt(zeta)*sin(bb)*cos(tb)],
[                                               0,                0,                           -B**2*Cm*zeta],
[      -SS*B**2*Cm_dtn*sqrt(zeta)*sin(bb)*cos(tb), B**2*Cm_dtn*zeta, SS*2*B**2*Cm*sqrt(zeta)*sin(bb)*cos(tb)],
[                                    B**2*Cm*zeta,                0,                                       0]])
testing_and_comparing_numerical_random_values((S.Half*rho*U**2) * A_delta_LnwLnw, A_delta_LnwLnw_simplest, n_tests=5)

simplify(A_delta_LnwLnw_simplest.subs(U, U_yz/sqrt(zeta)) / (S.Half*rho*U_yz**2))





# # wolfram_mathematica_equation_simplification(simplify(expand_trig(simplify(A_LnwGw_cos1_rel_simple[:,3:6].subs(tb,0) / (S.Half*rho*U**2)).subs(sign(cos(bb)),cos(bb)/Abs(cos(bb))))))
# Kse_LnwGw_cos1_rel_simple = S.Half*rho*U**2 * Matrix([
# [ B*Cl*sin(bb)*cos(bb)*Abs(cos(bb)), -B*Cl*sin(bb)**2*Abs(cos(bb)) + Cd_dtn*H*Abs(cos(bb)),    2*Cd*H*sin(bb)*cos(bb)],
# [     -B*Cl*cos(bb)**2*Abs(cos(bb)),                     B*Cl*sin(bb)*cos(bb)*Abs(cos(bb)),           Cd*H*cos(bb)**2],
# [-Cd*H*sin(bb)*cos(bb)*Abs(cos(bb)),  B*Cl_dtn*Abs(cos(bb)) + Cd*H*sin(bb)**2*Abs(cos(bb)),    2*B*Cl*sin(bb)*cos(bb)],
# [                                 0,                                                     0,       -B**2*Cm*cos(bb)**2],
# [                                 0,                              B**2*Cm_dtn*Abs(cos(bb)), 2*B**2*Cm*sin(bb)*cos(bb)],
# [   B**2*Cm*cos(bb)**2*Abs(cos(bb)),                 -B**2*Cm*sin(bb)*cos(bb)*Abs(cos(bb)),                         0]])
# # testing_and_comparing_numerical_random_values(A_LnwGw_cos1_rel_simple[:,3:6].subs(tb,0), Kse_LnwGw_cos1_rel_simple)



########################################################################################################################################################################################
# NOW IN Lnw COORDINATES
########################################################################################################################################################################################
# ONLY NORMAL WIND APPROXIMATION - MOTION-INDUCED FORCES INCLUDED - MOVING BRIDGE
########################################################################################################################################################################################
T_LnwbGwb = simplify((T_LsLnwb.T * T_LsGwb).subs(cosbb, cos(bb)).subs(tbn,tbn_EGV).subs(sign(cos(bb)), Abs(cos(bb))/cos(bb)))
T_LnwbGwb_6 = matrix_3dof_to_6dof(T_LnwbGwb)
# delta_rD, delta_rA, delta_rL = T_LnwbGwb @ Matrix([delta_rXu, delta_rYv, delta_rZw])

delta_Gw_ELnw = T_LnwbGwb_6.T @ delta_Lnw
delta_Gw_d_ELnw = T_LnwbGwb_6.T @ delta_Lnw_d
list_delta_Gw_to_ELnw = list(zip(delta_Gw, delta_Gw_ELnw))
list_delta_Gw_d_to_ELnw = list(zip(delta_Gw_d, delta_Gw_d_ELnw))
a_Gw = Matrix([u,v,w])
a_Lnw = Matrix([un,vn,wn])
a_Gw_ELnw = T_LnwbGwb.T @ a_Lnw
list_uvw_to_Eunvnwn = list(zip(a_Gw, a_Gw_ELnw))

# Re-writing V_yz_relrel_power2_Taylor
V_yz_relrel_power2_Taylor_ELnw = V_yz_relrel_power2_Taylor.subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn)
# wolfram_mathematica_equation_simplification(V_yz_relrel_power2_Taylor_ELnw)
U_EUyz = Un/sqrt(zeta)
V_yz_relrel_power2_Taylor_ELnw_M = Un*(Un + 2*(un-delta_D_d) + SS*2*U*delta_rL*cos(tb)*sin(bb))
testing_and_comparing_numerical_random_values(V_yz_relrel_power2_Taylor_ELnw , V_yz_relrel_power2_Taylor_ELnw_M.subs(Un,U_yz).subs(un, un_EGV), n_tests=5)
V_yz_relrel_power2_Taylor_ELnw = V_yz_relrel_power2_Taylor_ELnw_M

# Re-writing delta_tn_EGV_rel
delta_tn_EGV_rel_ELnw = delta_tn_EGV_rel.subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn)
# wolfram_mathematica_equation_simplification(expression=delta_tn_EGV_rel_ELnw)
delta_tn_EGV_rel_ELnw_M = (Un*delta_rA - SS*U*delta_rD*sin(bb)*cos(tb) - delta_L_d + wn)/Un
testing_and_comparing_numerical_random_values(delta_tn_EGV_rel_ELnw, delta_tn_EGV_rel_ELnw_M.subs([(Un,U_yz), (un, un_EGV),(vn, vn_EGV), (wn, wn_EGV)]), n_tests=5)
delta_tn_EGV_rel_ELnw = delta_tn_EGV_rel_ELnw_M

# Re-writing T_LnwLnwtt
T_LnwbLnwt_EGw_rel_Taylor_simple_ELnw = T_LnwbLnwt_EGw_rel_Taylor_simple.subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn)
# wolfram_mathematica_equation_simplification(expression=T_LnwbLnwt_EGw_rel_Taylor_simple_ELnw.subs(sign(cos(bb)), cos(bb)/Abs(cos(bb))))
T_LnwbLnwt_EGw_rel_Taylor_simple_ELnw_M = Matrix([
[                                 1, -delta_rL, -delta_tn_EGV_rel_ELnw_M + delta_rA],
[                          delta_rL,         1,                           -delta_rD],
[delta_tn_EGV_rel_ELnw_M - delta_rA,  delta_rD,                                   1]])
testing_and_comparing_numerical_random_values(T_LnwbLnwt_EGw_rel_Taylor_simple_ELnw, T_LnwbLnwt_EGw_rel_Taylor_simple_ELnw_M.subs([(Un,U_yz), (un, un_EGV),(vn, vn_EGV), (wn, wn_EGV)]), n_tests=5)
T_LnwbLnwt_EGw_rel_Taylor_simple_ELnw = T_LnwbLnwt_EGw_rel_Taylor_simple_ELnw_M


########################################################################################################################################################################################
# Axial contribution - Static bridge
########################################################################################################################################################################################
# V_Axial = -V_x
# V_Axial_linear = sqrt(simplify((expand(V_Axial**2) + O(u**2) + O(v**2) + O(w**2) + O(u*v) + O(u*w) + O(v*w)).removeO()))
# fad_Axial_sinr = S.Half * rho * V_Axial_linear**2 * Ca
# A_Axial_sinr = simplify(Matrix([Poly(fad_Axial_sinr, u, v, w).coeff_monomial([u, v, w][j]) for j in range(3)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
########################################################################################################################################################################################
# Axial contribution - Motion-dependent forces
########################################################################################################################################################################################
# # ASSUMPTION: sign(U_x) == sign(V_x) == sign(V_x_relrel)
V_x_square_Taylor_ELnw = simplify(Taylor_polynomial_sympy(V_x*Abs(V_x), [u,v,w],[0,0,0],1))  # "directional" square
V_x_square_Taylor_ELnw_simple = U_x*Abs(U_x) + 2*Abs(U_x)*(V_x-U_x)
testing_and_comparing_numerical_random_values(V_x_square_Taylor_ELnw, V_x_square_Taylor_ELnw_simple)
V_x_relrel_square_Taylor = simplify(Taylor_polynomial_sympy(V_x_relrel*Abs(V_x_relrel), [u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d], [0]*9, 1))
V_x_relrel_square_Taylor_simple = U_x*Abs(U_x) + 2*Abs(U_x)*((V_x-U_x-delta_x_d) + U_y*delta_rz - U_z*delta_ry)
testing_and_comparing_numerical_random_values(V_x_relrel_square_Taylor, V_x_relrel_square_Taylor_simple)
fad_Ls_axial = matrix_3dof_to_6dof(T_LsttLs_ELs).T * Matrix([S.Half*rho*B*Cx*(V_x_relrel_square_Taylor_simple),0,0,0,0,0])
fad_Ls_axial_Taylor = Matrix([Taylor_polynomial_sympy(fad_Ls_axial[i], [u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d], [0]*9, 1) for i in range(6)])
A_Ls_axial = simplify(Matrix([[Poly(fad_Ls_axial_Taylor[i],[u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d]).coeff_monomial([u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d][j]) for j in range(9)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
A_Ls_axial_ux_vy_wz = simplify(A_Ls_axial[:,:3] @ T_LsGwb.T)
A_Ls_axial_delta = expand_trig(A_Ls_axial[:,3:6])
A_Ls_axial_delta_d = A_Ls_axial[:,6:9]
A_Ls_axial_delta_confirm = S.Half * rho * B * Abs(U_x) * Matrix([
[0, -2*U_z*Cx, 2*U_y*Cx],
[0,         0,   U_x*Cx],
[0,   -U_x*Cx,        0],
[0,         0,        0],
[0,         0,        0],
[0,         0,        0]])
testing_and_comparing_numerical_random_values(A_Ls_axial_delta, A_Ls_axial_delta_confirm)



# V_x_square_Taylor_ELnw = simplify(Taylor_polynomial_sympy(V_x**2, [u,v,w],[0,0,0],1))  # "directional" square
# V_x_square_Taylor_ELnw_simple = U_x**2 + 2*U_x*(V_x-U_x)
# testing_and_comparing_numerical_random_values(V_x_square_Taylor_ELnw, V_x_square_Taylor_ELnw_simple)
# V_x_relrel_square_Taylor = simplify(Taylor_polynomial_sympy(V_x_relrel**2, [u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d], [0]*9, 1))
# V_x_relrel_square_Taylor_simple = U_x**2 + 2*U_x*((V_x-U_x-delta_x_d) + U_y*delta_rz - U_z*delta_ry)
# testing_and_comparing_numerical_random_values(V_x_relrel_square_Taylor, V_x_relrel_square_Taylor_simple)
# fad_Ls_axial = matrix_3dof_to_6dof(T_LsttLs_ELs).T * Matrix([S.Half*rho*Ca*(V_x_relrel_square_Taylor_simple),0,0,0,0,0])
# fad_Ls_axial_Taylor = Matrix([Taylor_polynomial_sympy(fad_Ls_axial[i], [u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d], [0]*9, 1) for i in range(6)])
# A_Ls_axial = simplify(Matrix([[Poly(fad_Ls_axial_Taylor[i],[u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d]).coeff_monomial([u,v,w,delta_rx,delta_ry,delta_rz,delta_x_d,delta_y_d,delta_z_d][j]) for j in range(9)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# A_Ls_axial_ux_vy_wz = simplify(A_Ls_axial[:,:3] @ T_LsGwb.T)
# A_Ls_axial_delta = expand_trig(A_Ls_axial[:,3:6])
# A_Ls_axial_delta_d = A_Ls_axial[:,6:9]
# A_Ls_axial_delta_confirm = S.Half * rho * U_x * Matrix([
# [0, -2*U_z*Ca, 2*U_y*Ca],
# [0,         0,   U_x*Ca],
# [0,   -U_x*Ca,        0],
# [0,         0,        0],
# [0,         0,        0],
# [0,         0,        0]])
# testing_and_comparing_numerical_random_values(A_Ls_axial_delta, A_Ls_axial_delta_confirm)
#















#
#
#
#
# V_x_relrel_square_Taylor_ELnw = simplify(Taylor_polynomial_sympy(V_x_relrel**2, [u,v,w],[0,0,0],1))
#
#
# fad_Ls_A = matrix_3dof_to_6dof(T_LsttLs_EGw).T * S.Half * rho * Matrix([Ca,0,0,0,0,0]) * (V_x_relrel_square_ELnw)
# fad_Ls_A_ELnw = fad_Ls_A.subs(list_delta_Ls_to_EGw).subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn).subs(sign(cos(bb)),cos(bb)/Abs(cos(bb)))
# fad_Ls_A_ELnw_Taylor = Matrix([Taylor_polynomial_sympy(fad_Ls_A_ELnw[i], [un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d], [0]*9, 1) for i in range(6)])
# A_Ls_A_ELnw = simplify(Matrix([[Poly(fad_Ls_A_ELnw_Taylor[i],[un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d]).coeff_monomial([un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d][j]) for j in range(9)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# # wolfram_mathematica_equation_simplification(expression=A_Ls_A_ELnw)
# A_Ls_A_ELnw_M = Matrix([
# [0, Ca*U*rho*cos(tb)*tan(bb)*Abs(cos(bb)), 0,                                                                                       0, 0,                       -Ca*U**2*rho*sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2)*cos(tb)*tan(bb)*Abs(cos(bb)), 0, -Ca*U*rho*cos(tb)*tan(bb)*Abs(cos(bb)), 0],
# [0,                                     0, 0,  0.5*Ca*U**2*rho*sin(bb)**2*sin(tb)*cos(tb)**2/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2), 0,              0.5*Ca*U**2*rho*sin(bb)**2*cos(tb)**3*Abs(cos(bb))/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2), 0,                                      0, 0],
# [0,                                     0, 0, -0.5*Ca*U**2*rho*sin(bb)**2*cos(bb)*cos(tb)**3/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2), 0, 0.5*Ca*U**2*rho*sin(bb)*sin(tb)*cos(tb)**2*tan(bb)*Abs(cos(bb))/sqrt(sin(tb)**2 + cos(bb)**2*cos(tb)**2), 0,                                      0, 0],
# [0,                                     0, 0,                                                                                       0, 0,                                                                                                        0, 0,                                      0, 0],
# [0,                                     0, 0,                                                                                       0, 0,                                                                                                        0, 0,                                      0, 0],
# [0,                                     0, 0,                                                                                       0, 0,                                                                                                        0, 0,                                      0, 0]])
# testing_and_comparing_numerical_random_values(A_Ls_A_ELnw, A_Ls_A_ELnw_M)
# A_Ls_A_ELnw = A_Ls_A_ELnw_M











# # todo: TRASH
# un_tt = un - delta_D_d
# vn_tt = vn - delta_A_d
# wn_tt = wn - delta_L_d
# U_yz_tt = sqrt((Un+un_tt)**2 + wn_tt**2)
# U_Lnwb_tt = Matrix([Un+un_tt, -U_x+vn_tt, wn_tt])
#


# ttn_rel = asin(U_Lnwb_tt[2] / ...

#
#
# # TODO: ATTENTION THE LINEARIZATION IS NOT VALID IN THE VICINITY OF BETA=+-90deg BECAUSE OF THE ABRUPT CHANGE IN THE A-axes THERE.
# # Within the linear SS = SSt = SStt assumption:
# U_A = -SS*U_x
# V_A = -SS*V_x
# V_A_relrel = -SS*V_x_relrel
# V_A_AbsVa_Taylor_ELnw = simplify(Taylor_polynomial_sympy(V_A*Abs(V_A), [u,v,w],[0,0,0],1))  # "directional" square
# wolfram_mathematica_equation_simplification(expression=replace_sign(V_A_AbsVa_Taylor_ELnw))
#
# V_A_AbsVa_Taylor_ELnw_simple = U*(2*v - (2*w*sin(tb) - (U + 2*u)*cos(tb))*tan(bb))*cos(bb)**2*cos(tb)*Abs(tan(bb))
# testing_and_comparing_numerical_random_values(V_A_AbsVa_Taylor_ELnw, V_A_AbsVa_Taylor_ELnw_simple)
#
# V_A_relrel_ELnw = V_A_relrel.subs(list_delta_Ls_to_EGw).subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn)
# V_A_relrel_AbsVa_Taylor_ELnw = Taylor_polynomial_sympy(V_A_relrel_ELnw*Abs(V_A_relrel_ELnw), [un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d], [0]*9, 1)
# wolfram_mathematica_equation_simplification(expression=replace_sign(V_A_relrel_AbsVa_Taylor_ELnw))
#
#
# V_A_relrel_AbsVa_Taylor_ELnw_M = U_A**2 + 2*U_A*((vn-delta_A_d) - U_yz*delta_rL)
# testing_and_comparing_numerical_random_values(V_A_relrel_AbsVa_Taylor_ELnw, V_A_relrel_AbsVa_Taylor_ELnw_M)
# V_A_relrel_AbsVa_Taylor_ELnw = V_A_relrel_AbsVa_Taylor_ELnw_M
#
# fad_Lnw_A = matrix_3dof_to_6dof(T_LnwbLnwt_EGw_rel_Taylor) * Matrix([0,S.Half*rho*Ca*(V_A_relrel_AbsVa_Taylor_ELnw),0,0,0,0])
# fad_Lnw_A_ELnw = fad_Lnw_A.subs(list_delta_Ls_to_EGw).subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn).subs(sign(cos(bb)),cos(bb)/Abs(cos(bb)))
# fad_Lnw_A_ELnw_Taylor = Matrix([Taylor_polynomial_sympy(fad_Lnw_A_ELnw[i], [un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d], [0]*9, 1) for i in range(6)])
# A_Lnw_A_ELnw = simplify(Matrix([[Poly(fad_Lnw_A_ELnw_Taylor[i],[un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d]).coeff_monomial([un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d][j]) for j in range(9)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# # wolfram_mathematica_equation_simplification(expression=A_Lnw_A_ELnw)
# A_Lnw_A_ELnw_M = Matrix([
# [0,          0, 0,                 0, 0, -0.5*Ca*U_A**2*rho, 0,           0, 0],
# [0, Ca*U_A*rho, 0,                 0, 0,   -Ca*U_A*U_yz*rho, 0, -Ca*U_A*rho, 0],
# [0,          0, 0, 0.5*Ca*U_A**2*rho, 0,                  0, 0,           0, 0],
# [0,          0, 0,                 0, 0,                  0, 0,           0, 0],
# [0,          0, 0,                 0, 0,                  0, 0,           0, 0],
# [0,          0, 0,                 0, 0,                  0, 0,           0, 0]])
# testing_and_comparing_numerical_random_values(A_Lnw_A_ELnw , A_Lnw_A_ELnw_M)
#
#
#
#
#
#
# # TODO: ATTENTION THIS IS WRONG. U_A**2 IS ALWAYS POSITIVE SO THERE ARE NO CASES WHERE THE FORCE F_A IS NEGATIVE, WHICH IS WRONG
# # Within the linear SS = SSt = SStt assumption:
# U_A = -SS*U_x
# V_A = -SS*V_x
# V_A_relrel = -SS*V_x_relrel
# V_A_square_Taylor_ELnw = simplify(Taylor_polynomial_sympy(V_A**2, [u,v,w],[0,0,0],1))
# # wolfram_mathematica_equation_simplification(expression=V_A_square_Taylor_ELnw.subs(sign(sin(bb)*cos(tb)),sin(bb)*cos(tb)/Abs(sin(bb)*cos(tb))).subs(sign(cos(bb)),cos(bb)/Abs(cos(bb))))
# V_A_square_Taylor_ELnw_simple = -U_x*(-U_x + 2*u*sin(bb)*cos(tb) + 2*v*cos(bb) - 2*w*sin(bb)*sin(tb))
# V_A_square_Taylor_ELnw_simple = U_x**2 - 2*U_x*SS*vn_EGV
# V_A_square_Taylor_ELnw_simple = U_A**2 + 2*U_A*vn_EGV
# testing_and_comparing_numerical_random_values(V_A_square_Taylor_ELnw, V_A_square_Taylor_ELnw_simple)
#
# V_A_relrel_ELnw = V_A_relrel.subs(list_delta_Ls_to_EGw).subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn)
# V_A_relrel_square_Taylor_ELnw = Taylor_polynomial_sympy(V_A_relrel_ELnw**2, [un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d], [0]*9, 1)
# # wolfram_mathematica_equation_simplification(V_A_relrel_square_Taylor_ELnw.subs(sign(cos(bb)), cos(bb)/Abs(cos(bb))))
# V_A_relrel_square_Taylor_ELnw_M = U_A**2 + 2*U_A*((vn-delta_A_d) - U_yz*delta_rL)
# testing_and_comparing_numerical_random_values(V_A_relrel_square_Taylor_ELnw, V_A_relrel_square_Taylor_ELnw_M)
# V_A_relrel_square_Taylor_ELnw = V_A_relrel_square_Taylor_ELnw_M
#
# fad_Lnw_A = matrix_3dof_to_6dof(T_LnwbLnwt_EGw_rel_Taylor) * Matrix([0,S.Half*rho*Ca*(V_A_relrel_square_Taylor_ELnw),0,0,0,0])
# fad_Lnw_A_ELnw = fad_Lnw_A.subs(list_delta_Ls_to_EGw).subs(list_delta_Gw_to_ELnw+list_delta_Gw_d_to_ELnw+list_uvw_to_Eunvnwn).subs(sign(cos(bb)),cos(bb)/Abs(cos(bb)))
# fad_Lnw_A_ELnw_Taylor = Matrix([Taylor_polynomial_sympy(fad_Lnw_A_ELnw[i], [un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d], [0]*9, 1) for i in range(6)])
# A_Lnw_A_ELnw = simplify(Matrix([[Poly(fad_Lnw_A_ELnw_Taylor[i],[un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d]).coeff_monomial([un,vn,wn,delta_rD,delta_rA,delta_rL,delta_D_d,delta_A_d,delta_L_d][j]) for j in range(9)] for i in range(6)]))  # <-------------------------------------------------- Enter in console to print nicely or do pprint
# # wolfram_mathematica_equation_simplification(expression=A_Lnw_A_ELnw)
# A_Lnw_A_ELnw_M = Matrix([
# [0,          0, 0,                 0, 0, -0.5*Ca*U_A**2*rho, 0,           0, 0],
# [0, Ca*U_A*rho, 0,                 0, 0,   -Ca*U_A*U_yz*rho, 0, -Ca*U_A*rho, 0],
# [0,          0, 0, 0.5*Ca*U_A**2*rho, 0,                  0, 0,           0, 0],
# [0,          0, 0,                 0, 0,                  0, 0,           0, 0],
# [0,          0, 0,                 0, 0,                  0, 0,           0, 0],
# [0,          0, 0,                 0, 0,                  0, 0,           0, 0]])
# testing_and_comparing_numerical_random_values(A_Lnw_A_ELnw , A_Lnw_A_ELnw_M)
# # TODO: ATTENTION THIS IS WRONG. U_A**2 IS ALWAYS POSITIVE SO THERE ARE NO CASES WHERE THE FORCE F_A IS NEGATIVE, WHICH IS WRONG
#





