from sympy import symbols, Matrix, simplify, cos, sin, tan, acos, asin, atan, sqrt, eye, diag, expand, solve, linsolve, Eq, O, Poly, pi
import numpy as np

def matrix_3dof_to_6dof(M3):
    M6 = Matrix([[M3[0,0], M3[0,1], M3[0,2], 0, 0, 0],
                 [M3[1,0], M3[1,1], M3[1,2], 0, 0, 0],
                 [M3[2,0], M3[2,1], M3[2,2], 0, 0, 0],
                 [0, 0, 0, M3[0,0], M3[0,1], M3[0,2]],
                 [0, 0, 0, M3[1,0], M3[1,1], M3[1,2]],
                 [0, 0, 0, M3[2,0], M3[2,1], M3[2,2]]])
    return M6

# Variables
# beta_t = symbols(r'\tilde{\beta}')
# beta_b = symbols(r'\bar{\beta}')
# theta_t = symbols(r'\tilde{\theta}')
# theta_b = symbols(r'\bar{\theta}')

beta_t = symbols(r' bt')
beta_b = symbols(r' bb')
theta_t = symbols(r'tt')
theta_b = symbols(r'tb')

# Transformation matrices. b-"bar"; t-"tilde", to represent mean and instantaneous.
T_LsLwb = Matrix([[cos(beta_b), -cos(theta_b)*sin(beta_b),  sin(theta_b)*sin(beta_b)],
                  [sin(beta_b),  cos(theta_b)*cos(beta_b), -sin(theta_b)*cos(beta_b)],
                  [          0,              sin(theta_b),              cos(theta_b)]])

T_LsLwt = Matrix([[cos(beta_t), -cos(theta_t)*sin(beta_t),  sin(theta_t)*sin(beta_t)],
                  [sin(beta_t),  cos(theta_t)*cos(beta_t), -sin(theta_t)*cos(beta_t)],
                  [          0,              sin(theta_t),              cos(theta_t)]])

# T_LwbLwt = simplify(T_LsLwb.T * T_LsLwt)

T_LwbGw = Matrix([[0., -1., 0.],
                  [1.,  0., 0.],
                  [0.,  0., 1.]])
T_LwbGw_6 = matrix_3dof_to_6dof(T_LwbGw)

T_LsGw = T_LsLwb * T_LwbGw
T_LsGw_6 = matrix_3dof_to_6dof(T_LsGw)

t11 = T_LsGw[0,0]
t12 = T_LsGw[0,1]
t13 = T_LsGw[0,2]
t21 = T_LsGw[1,0]
t22 = T_LsGw[1,1]
t23 = T_LsGw[1,2]
t31 = T_LsGw[2,0]
t32 = T_LsGw[2,1]
t33 = T_LsGw[2,2]

s1 = (t11 * t22 - t21 * t12) / sqrt(t11 ** 2 + t21 ** 2)
s2 = (t11 * t22 - t21 * t12) / (t11 ** 2 + t21 ** 2)
s3 = t32 / sqrt(t11 ** 2 + t21 ** 2)
s4 = (t11 * t23 - t21 * t13) / sqrt(t11 ** 2 + t21 ** 2)
s5 = (t11 * t23 - t21 * t13) / (t11 ** 2 + t21 ** 2)
s6 = t33 / sqrt(t11 ** 2 + t21 ** 2)

Cq = symbols('Cq')
Cp = symbols('Cp')
Ch = symbols('Ch')
Cqq = symbols('Cqq')
Cpp = symbols('Cpp')
Chh = symbols('Chh')

Cq_db = symbols('Cq_db')
Cp_db = symbols('Cp_db')
Ch_db = symbols('Ch_db')
Cqq_db = symbols('Cqq_db')
Cpp_db = symbols('Cpp_db')
Chh_db = symbols('Chh_db')

Cq_dt = symbols('Cq_dt')
Cp_dt = symbols('Cp_dt')
Ch_dt = symbols('Ch_dt')
Cqq_dt = symbols('Cqq_dt')
Cpp_dt = symbols('Cpp_dt')
Chh_dt = symbols('Chh_dt')

Cu = symbols('Cu')
Cv = symbols('Cv')
Cw = symbols('Cw')
Cuu = symbols('Cuu')
Cvv = symbols('Cvv')
Cww = symbols('Cww')

Cu_db = symbols('Cu_db')
Cv_db = symbols('Cv_db')
Cw_db = symbols('Cw_db')
Cuu_db = symbols('Cuu_db')
Cvv_db = symbols('Cvv_db')
Cww_db = symbols('Cww_db')

Cu_dt = symbols('Cu_dt')
Cv_dt = symbols('Cv_dt')
Cw_dt = symbols('Cw_dt')
Cuu_dt = symbols('Cuu_dt')
Cvv_dt = symbols('Cvv_dt')
Cww_dt = symbols('Cww_dt')

Cx = symbols('Cx')
Cy = symbols('Cy')
Cz = symbols('Cz')
Cxx = symbols('Cxx')
Cyy = symbols('Cyy')
Czz = symbols('Czz')

Cx_db = symbols('Cx_db')
Cy_db = symbols('Cy_db')
Cz_db = symbols('Cz_db')
Cxx_db = symbols('Cxx_db')
Cyy_db = symbols('Cyy_db')
Czz_db = symbols('Czz_db')

Cx_dt = symbols('Cx_dt')
Cy_dt = symbols('Cy_dt')
Cz_dt = symbols('Cz_dt')
Cxx_dt = symbols('Cxx_dt')
Cyy_dt = symbols('Cyy_dt')
Czz_dt = symbols('Czz_dt')

Cqph = Matrix([Cq, Cp, Ch, Cqq, Cpp, Chh])
Cqph_db = Matrix([Cq_db, Cp_db, Ch_db, Cqq_db, Cpp_db, Chh_db])
Cqph_dt = Matrix([Cq_dt, Cp_dt, Ch_dt, Cqq_dt, Cpp_dt, Chh_dt])

Cuvw = Matrix([Cu, Cv, Cw, Cuu, Cvv, Cww])
Cuvw_db = Matrix([Cu_db, Cv_db, Cw_db, Cuu_db, Cvv_db, Cww_db])
Cuvw_dt = Matrix([Cu_dt, Cv_dt, Cw_dt, Cuu_dt, Cvv_dt, Cww_dt])

Cxyz = Matrix([Cx, Cy, Cz, Cxx, Cyy, Czz])
Cxyz_db = Matrix([Cx_db, Cy_db, Cz_db, Cxx_db, Cyy_db, Czz_db])
Cxyz_dt = Matrix([Cx_dt, Cy_dt, Cz_dt, Cxx_dt, Cyy_dt, Czz_dt])

# Cuvw = T_LwbGw_6.T * Cqph
# Cuvw_db = T_LwbGw_6.T * Cqph_db
# Cuvw_dt = T_LwbGw_6.T * Cqph_dt

# Cxyz = T_LsGw_6 * Cuvw
# Cxyz_db = T_LsGw_6 * Cuvw_db
# Cxyz_dt = T_LsGw_6 * Cuvw_dt

u = symbols('u')
v = symbols('v')
w = symbols('w')
U = symbols('U')

delta_t_dumb = simplify( t32 / (sqrt(t11**2 + t21**2)) * v/U  +  t33 / sqrt(t11**2 + t21**2) * w/U )
delta_t = w/U  # delta_t_dumb became simplified since cos(tb)/sqrt(cos(tb)**2) == 1 for all -pi/2 <= tb <= pi/2 !
delta_b = simplify( (t11*t22 - t12*t21) / (t11**2 + t21**2) * v/U  +  (t11*t23 - t13*t21)/(t11**2 + t21**2) * w/U )

Tv = Matrix([[0      , -s1, s2*t31],
             [s1     ,   0,    -s3],
             [-s2*t31,  s3,      0]])

Tv_6 = matrix_3dof_to_6dof(simplify(Tv))

Tw = Matrix([[0      , -s4, s5*t31],
             [s4     ,   0,    -s6],
             [-s5*t31,  s6,      0]])

Tw_6 = matrix_3dof_to_6dof(simplify(Tw))

T_LwbLwt_dumb = eye(6,6) + Tv_6 * v/U + Tw_6 * w/U
# The above matrix can be simplified since cos(tb)/sqrt(cos(tb)**2) == 1 for all -pi/2 <= tb <= pi/2 ! So:
T_LwbLwt = matrix_3dof_to_6dof( Matrix([[1                , -v/U, v*tan(theta_b)/U],
                                           [v/U              ,    1,             -w/U],
                                           [-v*tan(theta_b)/U,  w/U,                1]]))
# todo: WRONG. T_LwbLwt is NOT the same as T_GwbGwt, some axes are swaped

V = symbols('V')
V = sqrt((U+u)**2 + v**2 + w**2)
rho = symbols('rho')
B = symbols('B')

B_mat = diag([B, B, B, B**2, B**2, B**2], unpack=True)
fb_quad = simplify(0.5 * rho * V**2 * T_LwbLwt * B_mat)

# Removing quadratic terms
V_no_quad = sqrt(expand(V)**2 - u**2 - v**2 - w**2)
Cuvw_Taylor = Cuvw + Cuvw_db*delta_b + Cuvw_dt*delta_t

from transformations import T_LwGw_func

fb_Gw =  0.5 * rho * V_no_quad**2 * T_LwbGw_6.T * T_LwbLwt * B_mat * Cuvw_Taylor - 0.5*rho*U**2*B_mat*Cuvw

# Building the Matrix A = [Au, Av, Aw]. Beautiful!!! :) :) <3 <3
A_Gw = np.zeros((6, 3)).tolist()
for i in range(6):
    for j in range(3):
        A_Gw[i][j] = Poly(fb_Gw[i], u, v, w).coeff_monomial([u,v,w][j])
A_Gw = Matrix(A_Gw)
A0_Gw = A_Gw / (0.5*rho*U)
simplify(A0_Gw)


############################################################################################################################################################################
Poly(fb_Gw[i], u, v, w).degree_list()
Poly(fb_Gw[i], u, v, w).factor_list()
Poly(fb_Gw[i], u, v, w).coeffs()


############################################################################################################################################################################
# AND IN LOCAL STRUCTURAL COORDINATES AND COEFFICIENTS:
fb_Ls = T_LsGw_6 * fb_Gw

Cuvw_Ls = T_LsGw_6.T * Cxyz
Cuvw_db_Ls = T_LsGw_6.T * Cxyz_db
Cuvw_dt_Ls = T_LsGw_6.T * Cxyz_dt

fb_Ls_Ls = fb_Ls.subs({Cu:Cuvw_Ls[0],
                        Cv:Cuvw_Ls[1],
                        Cw:Cuvw_Ls[2],
                        Cuu:Cuvw_Ls[3],
                        Cvv:Cuvw_Ls[4],
                        Cww:Cuvw_Ls[5],
                        Cu_db: Cuvw_db_Ls[0],
                        Cv_db: Cuvw_db_Ls[1],
                        Cw_db: Cuvw_db_Ls[2],
                        Cuu_db: Cuvw_db_Ls[3],
                        Cvv_db: Cuvw_db_Ls[4],
                        Cww_db: Cuvw_db_Ls[5],
                        Cu_dt: Cuvw_dt_Ls[0],
                        Cv_dt: Cuvw_dt_Ls[1],
                        Cw_dt: Cuvw_dt_Ls[2],
                        Cuu_dt: Cuvw_dt_Ls[3],
                        Cvv_dt: Cuvw_dt_Ls[4],
                        Cww_dt: Cuvw_dt_Ls[5],
                        })

# Building the Matrix A = [Au, Av, Aw].
A_Ls = np.zeros((6, 3)).tolist()
for i in range(6):
    for j in range(3):
        A_Ls[i][j] = Poly(fb_Ls_Ls[i], u, v, w).coeff_monomial([u,v,w][j])
A_Ls = Matrix(A_Ls)
A0_Ls = A_Ls / (0.5*rho*U)
simplify(A0_Ls)



############################################################################################################################################################################
# Some confirmations with transformation matrices
alpha = symbols(r' a')
beta_g = symbols(r' bg')
theta_g = symbols(r'tg')

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

T_GwGs = (R_z(beta_g + pi/2) @ R_y(-theta_g)).T
T_GwGs @ Matrix([1,0,0])


T_LnwLs = (R_z(pi/2) @ R_y(-theta_b)).T
T_LnwGw =  simplify((R_y(theta_b) @ R_z(-beta_b) @ R_y(-theta_b)).T)
T_LnwGw_2 = simplify(T_LnwLs @ T_LsGw)

# Confirmation:
T_LnwGw - T_LnwGw_2

############################################################################################################################################################################
############################################################################################################################################################################
# Some confirmations with cosine rule
############################################################################################################################################################################
############################################################################################################################################################################
from sympy import symbols, Matrix, simplify, cos, sin, tan, acos, asin, atan, sqrt, eye, diag, expand, solve, linsolve, Eq, O, Poly, pi
import numpy as np

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

u = symbols('u')
v = symbols('v')
w = symbols('w')
U = symbols('U')
beta_t = symbols(r' bt')
beta_b = symbols(r' bb')
theta_t = symbols(r'tt')
theta_b = symbols(r'tb')


# Transformation matrices. b-"bar"; t-"tilde", to represent mean and instantaneous.
T_LsLwb = Matrix([[cos(beta_b), -cos(theta_b)*sin(beta_b),  sin(theta_b)*sin(beta_b)],
                  [sin(beta_b),  cos(theta_b)*cos(beta_b), -sin(theta_b)*cos(beta_b)],
                  [          0,              sin(theta_b),              cos(theta_b)]])

T_LsLwt = Matrix([[cos(beta_t), -cos(theta_t)*sin(beta_t),  sin(theta_t)*sin(beta_t)],
                  [sin(beta_t),  cos(theta_t)*cos(beta_t), -sin(theta_t)*cos(beta_t)],
                  [          0,              sin(theta_t),              cos(theta_t)]])

T_LwbGw = Matrix([[0., -1., 0.],
                  [1.,  0., 0.],
                  [0.,  0., 1.]])

T_LwtGwt = T_LwbGw

T_LsGw = T_LsLwb * T_LwbGw
T_LsGwt = T_LsLwt * T_LwtGwt

V = symbols('V')
V = sqrt((U+u)**2 + v**2 + w**2)
V_Gw = Matrix([U+u, v, w])
V_no_quad = sqrt(expand(V)**2 - u**2 - v**2 - w**2)

V_Ls = T_LsGw @ V_Gw
V_Ls_x = V_Ls[0]
V_Ls_y = V_Ls[1]
V_Ls_z = V_Ls[2]
V_Ls_xy = sqrt(V_Ls_x**2 + V_Ls_y**2)

U_Gw = Matrix([U, 0, 0])
U_Ls = T_LsGw @ U_Gw
U_Ls_x = U_Ls[0]
U_Ls_y = U_Ls[1]
U_Ls_z = U_Ls[2]
U_Ls_xy = sqrt(U_Ls_x**2 + U_Ls_y**2)

# beta_b = acos(U_Ls_y/U_Ls_xy)
beta_t = acos(V_Ls_y/V_Ls_xy)
# theta_b = atan(U_Ls_z/U_Ls_y)
theta_t = atan(V_Ls_z/V_Ls_y)

# V_normal_0 = (V_no_quad * cos(beta_t))
V_normal_1 = V * cos(beta_t)
V_normal_2 = sqrt(V_Ls_y**2+V_Ls_z**2)

# V_normal_0.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })
V_normal_1.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })
V_normal_2.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })


V_Ls.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })
V.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })
deg(beta_t.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 }))


V_Ls_xy.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })
beta_t.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })
V_normal_2.subs({theta_b:rad(30), beta_b:rad(80), U:30, u:3.1, v:4.4, w:1.5 })



simplify(cos(acos(V_Ls_y/V_Ls_xy))*V)


simplify(V_normal_1 - V_normal_2)

V_normal_1 = V_normal_1.expand() + O(u**2) + O(v**2) + O(w**2)
V_normal_1.removeO()


u + v + u*v**3 + O(v**2)

simplify(beta_t.subs({theta_b:0.1}))


