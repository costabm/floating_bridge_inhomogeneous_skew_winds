import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary even though it doesn't seem so.
# from mass_and_stiffness_matrix import g_elem_nodes, g_elem_num, g_elem, g_L_2D
# from mass_and_stiffness_matrix import g_node_coor, g_node_num, g_nodes

def normalize(v):
    return v/np.linalg.norm(v)

# Global structural axes "XYZ" and local structural axes "xyz"... Global:
vector_X = np.array([1, 0, 0])
vector_Y = np.array([0, 1, 0])
vector_Z = np.array([0, 0, 1])
# vector_x = np.einsum('ij,i->ij', g_node_coor[g_elem_nodes[:, 1]] - g_node_coor[g_elem_nodes[:, 0]], 1 / L_3D)
# vector_y = normalize(-np.cross(vector_x, vector_Z))  # perpendicular to plane containing loc_x & glob_Z
# vector_z = normalize(np.cross(vector_x, vector_y))

def angle_2_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))  # in radians.

# Plotting Axes
def plotting_3D_axes(m, v):
    """
    :param m: is the [3x3] matrix containing the 3 column-vectors of the axes. Vectors as columns!!
    :param v: is the (3,) array containing the wind direction
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # vector x
    xs = [0, m[0,0]]
    ys = [0, m[1,0]]
    zs = [0, m[2,0]]
    ax.plot(xs, ys, zs, color='red', label='x')
    # vector y
    xs = [0, m[0,1]]
    ys = [0, m[1,1]]
    zs = [0, m[2,1]]
    ax.plot(xs, ys, zs, color='green', label='y')
    # vector z
    xs = [0, m[0,2]]
    ys = [0, m[1,2]]
    zs = [0, m[2,2]]
    ax.plot(xs, ys, zs, color='blue', label='z')
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color='black', label='wind vector')  # wind vector
    vector_x = m[:,0]
    vector_y = m[:,1]
    vector_z = m[:,2]

    # Local axes: Beta and Theta
    v_proj_z = np.linalg.norm(v)*np.cos(angle_2_vectors(v, vector_z))*normalize(vector_z)
    v_proj_xy = v - v_proj_z  # the projection of v into local plane xy is the same as v minus its z-component (the projection of v to the plane's normal vec z).
    theta_loc = angle_2_vectors(v, v_proj_xy) *180/np.pi  # angle of attack.
    if np.dot(vector_x, v) >= 0:  # if v and X_axis point in same dir. necessary because the range of the arccos function is only 0 to 180, and we need -180 to 180.
        beta_loc = -angle_2_vectors(v_proj_xy, vector_y) *180/np.pi  # skew angle.
    elif np.dot(vector_x, v) < 0:  # if opposite directions.
        beta_loc =  angle_2_vectors(v_proj_xy, vector_y) *180/np.pi  # skew angle.

    # Global axes: Beta and Theta
    v_proj_XY = normalize(np.array([v[0], v[1], 0]))
    theta_glob = np.arccos(np.dot(v, v_proj_XY)) *180/np.pi  # angle of attack.
    if np.dot(vector_X, v) >= 0:  # if v and X_axis point in same dir. necessary because the range of the arccos function is only 0 to 180, and we need -180 to 180.
        beta_glob = -angle_2_vectors(v_proj_XY, vector_Y) *180/np.pi  # skew angle.
    elif np.dot(vector_X, v) < 0:  # if opposite directions.
        beta_glob =  angle_2_vectors(v_proj_XY, vector_Y) *180/np.pi  # skew angle.

    plt.title(r'Local: $\beta$ = ' + str(round(beta_loc,1)) + r' deg. $\theta$ = ' + str(round(theta_loc,1)) + ' deg.\n'
             +r'Global: $\beta$ = ' + str(round(beta_glob,1)) + r' deg. $\theta$ = ' + str(round(theta_glob,1)) + ' deg.')
    plt.legend()
    plt.show()
    return None

# Real bridge coordinates
# wind_vector = np.array([np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)])
# M = np.column_stack((vector_x[0], vector_y[0], vector_z[0]))  # 1 vector per column! (not per row)
# T = np.linalg.inv(M)  # Rotation Matrix. It's the inverse of M (if the rotation is to bring M to unity)
# plotting_3D_axes(M, wind_vector)
# plotting_3D_axes(T @ M, T @ wind_vector)  # the pre-multiplication only works if M is a column oriented matrix

# Testing other bridge axes
vector_x = normalize(np.array([1,0,0]))
vector_y = normalize(-np.cross(vector_x, vector_Z))  # perpendicular to plane containing loc_x & glob_Z
vector_z = normalize(np.cross(vector_x, vector_y))

#wind_vector = normalize(np.array([1, 1, np.sqrt(2)]))  # an array([-0.5, 0.5, np.sqrt(2)/2]) should have beta=theta=45deg
wind_vector = normalize(np.array([-1, 0.1, 0.1]))  # normalize(np.array([0.4698, 0.171, 0.866]))

M = np.column_stack((vector_x, vector_y, vector_z))  # 1 vector per column! (not per row)
T = np.linalg.inv(M)  # Rotation Matrix. It's the inverse of M (if the rotation is to bring M to unity)

plotting_3D_axes(M, wind_vector)
plotting_3D_axes(T @ M, T @ wind_vector)  # the pre-multiplication only works if M is a column oriented matrix

# PLOT THE EQUIVALENT BETA AND THETA FROM SOH'S BETA AND ALPHA.
# How to find the red vector, having only the green vector??? Intersection of a cone an the bridge XY plane.
from rotate_vectors import xParV, rotateAbout
wind_parallel = xParV(-wind_vector, vector_x)  # parallel to the bridge x axis
wind_perpendicular = -wind_vector - wind_parallel  # PERPENDICULAR SHOULD NOT HAVE AN X-COMPONENT
alpha = angle_2_vectors(wind_perpendicular, -vector_y) *180/np.pi
beta_SOH = angle_2_vectors(np.array([wind_vector[0], wind_vector[1], 0]), vector_y) *180/np.pi

v = wind_vector
v_proj_z = np.linalg.norm(v) * np.cos(angle_2_vectors(v, vector_z)) * normalize(vector_z)
v_proj_xy = v - v_proj_z  # the projection of v into local plane xy is the same as v minus its z-component (the projection of v to the plane's normal vec z).
theta_loc = angle_2_vectors(v, v_proj_xy) * 180 / np.pi  # angle of attack.
if np.dot(vector_x,
          v) >= 0:  # if v and X_axis point in same dir. necessary because the range of the arccos function is only 0 to 180, and we need -180 to 180.
    beta_loc = -angle_2_vectors(v_proj_xy, vector_y) * 180 / np.pi  # skew angle.
elif np.dot(vector_x, v) < 0:  # if opposite directions.
    beta_loc = angle_2_vectors(v_proj_xy, vector_y) * 180 / np.pi  # skew angle.

beta = beta_loc * np.pi/180
theta = theta_loc * np.pi/180

def wind_vector_from_beta_theta(beta, theta):
    """
    :param beta: as in You-Lin Xu's book. Page 388.
    :param theta: as in You-Lin Xu's book. Page 388.
    :return: vector with wind direction
    """
    vec_x = -np.sin(beta)
    vec_y = np.cos(beta)
    vec_z = np.sin(theta)
    return normalize(np.array([vec_x ,vec_y, vec_z]))

alpha = np.arctan(np.tan(theta)/np.cos(beta)) * 180/np.pi
theta = np.arctan(np.tan(alpha)*np.cos(beta)) * 180/np.pi


# SOH case and correction of the Betas
import pandas as pd

betas_SOH = np.array([0., 10., 20., 30., 40., 50.]) * np.pi/180
alphas_SOH = np.array([-3., -1.5, 0, 1.5, 3.]) * np.pi/180

betas_corrected_SOH = np.arctan(np.outer(np.tan(betas_SOH) , 1/np.cos(alphas_SOH))) * 180/np.pi
thetas_SOH = np.arcsin(np.outer(np.cos(betas_SOH), np.sin(alphas_SOH))) * 180/np.pi


betas_corrected_SOH = pd.DataFrame(betas_corrected_SOH)
thetas_SOH = pd.DataFrame(thetas_SOH)

betas_corrected_SOH.to_excel('betas_corrected_SOH.xls')
thetas_SOH.to_excel('thetas_SOH.xls')