# -*- coding: utf-8 -*-
"""
created: 2019
author: Bernardo Costa
email: bernamdc@gmail.com
"""

import numpy as np

# Including off-diagonals in K_tilde and M_tilde for modal coupling
def modal_analysis_func(M, K):
    """Get: \n
        Generalized mass matrix for each mode; \n
        Generalized stiffness matrix for each mode; \n
        Eigen angular frequencies (rad/s); \n
        Eigen mode shapes. \n
    M -- Mass matrix (array of arrays) \n
    K -- Stiffness matrix (array of arrays) \n
    Off-diagonals are included!
    """

    # Eigen values and vectors (unsorted):
    var1 = np.dot(np.linalg.inv(M), K)  # K/M. Matrix for which the eigenvalues and right eigenvectors will be computed
    values, vectors = np.linalg.eig(var1)  # obtaining eigen values and eigen vectors
    omega_0 = np.sqrt(values)  # eigen frequencies in rad/s
    shape_0 = np.array([i/np.max(np.abs(i)) for i in np.transpose(vectors)])  # mode shapes. Normalizing the vectors.

    # Sorting the omegas and shapes
    shapes = np.array([s for w, s in sorted(zip(omega_0, shape_0))])  # shape: (n_modes, n_nodes)
    omegas = np.array([w for w, s in sorted(zip(omega_0, shape_0))])

    # Generalized Mass and Stiffness for each mode.
    M_tilde = shapes @ M @ np.transpose(shapes)
    K_tilde = shapes @ K @ np.transpose(shapes)

    return M_tilde, K_tilde, omegas, shapes   # todo: ATTENTION. This produces complex modes when Kse is included in the Modal (include_SE_in_modal = True)
    # return np.real(M_tilde), np.real(K_tilde), np.real(omegas), np.real(shapes)
    # return np.sign(np.real(M_tilde))*np.abs(M_tilde), np.sign(np.real(K_tilde))*np.abs(K_tilde), np.sign(np.real(omegas))*np.abs(omegas), np.sign(np.real(shapes))*np.abs(shapes)


# Discarding off-diagonals for modal coupling
def simplified_modal_analysis_func(M, K):
    """Get: \n
        Diagonals of:
        Generalized mass matrix for each mode; \n
        Generalized stiffness matrix for each mode; \n
        Eigen angular frequencies (rad/s); \n
        Eigen mode shapes. \n
    M -- Mass matrix (array of arrays) \n
    K -- Stiffness matrix (array of arrays) \n
    Off-diagonals are discarded!
    """

    ndof = len(M)  # number of degrees of freedom == number of modes
    
    # Eigen values and vectors (unsorted):
    var1 = np.dot(np.linalg.inv(M), K)  # K/M. Matrix for which the eigenvalues and right eigenvectors will be computed
    values, vectors = np.linalg.eig(var1)  # obtaining eigen values and eigen vectors
    omega_0 = np.sqrt(values)  # eigen frequencies in rad/s
    shape_0 = np.array([i/np.max(np.abs(i)) for i in np.transpose(vectors)])  # mode shapes. Normalizing the vectors.
    
    # Sorting the omegas and shapes
    shapes = np.array([s for w, s in sorted(zip(omega_0,shape_0))])
    omegas = np.array([w for w, s in sorted(zip(omega_0,shape_0))])
    
    # Generalized Mass and Stiffness for each mode.
    M_tilde = np.array([np.matmul(np.matmul(np.transpose(shapes[i]), M), shapes[i]) for i in range(ndof)])
    K_tilde = np.array([np.matmul(np.matmul(np.transpose(shapes[i]), K), shapes[i]) for i in range(ndof)])

    return M_tilde, K_tilde, omegas, shapes


















