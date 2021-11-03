import numpy as np
import time

n_freq = 100
n_nodes = 200
n_dof = 6

# Potential sparse matrix:
M_1 = (np.random.rand(n_freq, n_nodes, n_nodes, n_dof, n_dof) + 0.05)**200  # bigger matrix
M_2 = (np.random.rand(n_freq, n_nodes, n_dof) + 0.05)**200  # smaller matrix

########################################################################################################################
# Vector @ Matrix @ Vector.T
########################################################################################################################
print('TESTING: V @ M @ V.T')

# Testing einsum
start_time = time.time()
R_ein = np.einsum('fmj, fmnjk, fnk -> f', M_2, M_1, M_2, optimize=True)
print('Pure einsum time: ' + str(np.round((time.time() - start_time))) )


# Testing 1 loop + einsum
start_time = time.time()
R_loop = np.zeros(n_freq)
for f in range(n_freq):
    R_loop[f] = np.einsum('mj, mnjk, nk -> ', M_2[f], M_1[f], M_2[f], optimize=True)
print('1 loop + einsum time: ' + str(np.round((time.time() - start_time))) )


# 2D dot operations with einsum
M_1_new = np.reshape(np.moveaxis(M_1, 3, 2), (n_freq, n_nodes*n_dof, n_nodes*n_dof))  # confirmed to be equivalent
M_2_new = np.reshape(M_2, (n_freq, n_nodes*n_dof))  # confirmed to be equivalent
start_time = time.time()
R_2D = np.zeros(n_freq)
for f in range(n_freq):
    R_2D[f] = np.einsum('m, mn, n -> ', M_2_new[f], M_1_new[f], M_2_new[f], optimize=True)
print('2D dot operations with einsum time: ' + str(np.round((time.time() - start_time))) )


# 2D dot operations with @
start_time = time.time()
R_2D_2 = np.zeros(n_freq)
for f in range(n_freq):
    R_2D_2[f] = M_2_new[f] @ M_1_new[f] @ np.transpose(M_2_new[f][np.newaxis])
print('2D dot operations with @ time: ' + str(np.round((time.time() - start_time))) )


# 2D dot operations with dot
start_time = time.time()
R_2D_dot = np.zeros(n_freq)
for f in range(n_freq):
    R_2D_dot[f] = np.dot(M_2_new[f], np.dot(M_1_new[f], np.transpose(M_2_new[f][np.newaxis])))
print('2D dot operations with @ time: ' + str(np.round((time.time() - start_time))) )


# Sparse operations. Results are wronger if np.round is done with fewer decimal places.
from scipy.sparse import csr_matrix
start_time = time.time()
M_1_round = np.round(M_1_new, 3)
M_2_round = np.round(M_2_new, 3)
M_1_sparse = np.zeros((n_freq, n_nodes*n_dof, n_nodes*n_dof)).tolist()
M_2_sparse = np.zeros((n_freq, n_nodes*n_dof)).tolist()
M_2_sparse_transp = np.zeros((n_freq, n_nodes*n_dof)).tolist()
R_sparse = np.zeros(n_freq)
for f in range(n_freq):
    M_1_sparse[f] = csr_matrix(M_1_round[f])
    M_2_sparse[f] = csr_matrix(M_2_round[f])
    M_2_sparse_transp[f] = M_2_sparse[f].transpose()
    R_sparse[f] = M_2_sparse[f].dot(M_1_sparse[f].dot(M_2_sparse_transp[f])).todense()

print('sparse matrix time: ' + str(np.round((time.time() - start_time))) )

print('V @ M @ V.T   -   Differences in values (should be 0):')
print(np.max(abs(R_ein - R_loop)))
print(np.max(abs(R_2D - R_ein)))
print(np.max(abs(R_2D_2 - R_ein)))
print(np.max(abs(R_2D_dot - R_ein)))
print(np.max(abs(R_sparse - R_2D)))



########################################################################################################################
# Matrix @ Matrix @ Matrix
########################################################################################################################
print('TESTING: M @ M @ M')

# Testing einsum
start_time = time.time()
R_ein = np.einsum('fmnij, fnojk, fopkl -> fmpil', M_1, M_1, M_1, optimize=True)
print('Pure einsum time: ' + str(np.round((time.time() - start_time))) )



# Testing 1 loop + einsum
start_time = time.time()
R_loop = np.zeros((n_freq, n_nodes, n_nodes, n_dof, n_dof))
for f in range(n_freq):
    R_loop[f] = np.einsum('mnij, nojk, opkl -> mpil', M_1[f], M_1[f], M_1[f], optimize=True)
print('1 loop + einsum time: ' + str(np.round((time.time() - start_time))) )



# 2D dot operations with einsum
M_1_new = np.reshape(np.moveaxis(M_1, 3, 2), (n_freq, n_nodes*n_dof, n_nodes*n_dof))  # confirmed to be equivalent
start_time = time.time()
R_2D = np.zeros((n_freq, n_nodes*n_dof, n_nodes*n_dof))
for f in range(n_freq):
    R_2D[f] = np.einsum('mn, no, op -> mp', M_1_new[f], M_1_new[f], M_1_new[f], optimize=True)
R_2D = np.moveaxis(np.reshape(R_2D, (n_freq, n_nodes, n_dof, n_nodes, n_dof)), 2, 3)
print('2D dot operations with einsum time: ' + str(np.round((time.time() - start_time))) )



# 2D dot operations with @
start_time = time.time()
R_2D_2 = np.zeros((n_freq, n_nodes*n_dof, n_nodes*n_dof))
for f in range(n_freq):
    R_2D_2[f] = M_1_new[f] @ M_1_new[f] @ M_1_new[f]
R_2D_2 = np.moveaxis(np.reshape(R_2D_2, (n_freq, n_nodes, n_dof, n_nodes, n_dof)), 2, 3)
print('2D dot operations with @ time: ' + str(np.round((time.time() - start_time))) )



# 2D dot operations with dot
start_time = time.time()
R_2D_dot = np.zeros((n_freq, n_nodes*n_dof, n_nodes*n_dof))
for f in range(n_freq):
    R_2D_dot[f] = np.dot(M_1_new[f], np.dot(M_1_new[f], M_1_new[f]))
R_2D_dot = np.moveaxis(np.reshape(R_2D_dot, (n_freq, n_nodes, n_dof, n_nodes, n_dof)), 2, 3)
print('2D dot operations with @ time: ' + str(np.round((time.time() - start_time))) )



# Sparse operations.
from scipy.sparse import csr_matrix
start_time = time.time()
M_1_round = np.round(M_1_new, 3)
M_1_sparse = np.zeros((n_freq, n_nodes*n_dof, n_nodes*n_dof)).tolist()
R_sparse = np.zeros((n_freq, n_nodes*n_dof, n_nodes*n_dof))
for f in range(n_freq):
    M_1_sparse[f] = csr_matrix(M_1_round[f])
    R_sparse[f] = M_1_sparse[f].dot(M_1_sparse[f].dot(M_1_sparse[f])).todense()
R_sparse = np.moveaxis(np.reshape(R_sparse, (n_freq, n_nodes, n_dof, n_nodes, n_dof)), 2, 3)
print('sparse matrix time: ' + str(np.round((time.time() - start_time))) )



# Confirmations
print('M @ M @ M   -   Differences in values (should be 0):')
print(np.max(abs(R_ein - R_loop)))
print(np.max(abs(R_2D - R_ein)))
print(np.max(abs(R_2D_2 - R_ein)))
print(np.max(abs(R_2D_dot - R_ein)))
print(np.max(abs(R_sparse - R_2D)))

















# Results for n_freq = 1024, g_node_num = 277, n_dof = 6 :
# Pure einsum time: 285.0
# 1 loop + einsum time: 271.0
# 2D dot operations with einsum time: 275.0
# 2D dot operations with @ time: 170.0