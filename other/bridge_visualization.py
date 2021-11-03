import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary even though it doesn't seem so.
from mass_and_stiffness_matrix import g_elem_nodes, g_elem_num, g_elem, L_2D
from mass_and_stiffness_matrix import g_node_coor, g_node_num, g_nodes

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

x = g_node_coor[:, 0]
y = g_node_coor[:, 1]
z = g_node_coor[:, 2]

ax.scatter(x, y, z, label='bridge girder')
ax.legend()
# ax.set_xlim3d(0, 6000)
# ax.set_ylim3d(-6000, 0)
# ax.set_zlim3d(0, 6000)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

plt.show()



