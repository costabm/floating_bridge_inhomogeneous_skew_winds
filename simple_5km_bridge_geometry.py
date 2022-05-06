"""
Generates the node coordinates of a Bjornafjorden-like floating bridge.

created: 2019
author: Bernardo Costa
email: bernamdc@gmail.com

g - girder
p - pontoons
c - columns
"""

import numpy as np

########################################################################################################################
# Floating bridge Properties:
########################################################################################################################
R = 5000  # m. Horizontal radius of the bridge. (R = 5000 for Bjornafjorden. R = 500000 for almost straight)
arc_length = 5000  # m. Arc length of the bridge. (arc_length = 5000 for Bjornafjorden)
pontoons_s = np.cumsum([100]*49)  # Total of 49 pontoons: np.cumsum([100]*49)
zbridge = 14.5  # m. (deck height above water, measured at the Shear Centre!)
p_freeboard = 4  # m. (height of pontoon above water level)
CS_height = 4  # m. CS height
FEM_max_length = 25  # Paper 2 & 4: 25. Choose: 10,12.5,14.29,16.67,20,25,33.34,50,100  # Horizontal response->100 (accurate to 0.3%). Vertical->50(6.2%) 33.34(2.6%) 25(1.4%). Torsional->50(8.1%) 33.34(3.6%) 25(1.9%)
vertical_curvature = False  # False: horizontal bridge girder.
bridge_shape = 'C'
# Dependent variables
n_pontoons = len(pontoons_s)
########################################################################################################################


# # TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
# ########################################################################################################################
# # Floating bridge Properties:
# ########################################################################################################################
# R = 500000  # m. Horizontal radius of the bridge. (R = 5000 for Bjornafjorden)
# arc_length = 1000  # m. Arc length of the bridge.
# pontoons_s = np.cumsum([100]*1)  # Total of 49 pontoons
# zbridge = 14.5  # m. (deck height above water, measured at the Shear Centre!)
# p_freeboard = 4  # m. (height of pontoon above water level)
# CS_height = 4  # m. CS height
# FEM_max_length = 100
# vertical_curvature = False  # False: horizontal bridge girder.
# bridge_shape='C'
# # Dependent variables
# n_pontoons = len(pontoons_s)
# ########################################################################################################################
# # TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
#

def g_L_2D_without_g_node_coor_func(arc_length, pontoons_s, FEM_max_length):
    spans = np.diff([0] + list(pontoons_s) + [arc_length])
    # Creating array of element lengths that respect FEM_max_length
    g_L_2D = list(spans)  # begin w/ elements = spans, and then increase num elements
    for i, el in enumerate(g_L_2D):
        if el > FEM_max_length:
            local_elem_num = int(g_L_2D[i] / FEM_max_length)  # how many times does that g_elem need to be divided
            if g_L_2D[i] % FEM_max_length > 0:
                local_elem_num += 1  # One more element if division is not integer
            g_L_2D[i] = g_L_2D[i] / local_elem_num  # to divide that element
            for _ in range(local_elem_num - 1):
                g_L_2D.insert(i, g_L_2D[i])  # to insert in the list the remaining equal-sized elements
    if not all(el <= FEM_max_length for el in g_L_2D):
        print('Error. max_length not fulfilled')
    g_L_2D = np.array(g_L_2D)
    return g_L_2D

def g_node_coor_func(R, arc_length, pontoons_s, zbridge, FEM_max_length, bridge_shape):
    arc_angle = arc_length / R  # rad. "Aperture" angle of the whole bridge arc.
    chord = np.sin(arc_angle / 2) * R * 2
    sagitta = R - np.sqrt(R ** 2 - (chord / 2) ** 2)
    angle = 2 * np.arcsin(chord / 2 / R)  # (rad) (angle from shore to shore)
    start_angle = -angle / 2
    end_angle = angle / 2
    n_pontoons = len(pontoons_s)

    g_L_2D = g_L_2D_without_g_node_coor_func(arc_length, pontoons_s, FEM_max_length)

    g_s_2D = np.array([0] + list(np.cumsum(g_L_2D)))  # arc length along arch at each node, projected on a horizontal plan (2D).
    g_elem_num = len(g_L_2D)  # (num. of FEM elements along the arc)
    g_node_num = g_elem_num + 1  # (num. of FEM g_nodes along the arc, including boundaries)
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0

    delta_angle = g_L_2D / R  # (rad) each element's delta angle
    node_angle = np.array([start_angle] + list(start_angle + np.cumsum(delta_angle)))  # angle of each node

    if bridge_shape == 'C':
        nodesxcoor0 = R * np.sin(node_angle)
        nodesycoor0 = -R * np.cos(node_angle)
        nodeszcoor0 = zbridge + g_s_2D**0  # np.array([zbridge] * g_node_num)  # CHANGE HERE as desired.
        if vertical_curvature:
            nodeszcoor0 = -nodesycoor0 / 5  # CHANGE HERE as desired. -nodesycoor0 / 5 creates an average slope of 5% in each half of bridge (from 10% to - 10%)
        nodescoor0 = np.column_stack((nodesxcoor0, nodesycoor0, nodeszcoor0))  # First with origin at circle center
        g_node_coor = nodescoor0 - nodescoor0[0] + [0, 0, zbridge]  # Then changing Origin to first node
    elif bridge_shape == 'S':
        pass
    elif bridge_shape == 'I':
        pass
    return g_node_coor

def g_elem_nodes_func(g_node_coor):
    g_node_num = len(g_node_coor)
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0
    elemnode1 = g_nodes[:-1]
    elemnode2 = g_nodes[:-1] + 1
    return np.column_stack((elemnode1, elemnode2))

def g_elem_L_3D_func(g_node_coor):
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem_nodes = g_elem_nodes_func(g_node_coor)
    # 3D lengths (real, not projected)
    g_elem_L_3D = np.array([np.linalg.norm(g_node_coor[g_elem_nodes[i, 1]] - g_node_coor[g_elem_nodes[i, 0]]) for i in range(g_elem_num)])
    return g_elem_L_3D

def g_s_3D_func(g_node_coor):
    g_elem_L_3D = g_elem_L_3D_func(g_node_coor)
    return np.array([0] + list(np.cumsum(g_elem_L_3D)))

def p_node_idx_func(arc_length, pontoons_s, FEM_max_length):
    g_L_2D = g_L_2D_without_g_node_coor_func(arc_length, pontoons_s, FEM_max_length)
    n_pontoons = len(pontoons_s)
    g_s_2D = np.array([0] + list(np.cumsum(g_L_2D)))  # arc length along arch at each node, projected on a horizontal plan (2D).
    # p_node_idx = np.array([np.where(g_s_2D == pontoons_s[i])[0][0] for i in range(n_pontoons)])
    p_node_idx = np.array([np.where(np.isclose(g_s_2D, pontoons_s[i]))[0][0] for i in range(n_pontoons)])
    return p_node_idx

def p_node_coor_func(g_node_coor, arc_length, pontoons_s, FEM_max_length):
    # Pontoon indexes (which girder g_nodes have columns):
    p_node_idx = p_node_idx_func(arc_length, pontoons_s, FEM_max_length)
    p_node_coor = np.array([[g_node_coor[i, 0], g_node_coor[i, 1], 0] for i in p_node_idx])
    return p_node_coor

def c_height_func(g_node_coor, arc_length, pontoons_s, FEM_max_length):
    g_node_num = len(g_node_coor)
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0
    p_node_idx = p_node_idx_func(arc_length, pontoons_s, FEM_max_length)
    # Column heights, g_nodes & Pontoon g_nodes.
    c_height = np.array([(g_node_coor[i, 2] - p_freeboard - CS_height/2) for i in p_node_idx])  # todo: do it as function of coordinates of girder and pontoon.
    return c_height



g_node_coor = g_node_coor_func(R=R, arc_length=arc_length, pontoons_s=pontoons_s, zbridge=zbridge, FEM_max_length=FEM_max_length, bridge_shape=bridge_shape)
p_node_coor = p_node_coor_func(g_node_coor, arc_length=arc_length, pontoons_s=pontoons_s, FEM_max_length=FEM_max_length)
p_node_idx = p_node_idx_func(arc_length=arc_length, pontoons_s=pontoons_s, FEM_max_length=FEM_max_length)
c_height = c_height_func(g_node_coor, arc_length=arc_length, pontoons_s=pontoons_s, FEM_max_length=FEM_max_length)

