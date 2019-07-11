#!/usr/bin/env python3

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

from network_methods import cubic_network

##########################################################################################
# methods for constructing a tensor network that represents the partition function
#   of a classical q-state Potts model on a periodic primitive hypercubic lattice
# hamiltonian: H = -\sum_{<j,k>} cos(s_j -s_k) - h \sum_j cos(s_j),
# with s_j = 2\pi/q \times j; note that the ising model is a 2-state potts model
##########################################################################################

# construct a "bare" copy tensor to place at each vertex
def bare_vertex_tensor(spokes, neighbors, field_over_temp):
    tensor_shape = (spokes,)*neighbors
    tensor_values = [ np.exp(field_over_temp * np.cos(2*np.pi*idx[0]/spokes))
                      if len(set(idx)) == 1 else 0
                      for idx in np.ndindex(tensor_shape) ]
    return tf.reshape(tensor_values, tensor_shape)

# compute a link tensor element
def link_tensor_val(spokes, inv_temp, idx):
    angles = 2*np.pi * np.array(idx) / spokes
    return np.exp(inv_temp * np.cos(angles[0] - angles[1]))

# construct the entire link tensor
def link_tensor(spokes, inv_temp):
    link_shape = (spokes,)*2
    tensor_vals = [ link_tensor_val(spokes, inv_temp, idx)
                    for idx in np.ndindex(link_shape) ]
    return tf.reshape(tf.constant(tensor_vals), link_shape)

# construct the "fused" vertex tensor by contracting the square root of the edge tensor
#   at each leg of the "bare" vertex tensor
def fused_vertex_tensor(neighbors, spokes, inv_temp, field):
    vals_D, mat_V_L, mat_V_R = tf.linalg.svd(link_tensor(spokes, inv_temp))
    sqrt_T_L = tf.matmul(mat_V_L * tf.sqrt(vals_D), mat_V_R, adjoint_b = True)
    T_V = bare_vertex_tensor(spokes, neighbors, field * inv_temp)
    for axis in range(len(T_V.shape)):
        T_V = tf.tensordot(sqrt_T_L, T_V, axes = [ [1], [axis] ])
    return T_V

# construct tensor network on a periodic primitive hypercubic lattice
def potts_network(lattice_shape, spokes, inv_temp, field = 0):
    tensor = fused_vertex_tensor(2*len(lattice_shape), spokes, inv_temp, field)
    tensor_norm = tf.norm(tensor)
    normed_tensor = tensor / tensor_norm
    log_net_scale = np.prod(lattice_shape) * np.log(tensor_norm)

    def tensor_bundle(idx): return normed_tensor
    net, nodes, edges = cubic_network(lattice_shape, tensor_bundle)
    return net, nodes, edges, log_net_scale
