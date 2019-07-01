#!/usr/bin/env python3

##########################################################################################
# methods for constructing a tensor network that represents the partition function
#   of a classical Ising model on a periodic primitive hypercubic lattice
# hamiltonian: H = -\sum_{<j,k>} s_j s_k - h \sum_j s_j, with each s_j \in { +1, -1 }
##########################################################################################

import os
import numpy as np

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

# compute a "bare" vertex tensor element
def bare_vertex_tensor_val(idx, field_over_temp):
    if len(set(idx)) != 1: return 0
    else: return np.exp(field_over_temp * (2*idx[0]-1))

# construct the entire "bare" vertex tensor
def bare_vertex_tensor(field_over_temp, lattice_dim):
    vertex_shape = (2,2) * lattice_dim
    tensor_vals = [ bare_vertex_tensor_val(idx, field_over_temp)
                    for idx in np.ndindex(vertex_shape) ]
    return tf.reshape(tf.constant(tensor_vals), vertex_shape)

# compute a link tensor element
def link_tensor_val(idx, inv_temp):
    s_idx = 2 * np.array(idx) - 1
    return np.exp(inv_temp * np.prod(s_idx))

# construct the entire link tensor
def link_tensor(inv_temp):
    link_shape = (2,2)
    tensor_vals = [ link_tensor_val(idx, inv_temp)
                    for idx in np.ndindex(link_shape) ]
    return tf.reshape(tf.constant(tensor_vals), link_shape)

# construct the "fused" vertex tensor by contracting the square root of the edge tensor
#   at each leg of the "bare" vertex tensor
def fused_vertex_tensor(inv_temp, field, lattice_dim):
    sqrt_T_L = tf.linalg.sqrtm(link_tensor(inv_temp))
    T_V = bare_vertex_tensor(field * inv_temp, lattice_dim)
    for axis in range(len(T_V.shape)):
        T_V = tf.tensordot(sqrt_T_L, T_V, axes = [ [1], [axis] ])
    return T_V

# construct tensor network on a periodic primitive hypercubic lattice
def ising_network(inv_temp, field, lattice_shape):
    lattice_dim = len(lattice_shape)
    net = tn.TensorNetwork() # initialize empty tensor network

    # build the tensor at each vertex, and normalize it to its maximum value
    tensor = fused_vertex_tensor(inv_temp, field, lattice_dim)
    tensor_norm = tf.norm(tensor)
    normed_tensor = tensor / tensor_norm

    # make all nodes, indexed by lattice coorinates
    nodes = { idx : net.add_node(normed_tensor, name = str(idx))
              for idx in np.ndindex(lattice_shape) }
    log_val_estimate = len(nodes) * np.log(tensor_norm)

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for base_idx in np.ndindex(lattice_shape): # for vertex index
        # for each physical axis
        for axis in range(lattice_dim):
            # identify the tensor axes corresponding each direction along this physical axis
            base_axis, trgt_axis = axis, axis + lattice_dim

            # identify the neighboring vertex in the direction of base_axis
            trgt_idx = list(base_idx)
            trgt_idx[base_axis] = ( trgt_idx[base_axis] + 1 ) % lattice_shape[base_axis]
            trgt_idx = tuple(trgt_idx)

            # connect the neighboring nodes (tensors)
            edge = (base_idx, trgt_idx)
            edges[edge] = net.connect(nodes[base_idx][base_axis],
                                      nodes[trgt_idx][trgt_axis],
                                      name = str(edge))

    return net, nodes, edges, log_val_estimate
