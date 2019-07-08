#!/usr/bin/env python3

##########################################################################################
# methods for constructing a tensor network that represents the partition function
#   of a classical Ising model on a periodic primitive hypercubic lattice
# hamiltonian: H = -\sum_{<j,k>} s_j s_k - h \sum_j s_j, with each s_j \in { +1, -1 }
##########################################################################################

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

from network_assemblers import cubic_network

# compute a "bare" vertex tensor element
def bare_vertex_tensor_val(idx, field_over_temp, spin_factor = False):
    if len(set(idx)) != 1: return 0
    else:
        spin = 2*idx[0] - 1
        return np.exp(field_over_temp * spin) * ( spin if spin_factor else 1 )

# construct the entire "bare" vertex tensor
def bare_vertex_tensor(lattice_dim, field_over_temp, spin_factor = False):
    vertex_shape = (2,2) * lattice_dim
    tensor_vals = [ bare_vertex_tensor_val(idx, field_over_temp, spin_factor)
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
def fused_vertex_tensor(lattice_dim, inv_temp, field, spin_factor = False):
    sqrt_T_L = tf.linalg.sqrtm(link_tensor(inv_temp))
    T_V = bare_vertex_tensor(lattice_dim, field * inv_temp, spin_factor)
    for axis in range(len(T_V.shape)):
        T_V = tf.tensordot(sqrt_T_L, T_V, axes = [ [1], [axis] ])
    return T_V

# construct tensor network on a periodic primitive hypercubic lattice
def ising_network(lattice_shape, inv_temp, field, spin_factor_nodes = []):
    assert( node_idx in np.ndindex(lattice_shape) for node_idx in spin_factor_nodes )
    lattice_dim = len(lattice_shape)

    # build normalized "regular" tensors to put on ordinary vertices of the lattice
    tensor_reg = fused_vertex_tensor(lattice_dim, inv_temp, field)
    tensor_norm_reg = tf.norm(tensor_reg)
    normed_tensor_reg = tensor_reg / tensor_norm_reg

    # build normalized "impurity" tensors to put on the vertices
    #   specified by spin_factor_nodes
    tensor_imp = fused_vertex_tensor(lattice_dim, inv_temp, field, spin_factor = True)
    tensor_norm_imp = tf.norm(tensor_imp)
    if tensor_norm_imp.numpy() == 0: tensor_norm_imp = 1
    normed_tensor_imp = tensor_imp / tensor_norm_imp

    # compute the (logarithm of the) factor taken out by normalizing tensors
    tensors_imp = len(spin_factor_nodes)
    tensors_reg = np.prod(lattice_shape) - tensors_imp
    log_net_scale = tensors_reg * np.log(tensor_norm_reg) \
                  + tensors_imp * np.log(tensor_norm_imp)

    # define the tensor at each node, indexed by lattice coordinates
    def tensor_bundle(node_idx):
        if node_idx not in spin_factor_nodes:
            return normed_tensor_reg
        else:
            return normed_tensor_imp

    net, nodes, edges = cubic_network(lattice_shape, tensor_bundle)
    return net, nodes, edges, log_net_scale
