#!/usr/bin/env python3

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from linalg_methods import tf_outer_product, tensor_power
from network_methods import cubic_network, checkerboard_network

from itertools import product as set_product
from functools import reduce

##########################################################################################
# methods for constructing a tensor network that represents the partition function
#   of a classical q-state clock model on a periodic primitive hypercubic lattice
# hamiltonian: H = -\sum_{<j,k>} cos(s_j -s_k) - h \sum_j cos(s_j),
# with s_j \in 2\pi/q \times \Z; note that the ising model is a 2-state potts model
##########################################################################################

# singular values of the link matrix
def diag_val(spokes, idx, inv_temp):
    return sum( np.exp(inv_temp * np.cos(angle) + 1j * idx * angle )
                for angle in np.array(range(spokes))*2*np.pi/spokes )

# thermal edge state vectors
def temp_vec(spokes, idx, inv_temp):
    vec = np.array([ np.real( np.sqrt(diag_val(spokes, ww, inv_temp)) *
                              np.exp(1j*ww*idx*2*np.pi/spokes) )
                     for ww in range(spokes) ])
    return tf.constant(vec) / np.sqrt(spokes)

# vertex tensor in the cubic tensor network of the clock model
def vertex_tensor(dimension, spokes, inv_temp, field):
    return sum( np.exp(inv_temp*field * np.cos(idx*2*np.pi/spokes))
                * tensor_power(temp_vec(spokes, idx, inv_temp), 2*dimension)
                for idx in range(spokes) )

# checkerboard tensor in the checkerboard tensor network of the clock model
def checkerboard_tensor(dimension, spokes, inv_temp, field):
    assert(dimension == 2) # only allow 2-D for now

    def _tensor_factor(idx, angle_vals):
        angles = np.array(angle_vals) * 2*np.pi/spokes
        scalar = np.exp( inv_temp * field/2 * np.cos(angles[idx]) +
                         inv_temp * np.cos(angles[idx]-angles[(idx+1)%4]) )
        return tf.one_hot(angle_vals[idx], spokes, on_value = scalar)

    def _tensor_term(angle_vals):
        tensor_factors = [ _tensor_factor(idx, angle_vals)
                           for idx in range(len(angle_vals)) ]
        return reduce(tf_outer_product, tensor_factors)

    return sum( _tensor_term(angle_vals)
                for angle_vals in set_product(range(spokes), repeat = 4) )

# construct tensor network on a periodic primitive hypercubic lattice
def clock_network(lattice_shape, spokes, inv_temp, field = 0, use_vertex = True):
    if use_vertex:
        tensor = vertex_tensor(len(lattice_shape), spokes, inv_temp, field)
    else:
        tensor = checkerboard_tensor(len(lattice_shape), spokes, inv_temp, field)

    tensor_norm = tf.norm(tensor)
    normed_tensor = tensor / tensor_norm
    def tensor_bundle(_): return normed_tensor

    if use_vertex:
        net, nodes, edges = cubic_network(lattice_shape, tensor_bundle)
        log_net_scale = np.prod(lattice_shape) * np.log(tensor_norm)
    else:
        net, nodes, edges = checkerboard_network(lattice_shape, tensor_bundle)
        log_net_scale = 1/2 * np.prod(lattice_shape) * np.log(tensor_norm)

    return net, nodes, edges, log_net_scale
