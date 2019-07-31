#!/usr/bin/env python3

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

from linalg_methods import tensor_power
from network_methods import cubic_network

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
def vertex_tensor(neighbors, spokes, inv_temp, field):
    return sum( np.exp(inv_temp*field * np.cos(idx*2*np.pi/spokes))
                * tensor_power(temp_vec(spokes, idx, inv_temp), neighbors)
                for idx in range(spokes) )

# construct tensor network on a periodic primitive hypercubic lattice
def clock_network(lattice_shape, spokes, inv_temp, field = 0):
    tensor = vertex_tensor(2*len(lattice_shape), spokes, inv_temp, field)
    tensor_norm = tf.norm(tensor)
    normed_tensor = tensor / tensor_norm
    log_net_scale = np.prod(lattice_shape) * np.log(tensor_norm)

    def tensor_bundle(_): return normed_tensor
    net, nodes, edges = cubic_network(lattice_shape, tensor_bundle)
    return net, nodes, edges, log_net_scale
