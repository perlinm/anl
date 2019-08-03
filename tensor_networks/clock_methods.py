#!/usr/bin/env python3

import numpy as np
import scipy

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

def _integers(spokes):
    return ( val-(spokes-1)/2 for val in range(spokes) )

def _angles(spokes):
    return ( val * 2*np.pi/spokes for val in _integers(spokes) )

# singular values of the link matrix
def _diag_val(spokes, idx, inv_temp):
    return sum( np.exp( inv_temp * np.cos(angle) ) * np.cos( idx * angle )
                for angle in _angles(spokes) ) / spokes

# thermal edge state vectors
def _therm_vec(spokes, idx, inv_temp):
    return tf.constant([ np.sqrt(abs(_diag_val(spokes, ww, inv_temp))) *
                         np.exp(1j*ww*idx*2*np.pi/spokes)
                         for ww in _integers(spokes) ])
def _therm_mat(dimension, spokes, idx, inv_temp):
    vec = tensor_power(_therm_vec(spokes, idx, inv_temp), dimension)
    return tf_outer_product(vec, tf.math.conj(vec))

# vertex tensor in the cubic tensor network of the clock model
def vertex_tensor(dimension, spokes, inv_temp, field):
    tensor = sum( np.exp(inv_temp*field * np.cos(idx*2*np.pi/spokes)) *
                  _therm_mat(dimension, spokes, idx, inv_temp)
                  for idx in _integers(spokes) )
    return tf.math.real(tensor) / spokes

# vertex tensor in the cubic tensor network of the XY model
def vertex_tensor_XY(dimension, bond_dimension, inv_temp, field):
    assert(bond_dimension % 2 == 1) # only allow for odd bond dimensions
    def _prod_diag_val(indices, xx):
        return np.prod([ scipy.special.iv(idx, xx) for idx in indices ])
    def _mod_diag_val(indices, xx):
        idx_sum = sum(indices[:dimension]) - sum(indices[dimension:])
        return scipy.special.iv(idx_sum, xx)
    index_vals = set_product(_integers(bond_dimension), repeat = 2*dimension)
    vector = tf.constant([ np.sqrt(_prod_diag_val(indices, inv_temp)) *
                           _mod_diag_val(indices, inv_temp*field)
                           for indices in index_vals ])
    return tf.reshape(vector, [bond_dimension]*2*dimension)

# checkerboard tensor in the checkerboard tensor network of the clock model
def checkerboard_tensor(dimension, spokes, inv_temp, field):
    def _shift_idx(idx, direction):
        return ( idx + 2**direction ) % 2**dimension

    def _angle_diff(angles,idx,direction):
        return angles[idx] - angles[_shift_idx(idx,direction)]

    def _tensor_factor(idx, angle_vals):
        angles = np.array(angle_vals) * 2*np.pi/spokes
        site_term =  field / 2 * np.cos(angles[idx])
        edge_term = sum( np.cos(_angle_diff(angles,idx,direction))
                         for direction in range(dimension) ) / 2
        scalar = np.exp(inv_temp * ( site_term + edge_term ))
        return tf.one_hot(angle_vals[idx], spokes, on_value = scalar)

    def _tensor_term(angle_vals):
        tensor_factors = [ _tensor_factor(idx, angle_vals)
                           for idx in range(len(angle_vals)) ]
        return reduce(tf_outer_product, tensor_factors)

    tensor = sum( _tensor_term(angle_vals)
                  for angle_vals in set_product(_integers(spokes), repeat = 2**dimension) )
    return tensor / spokes**2

# construct tensor network on a periodic primitive hypercubic lattice
def clock_network(lattice_shape, spokes, inv_temp, field = 0,
                  use_vertex = True, use_XY = False):
    if use_vertex:
        if use_XY: vertex_tensor = vertex_tensor_XY
        tensor = vertex_tensor(len(lattice_shape), spokes, inv_temp, field)
        tensor_num = np.prod(lattice_shape)
    else:
        assert(all( num % 2 == 0 for num in lattice_shape ))
        tensor = checkerboard_tensor(len(lattice_shape), spokes, inv_temp, field)
        tensor_num = np.prod(lattice_shape) / 2**(len(lattice_shape)-1)

    tensor_norm = tf.norm(tensor)
    normed_tensor = tensor / tensor_norm
    log_net_scale = tensor_num * np.log(tensor_norm)
    def tensor_bundle(_): return normed_tensor

    if use_vertex:
        net, nodes, edges = cubic_network(lattice_shape, tensor_bundle)
    else:
        net, nodes, edges = checkerboard_network(lattice_shape, tensor_bundle)

    return net, nodes, edges, log_net_scale
