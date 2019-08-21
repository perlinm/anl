#!/usr/bin/env python3

import numpy as np
import scipy.special

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = FutureWarning)
    import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from linalg_methods import tf_outer_product, tensor_power
from network_methods import cubic_network, checkerboard_network

from itertools import product as set_product
from functools import reduce

tf_real_dtype = tf.float64

##########################################################################################
# methods for constructing a tensor network that represents the partition function
#   of a classical q-state clock model on a periodic primitive hypercubic lattice
# hamiltonian: H = -\sum_{<j,k>} cos(s_j -s_k) - h \sum_j cos(s_j),
# with s_j \in 2\pi/q \times \Z; note that the ising model is a 2-state potts model
##########################################################################################

def _integers(spokes, center_on_zero = False):
    if not center_on_zero:
        return range(spokes)
    else:
        return ( val-(spokes-1)/2 for val in range(spokes) )

def _angles(spokes, center_on_zero = False):
    return ( val * 2*np.pi/spokes for val in _integers(spokes, center_on_zero) )

# vertex tensor in the "bare" tensor network of the clock model
def bare_node_tensor(dimension, spokes, inv_temp, field):
    return sum( np.exp(inv_temp*field * np.cos(angle)) *
                tensor_power(tf.one_hot(idx, spokes, dtype = tf_real_dtype), 2*dimension)
                for idx, angle in zip(_integers(spokes), _angles(spokes)) )

# link tensor in the "bare" tensor network of the clock model
def bare_link_tensor(spokes, inv_temp):
    return tf.constant([ [ np.exp(inv_temp * np.cos(theta-phi))
                           for phi in _angles(spokes) ]
                         for theta in _angles(spokes) ])

# singular values of the link matrix
def _diag_val(spokes, idx, inv_temp):
    return sum( np.exp( inv_temp * np.cos(angle) ) * np.cos( idx * angle )
                for angle in _angles(spokes) ) / spokes

# thermal edge state vectors
def _therm_vec(spokes, angle, inv_temp):
    return tf.constant([ np.sqrt(abs(_diag_val(spokes, idx, inv_temp))) *
                         np.exp(1j*idx*angle)
                         for idx in _integers(spokes) ])
def _therm_mat(dimension, spokes, angle, inv_temp):
    vec = tensor_power(_therm_vec(spokes, angle, inv_temp), dimension)
    return tf_outer_product(vec, tf.math.conj(vec))

# fused vertex tensor in the cubic tensor network of the clock model
def vertex_tensor_fused(dimension, spokes, inv_temp, field):
    tensor = sum( np.exp(inv_temp*field * np.cos(angle)) *
                  _therm_mat(dimension, spokes, angle, inv_temp)
                  for angle in _angles(spokes) )
    return tf.math.real(tensor) / spokes

# vertex tensor in the cubic tensor network of the XY model
def vertex_tensor_XY(dimension, bond_dimension, inv_temp, field):
    assert(bond_dimension % 2 == 1) # only allow for odd bond dimensions
    def _prod_diag_val(indices, xx):
        return np.prod([ scipy.special.iv(idx, xx) for idx in indices ])
    def _mod_diag_val(indices, xx):
        idx_sum = sum(indices[:dimension]) - sum(indices[dimension:])
        return scipy.special.iv(idx_sum, xx)
    index_vals = set_product(_integers(bond_dimension,True), repeat = 2*dimension)
    vector = tf.constant([ np.sqrt(_prod_diag_val(indices, inv_temp)) *
                           _mod_diag_val(indices, inv_temp*field)
                           for indices in index_vals ])
    return tf.reshape(vector, [bond_dimension]*2*dimension)

# vertex tensor in either of the clock or XY models
def vertex_tensor(network_type, dimension, spokes, inv_temp, field):
    if network_type == "fused":
        return vertex_tensor_fused(dimension, spokes, inv_temp, field)
    if network_type == "XY":
        return vertex_tensor_XY(dimension, spokes, inv_temp, field)

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

# construct tensor network that evaluates the the partition function of the clock model
def clock_network(lattice_shape, spokes, inv_temp, field = 0, network_type = "fused"):
    dimension = len(lattice_shape)

    if network_type == "bare":

        node_tensor = bare_node_tensor(dimension, spokes, inv_temp, field)
        link_tensor = bare_link_tensor(spokes, inv_temp)

        node_tensor_num = np.prod(lattice_shape)
        link_tensor_num = 2 * node_tensor_num

        node_tensor_norm = tf.norm(node_tensor)
        link_tensor_norm = tf.norm(link_tensor)

        normed_node_tensor = node_tensor / node_tensor_norm
        normed_link_tensor = link_tensor / link_tensor_norm

        log_net_scale = ( node_tensor_num * np.log(node_tensor_norm) +
                          link_tensor_num * np.log(link_tensor_norm) )

        def tensor_bundle(loc):
            if all( idx % 1 == 0 for idx in loc ):
                return normed_node_tensor
            else:
                return normed_link_tensor

        link_tensors = True
        net, nodes, edges = cubic_network(lattice_shape, tensor_bundle, link_tensors)
        return net, nodes, edges, log_net_scale

    if network_type in [ "fused", "XY" ]:
        tensor = vertex_tensor(network_type, dimension, spokes, inv_temp, field)
        tensor_num = np.prod(lattice_shape)
        _network_generator = cubic_network

    elif network_type == "chkr":
        assert(all( num % 2 == 0 for num in lattice_shape ))
        tensor = checkerboard_tensor(dimension, spokes, inv_temp, field)
        tensor_num = np.prod(lattice_shape) / 2**(dimension-1)
        _network_generator = checkerboard_network

    tensor_norm = tf.norm(tensor)
    normed_tensor = tensor / tensor_norm
    log_net_scale = tensor_num * np.log(tensor_norm)
    def tensor_bundle(_): return normed_tensor
    net, nodes, edges = _network_generator(lattice_shape, tensor_bundle)

    return net, nodes, edges, log_net_scale
