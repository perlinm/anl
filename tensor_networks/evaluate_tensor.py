#!/usr/bin/env python3

import os
import numpy as np

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.enable_v2_behavior()

import tensornetwork as tn

size = 3 # linear size of lattice
dim = 2 # dimension of lattice
temp = 1 # temperature

tensor_shape = (2,)*4 # dimension of space at each index of the tensor

# define order of indices
# note that this order is important for how we organize edges later
rt, dn, lt, up = 0, 1, 2, 3

# map between multi-index value of node and single-integer value of node
def idx_to_val(idx, dim = dim):
    if len(idx) == 1: return idx[0]
    return size * idx_val(idx[:-1], dim-1) + idx[-1]
def val_to_idx(val, dim = dim):
    if dim == 1: return (val,)
    return val_idx(val // size, dim-1) + (val % size,)

# value of tensor for particular indices
def tensor_val(idx):
    s_idx = 2 * np.array(idx) - 1
    return ( 1 + np.prod(s_idx) ) / 2 * np.exp(np.sum(s_idx)/2/temp)

# construct a single tensor in the (translationally-invariant) network
tensor_vals = [ tensor_val(idx) for idx in np.ndindex(tensor_shape) ]
tensor = tf.reshape(tf.constant(tensor_vals), tensor_shape)

def make_net():
    # initialize empty tensor network
    net = tn.TensorNetwork()

    # make all nodes and organize them according to lattice structure
    # nodes indexed by (pos_x,pos_y) from top left to bottom right
    nodes = { idx : net.add_node(tensor, name = str(idx))
              for idx in np.ndindex((size,)*dim) }

    # connect all edges and organize them according to lattice structure
    # edges indexed by (pos_x,pos_y,direction)
    # where pos_x, and pos_y index the "first" node, and direction is right or down (rt or dn)
    def idx_next(idx,direction):
        if direction == rt: return ( idx[0], (idx[1]+1)%size )
        if direction == dn: return ( (idx[0]+1)%size, idx[1] )
    edges = { idx + (dir_out,) :
              net.connect(nodes[idx][dir_out], nodes[idx_next(idx,dir_out)][dir_in],
                          name = str(idx + (dir_out,)))
              for idx in nodes.keys() for dir_out, dir_in in [ (rt,lt), (dn,up) ] }

    return net, nodes, edges

net, nodes, edges = make_net()

tn.contractors.naive(net)
print(net.get_final_node().tensor.numpy())
