#!/usr/bin/env python3

import os
import numpy as np
np.set_printoptions(linewidth = 200)

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.enable_v2_behavior()

import tensornetwork as tn

temp = 1 # temperature
lattice_shape = (3,3) # lattice sites per axis
print_node = (0,1)

# value of tensor for particular indices
def tensor_val(idx):
    s_idx = 2 * np.array(idx) - 1
    return ( 1 + np.prod(s_idx) ) / 2 * np.exp(np.sum(s_idx)/2/temp)

# construct a single tensor in the (translationally-invariant) network
tensor_shape = (2,)*len(lattice_shape)*2 # dimension of space at each index of the tensor
tensor_vals = [ tensor_val(idx) for idx in np.ndindex(tensor_shape) ]
tensor = tf.reshape(tf.constant(tensor_vals), tensor_shape)

# construct tensor network in a hyperrectangular lattice
def make_net(shape = lattice_shape):
    net = tn.TensorNetwork() # initialize empty tensor network

    # make all nodes, indexed by lattice coorinates
    nodes = { idx : net.add_node(tensor, name = str(idx))
              for idx in np.ndindex(shape) }

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for axis in range(len(shape)): # for each axis of the lattice

        # choose the axes of tensors that we will contract
        dir_fst, dir_snd = axis, axis + len(shape)

        for idx_fst in nodes: # loop over all nodes

            # identify the "next" neighboring node along this lattice axis
            idx_snd = list(idx_fst)
            idx_snd[dir_fst] = ( idx_snd[dir_fst] + 1 ) % shape[dir_fst]
            idx_snd = tuple(idx_snd)

            # connect up the nodes
            edge = (idx_fst,idx_snd)
            edges[edge] = net.connect(nodes[idx_fst][dir_fst],
                                      nodes[idx_snd][dir_snd],
                                      name = str(edge))

    return net, nodes, edges

net, nodes, edges = make_net()
tn.contractors.naive(net)
print(net.get_final_node().tensor.numpy())

##########################################################################################
# evaluation of tensor network via bubbling
##########################################################################################

def zero_state(length):
    length = max(0,length)
    return tf.reshape(tf.one_hot(0, 2**length, dtype = tf.float64), (2,)*length)

net, nodes, edges = make_net()
# bubbling_order = nodes.values()

eaten_nodes = set()
dangling_edges = []

state = zero_state(0)
for node_key, node in nodes.items():
    # identify input/output edges to this node, and the corresponding axes
    inp_edges, inp_axes = [], []
    out_edges, out_axes = [], []
    for edge in node.get_all_edges():
        if node == edge.node1:
            axis_to_other = edge.axis1
            other_node = edge.node2
        else:
            axis_to_other = edge.axis2
            other_node = edge.node1
        if other_node in eaten_nodes:
            inp_edges.append(edge)
            inp_axes.append(axis_to_other)
        else:
            out_edges.append(edge)
            out_axes.append(axis_to_other)

    # get the tensor associated with this node, reordering axes appropriately
    tensor = tf.transpose(node.get_tensor(), inp_axes + out_axes)

    # act the swallowing operator on the state
    state_axes = [ dangling_edges.index(edge) for edge in inp_edges ]
    tensor_axes = list(range(len(inp_edges)))
    state = tf.tensordot(state, tensor, axes = [ state_axes, tensor_axes ])

    # add to our list of "eaten" nodes, update the list of dangling edges
    eaten_nodes.add(node)
    for edge in inp_edges:
        dangling_edges.remove(edge)
    dangling_edges += out_edges

print(state.numpy())
