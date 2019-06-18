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

import qutip as qt

net, nodes, edges = make_net()
bubbling_order = nodes.values()

eaten_nodes = set()
dangling_edges = []

state = qt.basis(1,0)
for node in bubbling_order:
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

    # identify auxiliary dangling edges that do not participate in swallowing this node
    aux_edges = [ edge for edge in dangling_edges if edge not in inp_edges ]

    # identify number of input, output, and auxiliary edges
    inp_num = len(inp_edges)
    out_num = len(out_edges)
    aux_num = len(aux_edges)

    # total number of qubits necessary to swallow this node
    total_num = max(inp_num,out_num) + aux_num

    # identify the index (location) of input, auxiliary, and ancilla qubits (<--> edges)
    node_inp_idx = [ dangling_edges.index(edge) for edge in inp_edges ]
    node_aux_idx = [ dangling_edges.index(edge) for edge in aux_edges ]
    node_anc_idx = list(range(len(dangling_edges), total_num))

    # attach ancillas to the current state, if necessary
    if out_num > inp_num:
        ancillas = [ qt.basis(2,0) ] * ( out_num - inp_num )
        if state.dims == [[1]]*2:
            state = state[0,0] * qt.tensor(*ancillas)
        else:
            state = qt.tensor(state, *ancillas)

    # get the tensor associated with this node, reordering axes appropriately
    tensor = tf.transpose(node.get_tensor(), out_axes + inp_axes)

    # convert the tensor into a 2-D matrix addressing input/output degrees of freedom
    matrix_shape = (2**out_num, 2**inp_num)
    matrix = tf.reshape(tensor, matrix_shape)

    # make the matrix square by padding it in with zeros as necessary
    np_matrix = matrix.numpy()
    matrix_padding = [ (0, 2**max(inp_num,out_num) - matrix_shape[jj]) for jj in range(2) ]
    np_matrix = np.pad(np_matrix, matrix_padding, mode = "constant", constant_values = 0)
    np_matrix = np.kron(np_matrix, np.eye(2**aux_num))
    qt_matrix = qt.Qobj(np_matrix, dims = [ [2]*total_num ]*2) # convert to qutip object

    # rearrange qubit order to match that of the swallowing operator
    state = state.permute(node_anc_idx + node_inp_idx + node_aux_idx)

    # act on the current state by the swallowing operator
    state = qt_matrix * state

    # remove extra qubits if necessary
    if out_num < inp_num:
        kept_num = total_num - inp_num + out_num
        state = qt.Qobj(state[:2**kept_num], dims = [ [2]*kept_num, [1]*kept_num ])

    # add to our list of "eaten" nodes, update the list of dangling edges
    eaten_nodes.add(node)
    for edge in inp_edges:
        dangling_edges.remove(edge)
    dangling_edges = out_edges + dangling_edges

print(state[:].real[0,0])
