#!/usr/bin/env python3

import os
import numpy as np
np.set_printoptions(linewidth = 200)

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

##########################################################################################
# evaluation of tensor network via bubbling
##########################################################################################

import qutip as qt

net, nodes, edges = make_net()

edge_idx = { edge : idx for idx, edge in enumerate(list(edges.values())) }

eaten_nodes = set()
dangling_edges = []

state = qt.basis(1,0)
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
    matrix_padding = [ (0, 2**max(inp_num,out_num) - matrix_shape[jj])
                       for jj in range(2) ]
    np_matrix = np.pad(np_matrix, matrix_padding, mode = "constant", constant_values = 0)
    np_matrix = np.kron(np_matrix, np.eye(2**aux_num))
    qt_matrix = qt.Qobj(np_matrix, dims = [ [2]*total_num ]*2) # convert to qutip object

    # rearrange qubit order to match that of the swallowing operator
    op_inp_bit_idx = node_anc_idx + node_inp_idx + node_aux_idx
    state = state.permute(op_inp_bit_idx)

    # act on the current state by the swallowing operator
    state = qt_matrix * state

    # remove extra qubits if necessary
    trim_num = max(0, inp_num - out_num)
    if trim_num > 0:
        total_num -= trim_num
        state = qt.Qobj(state[:2**total_num], dims = [ [2]*total_num, [1]*total_num ])

    # add to our list of "eaten" nodes, update the list of dangling edges
    eaten_nodes.add(node)
    for edge in inp_edges:
        dangling_edges.remove(edge)
    dangling_edges = out_edges + dangling_edges

print(state[:].real[0,0])
