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

dtype = tf.float64

def zero_state(length):
    length = max(0,length)
    return tf.reshape(tf.one_hot(0, 2**length, dtype = dtype), (2,)*length)

net, nodes, edges = make_net()
bubbling_order = nodes.values()

eaten_nodes = set()
dangling_edges = []

norm_product = 1
state = zero_state(0)
for node in bubbling_order:

    # identify input/output edges to this node, and the corresponding axes
    inp_edges, inp_op_idx = [], []
    out_edges, out_op_idx = [], []
    for edge in node.get_all_edges():
        if node == edge.node1:
            axis_to_other = edge.axis1
            other_node = edge.node2
        else:
            axis_to_other = edge.axis2
            other_node = edge.node1
        if other_node in eaten_nodes:
            inp_edges.append(edge)
            inp_op_idx.append(axis_to_other)
        else:
            out_edges.append(edge)
            out_op_idx.append(axis_to_other)

    # identify auxiliary dangling edges that do not participate in swallowing this node
    aux_edges = [ edge for edge in dangling_edges if edge not in inp_edges ]

    # identify the numbers of different kinds of qubits
    inp_num = len(inp_edges) # number of input qubits to the "bare" swallowing operator
    out_num = len(out_edges) # number of output qubits to the "bare" swallowing operator
    act_num = max(inp_num,out_num) # number of qubits the swallowing operator acts on
    aux_num = len(aux_edges) # number of auxiliary qubits
    anc_num = len(state.shape) - ( inp_num + aux_num ) # number of ancilla qubits

    # if we gain qubits upon swallowing this tensor,
    # then we need to attach "extra" (new) qubits to our state
    state = tf.tensordot(state, zero_state(out_num-inp_num), axes = 0)

    # rearrange the order of qubits in the state
    inp_state_idx = [ dangling_edges.index(edge) for edge in inp_edges ]
    aux_state_idx = [ dangling_edges.index(edge) for edge in aux_edges ]
    anc_state_idx = list(range(inp_num+aux_num, inp_num+aux_num+anc_num))
    ext_state_idx = list(range(inp_num+aux_num+anc_num, act_num+aux_num+anc_num))
    qubit_order = inp_state_idx + ext_state_idx + aux_state_idx + anc_state_idx
    state = tf.transpose(state, qubit_order)

    # get the tensor associated with this node, reordering axes as necessary
    op_tensor = tf.transpose(node.get_tensor(), out_op_idx + inp_op_idx)

    # attach extra input/output legs to make the swallowing operator "square"
    op_tensor = tf.tensordot(op_tensor, zero_state(out_num-inp_num), axes = 0) # extra inputs
    op_tensor = tf.tensordot(zero_state(inp_num-out_num), op_tensor, axes = 0) # extra outputs

    # perform singular-value decomposition (SVD) of op_tensor as T = V_L @ D @ V_R^\dag,
    # where V_L, V_R are unitary; and D is diagonal and positive semi-definite
    op_matrix = tf.reshape(op_tensor, (2**act_num,)*2)
    vals_D, op_V_L, op_V_R = tf.svd(op_matrix)

    # rotate into the diagonal basis of D
    state = tf.reshape(state, (2**act_num,) + (2,)*(aux_num+anc_num))
    state = tf.tensordot(tf.transpose(op_V_R, conjugate = True), state, axes = [[1],[0]])

    # normalize D by its operator norm, keeping track of the norm independently
    norm_D = max(vals_D.numpy())
    normed_vals_D = vals_D/norm_D
    norm_product *= norm_D

    # construct the unitary action of D
    op_U_D = tf.tensordot(tf.eye(2, dtype = dtype),
                          tf.diag(normed_vals_D), axes = 0) \
           + tf.tensordot(tf.constant([[0,-1],[1,0]], dtype = dtype),
                          tf.diag(tf.sqrt(1-normed_vals_D**2)), axes = 0)

    # attach ancilla, act on the state by the unitary U_D, and throw ancilla in the back
    state = tf.tensordot(zero_state(1), state, axes = 0)
    state = tf.tensordot(op_U_D, state, axes = [ [1,3], [0,1] ])
    state = tf.transpose(state, list(np.roll(range(len(state.shape)),-1)))

    # rotate back to the standard qubit basis
    state = tf.tensordot(op_V_L, state, axes = [ [1], [0] ])
    state = tf.reshape(state, (2,)*(act_num+aux_num+anc_num+1))

    # remove (project out) unused qubits from the state
    state = tf.tensordot(zero_state(inp_num-out_num), state,
                         axes = [ list(range(inp_num-out_num)),
                                  list(range(inp_num-out_num)) ] )

    # add to our list of "eaten" nodes, update the list of dangling edges
    eaten_nodes.add(node)
    for edge in inp_edges:
        dangling_edges.remove(edge)
    dangling_edges = out_edges + dangling_edges

print("norm:", norm_product)
print("probability:", state.numpy().flatten()[0]**2)
print("value:", norm_product * state.numpy().flatten()[0])
