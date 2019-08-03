#!/usr/bin/env python3

import os
import numpy as np

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn
from tensornetwork.contractors import greedy_contractor

from linalg_methods import tf_outer_product, to_unitary

# return an indexed pure state in a space with a given tensor product structure (shape)
def idx_state(index, shape, dtype = tf.float64):
    dim = max(np.prod(shape, dtype = int), 1)
    return tf.reshape(tf.one_hot(index % dim, dim, dtype = dtype), shape)

# return the trivial (zero) state in a space with a given shape
def zero_state(shape = (), dtype = tf.float64):
    return idx_state(0, shape, dtype)

# attach (remove) subsystems to (from) a state
def attach_subsystems(state, shape):
    if len(shape) == 0: return state
    return tf_outer_product(zero_state(shape, dtype = state.dtype), state)
def remove_subsystems(state, del_num):
    if del_num <= 0: return state
    shape = state.get_shape()[:del_num]
    return tf.tensordot(zero_state(shape, dtype = state.dtype), state,
                        axes = [ list(range(del_num)) ]*2)

# identify input/output edges to this node, as well as the corresponding axes
def get_edge_info(node, eaten_nodes):
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
    return inp_edges, inp_op_idx, out_edges, out_op_idx

# simulate the contracting of a tensor network using a computer
# uses the method described in arxiv.org/abs/0805.0040
# accepts: a list of TensorNetwork nodes in bubbling order
# returns:
# (i) the logarithm of the probability of "success", i.e. finding all ancillas in |0>, and
# (ii) the logarithm of the product of the operator norms of the swallowing operators
def quantum_contraction(nodes, bubbler = None, print_status = False, dtype = tf.float64):
    if bubbler is None: bubbler = nodes.keys()
    tf_Z = tf.constant([[1,0],[0,-1]], dtype = dtype) # Pauli-Z
    tf_X = tf.constant([[0,1],[1,0]], dtype = dtype) # Pauli-X

    log_net_norm = 0
    state = zero_state(dtype = dtype)

    eaten_nodes = set()
    dangling_edges = []
    for node_idx in bubbler:
        node = nodes[node_idx]
        inp_edges, inp_op_idx, out_edges, out_op_idx = get_edge_info(node, eaten_nodes)

        # identify auxiliary dangling edges that do not participate in swallowing this node
        aux_edges = [ edge for edge in dangling_edges if edge not in inp_edges ]

        # identify the numbers of different kinds of subsystems
        inp_num = len(inp_edges) # number of input subsystems to the swallowing operator
        out_num = len(out_edges) # number of output subsystems to the swallowing operator
        aux_num = len(aux_edges) # number of auxiliary (bystander) subsystems
        assert( aux_num == len(state.shape) - inp_num ) # sanity check

        # rearrange the order of subsystems in the state
        inp_state_idx = [ dangling_edges.index(edge) for edge in inp_edges ]
        aux_state_idx = [ dangling_edges.index(edge) for edge in aux_edges ]
        state = tf.transpose(state, inp_state_idx + aux_state_idx)

        # get the dimensions of the input, output, and auxiliary spaces
        inp_dims = tuple( edge.dimension for edge in inp_edges )
        out_dims = tuple( edge.dimension for edge in out_edges )
        aux_dims = tuple( edge.dimension for edge in aux_edges )
        inp_dim = np.prod(inp_dims, dtype = int)
        out_dim = np.prod(out_dims, dtype = int)
        aux_dim = np.prod(aux_dims, dtype = int)

        # get the tensor/matrix associated with this node, reordering axes as necessary
        swallow_tensor = tf.transpose(tf.cast(node.get_tensor(), dtype),
                                      out_op_idx + inp_op_idx)
        swallow_matrix = tf.reshape(swallow_tensor, (out_dim, inp_dim))

        # perform singular-value decomposition of swallowing_matrix:
        #   S = V_L @ diag(D) @ V_R^\dag,
        # where V_L, V_R are isometric; and D is positive semi-definite
        vals_D, mat_V_L, mat_V_R = tf.linalg.svd(swallow_matrix)

        # convert isometries into unitaries on an extension of the domain
        #   achieved by attaching trivial ancillary subsystems (i.e. in |0>)
        mat_V_L = to_unitary(mat_V_L)
        mat_V_R = to_unitary(mat_V_R)

        # normalize D by its operator norm, keeping track of the norm independently
        norm_D = max(vals_D.numpy())
        normed_vals_D = vals_D/norm_D
        log_net_norm += np.log(norm_D)

        # construct the unitary action of D
        mat_U_D = tf_outer_product(tf_Z, tf.linalg.diag(normed_vals_D)) \
                + tf_outer_product(tf_X, tf.linalg.diag(tf.sqrt(1-normed_vals_D**2)))

        # rotate into the right-diagonal basis of the swallowing operator
        state = tf.reshape(state, (inp_dim,) + aux_dims)
        state = tf.tensordot(tf.transpose(mat_V_R, conjugate = True), state, axes = [[1],[0]])
        state = tf.reshape(state, inp_dims + aux_dims)

        # if we lose subsystems upon swallowing this operator, then project them out now
        state = remove_subsystems(state, max(0, inp_num - out_num))

        # attach ancilla, act on the state by the unitary U_D, and project out the ancilla
        state = tf.reshape(state, (min(inp_dim,out_dim),) + aux_dims)
        state = attach_subsystems(state, (2,))
        state = tf.tensordot(mat_U_D, state, axes = [ [1,3], [0,1] ])
        state = remove_subsystems(state, 1)

        # if we gain subsystems upon swallowing this tensor, then attach them now
        if out_dim > inp_dim:
            new_dims = list(out_dims)
            for dim in inp_dims:
                new_dims.remove(dim)
            state = attach_subsystems(state, new_dims)

        # rotate back from the left-diagonal basis of the swallowing operator
        state = tf.reshape(state, (out_dim,) + aux_dims)
        state = tf.tensordot(mat_V_L, state, axes = [ [1], [0] ])
        state = tf.reshape(state, out_dims + aux_dims)

        # add to our list of "eaten" nodes, update the list of dangling edges
        eaten_nodes.add(node)
        for edge in inp_edges:
            dangling_edges.remove(edge)
        dangling_edges = out_edges + dangling_edges

        # print status info
        if print_status:
            # the node we just swallowed
            print("node:", node_idx, node)

            # the norm of the state after swallowing this node and projecting out ancillas
            state_norm = tf.tensordot(tf.transpose(state, conjugate = True), state,
                                      axes = [ list(range(len(state.shape))) ]*2).numpy()
            print("norm:", state_norm)
            print("-"*10)

    assert(state.shape == ()) # we should have contracted the entire state to a single number
    log_net_prob = 2 * np.log(abs(state.numpy())) # probability of finding zero state
    return log_net_prob, log_net_norm

# classical backend to quantum_contraction
# accepts both a TensorNetwork object and a bubbler as input
# same outputs as quantum_contraction
def classical_contraction(net, nodes, bubbler = None):
    if bubbler is None: bubbler = nodes.keys()
    log_net_norm = 0
    eaten_nodes = set()
    dangling_edges = []

    for node_idx in bubbler:
        node = nodes[node_idx]
        inp_edges, inp_op_idx, out_edges, out_op_idx = get_edge_info(node, eaten_nodes)

        # get the dimensions of the input/output spaces
        inp_dim = np.prod([ edge.dimension for edge in inp_edges ], dtype = int)
        out_dim = np.prod([ edge.dimension for edge in out_edges ], dtype = int)

        # get the tensor associated with this node, reordering axes as necessary
        swallow_tensor = tf.transpose(node.get_tensor(), out_op_idx + inp_op_idx)
        swallow_matrix = tf.reshape(swallow_tensor, (out_dim, inp_dim))

        ### the tensorflow's svd algorithm is much faster than numpy's, but tensorlow
        ###   appears to have numerical issues and sometimes gives a segfault... :(
        # singular_vals = tf.linalg.svd(swallow_matrix)[0].numpy()
        singular_vals = np.linalg.svd(swallow_matrix.numpy())[1]
        matrix_norm = singular_vals.max()
        log_net_norm += np.log(matrix_norm)

        # add to our list of "eaten" nodes, update the list of dangling edges
        eaten_nodes.add(node)
        for edge in inp_edges:
            dangling_edges.remove(edge)
        dangling_edges = out_edges + dangling_edges

    # compute value of network
    tn.contractors.greedy_contractor.greedy(net)
    ### although the greedy contractor is generally faster, it sometimes
    ### runs out of memory in cases when the naive contractor does not...
    # tn.contractors.naive(net)
    log_net_val = np.log(net.get_final_node().tensor.numpy())

    log_net_prob = 2 * (log_net_val - log_net_norm)
    return log_net_prob, log_net_norm
