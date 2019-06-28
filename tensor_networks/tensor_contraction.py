#!/usr/bin/env python3

import os
import numpy as np

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

# return a pure state of given number of qubits
def idx_state(index, qubits, tf_dtype):
    qubits = max(0,qubits)
    dim = 2**qubits
    return tf.reshape(tf.one_hot(index % dim, dim, dtype = tf_dtype), (2,)*qubits)

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
# (i) the probability of "success", i.e. finding all ancillas in |0>,
# (i) the logarithm of the network "norm"
# (iii) the number of qubits necessary to run the computation, and
# (iv) the maximum number of qubits addressed by a single unitary
def quantum_contraction(bubbler, print_status = False, tf_dtype = tf.float64):
    zero_state = lambda qubits : idx_state(0, qubits, tf_dtype)
    tf_eye = tf.eye(2, dtype = tf_dtype)
    tf_iY = tf.constant([[0,1],[-1,0]], dtype = tf_dtype)
    net_size = len(bubbler)

    max_qubits = 0
    log_net_norm = 0
    state = zero_state(0)

    eaten_nodes = set()
    dangling_edges = []
    for node_idx, node in enumerate(bubbler):
        inp_edges, inp_op_idx, out_edges, out_op_idx = get_edge_info(node, eaten_nodes)

        # identify auxiliary dangling edges that do not participate in swallowing this node
        aux_edges = [ edge for edge in dangling_edges if edge not in inp_edges ]

        # identify the numbers of different kinds of qubits
        inp_num = len(inp_edges) # number of input qubits to the "bare" swallowing operator
        out_num = len(out_edges) # number of output qubits to the "bare" swallowing operator
        act_num = max(inp_num, out_num) # number of qubits the swallowing operator acts on
        aux_num = len(aux_edges) # number of auxiliary (bystander) qubits
        assert(aux_num == len(state.shape) - inp_num) # sanity check

        # update max_qubits, given the number of qubits required for this step
        if node_idx != 0 and node_idx != net_size-1:
            max_qubits = max(max_qubits, aux_num + act_num + 1)

        # if we gain qubits upon swallowing this tensor,
        # then we need "extra" qubits for the unitarized tensor to act on
        ext_num = max(0, out_num - inp_num) # number of extra qubits we need
        state = tf.tensordot(state, zero_state(ext_num), axes = 0)

        # rearrange the order of qubits in the state
        inp_state_idx = [ dangling_edges.index(edge) for edge in inp_edges ]
        aux_state_idx = [ dangling_edges.index(edge) for edge in aux_edges ]
        ext_state_idx = list(range(len(state.shape)-ext_num, len(state.shape)))
        qubit_order = inp_state_idx + ext_state_idx + aux_state_idx
        state = tf.transpose(state, qubit_order)

        # get the tensor associated with this node, reordering axes as necessary
        op_tensor = tf.transpose(node.get_tensor(), out_op_idx + inp_op_idx)

        # attach extra input/output legs to make the swallowing operator "square"
        op_tensor = tf.tensordot(op_tensor, zero_state(out_num-inp_num), axes = 0) # extra inputs
        op_tensor = tf.tensordot(zero_state(inp_num-out_num), op_tensor, axes = 0) # extra outputs

        # perform singular-value decomposition (SVD) of op_tensor as T = V_L @ D @ V_R^\dag,
        # where V_L, V_R are unitary; and D is diagonal and positive semi-definite
        op_matrix = tf.reshape(op_tensor, (2**act_num,)*2)
        vals_D, op_V_L, op_V_R = tf.linalg.svd(op_matrix)

        # normalize D by its operator norm, keeping track of the norm independently
        norm_D = max(vals_D.numpy())
        normed_vals_D = vals_D/norm_D
        log_net_norm += np.log(norm_D)

        # rotate into the diagonal basis of D
        state = tf.reshape(state, (2**act_num,) + (2,)*(len(state.shape)-act_num))
        state = tf.tensordot(tf.transpose(op_V_R, conjugate = True), state, axes = [[1],[0]])

        # construct the unitary action of D
        op_U_D = tf.tensordot(tf_eye, tf.linalg.diag(normed_vals_D), axes = 0) \
               + tf.tensordot(tf_iY, tf.linalg.diag(tf.sqrt(1-normed_vals_D**2)), axes = 0)

        # attach ancilla, act on the state by the unitary U_D, and remove (project out) the ancilla
        state = tf.tensordot(zero_state(1), state, axes = 0)
        state = tf.tensordot(op_U_D, state, axes = [ [1,3], [0,1] ])
        state = tf.tensordot(zero_state(1), state, axes = [ [0], [0] ] )

        # rotate back to the standard qubit basis
        state = tf.tensordot(op_V_L, state, axes = [ [1], [0] ])
        state = tf.reshape(state, (2,)*(act_num+aux_num))

        # remove (project out) unused qubits from the state
        state = tf.tensordot(zero_state(inp_num-out_num), state,
                             axes = [ list(range(inp_num-out_num)) ]*2)

        # add to our list of "eaten" nodes, update the list of dangling edges
        eaten_nodes.add(node)
        for edge in inp_edges:
            dangling_edges.remove(edge)
        dangling_edges = out_edges + dangling_edges

        # print status info
        if print_status:
            # the node we just swallowed
            print("node:", node_idx, node)

            # number of qubits required for this step
            print("qubits:", aux_num + act_num + 1)

            # the norm of the state after swallowing this node and projecting out ancillas
            state_norm = tf.tensordot(tf.transpose(state, conjugate = True), state,
                                      axes = [ list(range(len(state.shape))) ]*2).numpy()
            print("norm:", state_norm)
            print("-"*10)

    assert(state.shape == ()) # we should have contracted the entire state to a single number
    net_prob = state.numpy()**2 # probability of finding zero state
    return net_prob, log_net_norm, max_qubits

# classical backend to quantum_contraction
# accepts both a TensorNetwork object and a bubbler as input
# same outputs as quantum_contraction
def classical_contraction(net, bubbler, tf_dtype = tf.float64):
    net_size = len(bubbler)

    max_qubits = 0
    log_net_norm = 0

    eaten_nodes = set()
    dangling_edges = []
    for node_idx, node in enumerate(bubbler):
        inp_edges, inp_op_idx, out_edges, out_op_idx = get_edge_info(node, eaten_nodes)

        # identify auxiliary dangling edges that do not participate in swallowing this node
        aux_edges = [ edge for edge in dangling_edges if edge not in inp_edges ]

        inp_num = len(inp_edges) # number of input qubits to the "bare" swallowing operator
        out_num = len(out_edges) # number of output qubits to the "bare" swallowing operator
        act_num = max(inp_num, out_num) # number of qubits the swallowing operator acts on
        aux_num = len(aux_edges) # number of auxiliary (bystander) qubits

        # update max_qubits, given the number of qubits required for this step
        if node_idx != 0 and node_idx != net_size-1:
            max_qubits = max(max_qubits, aux_num + act_num + 1)

        # get the tensor associated with this node, reordering axes as necessary
        op_tensor = tf.transpose(node.get_tensor(), out_op_idx + inp_op_idx)
        op_matrix = tf.reshape(op_tensor, (2**out_num, 2**inp_num))
        vals_D, _, _ = tf.linalg.svd(op_matrix)
        log_net_norm += np.log(vals_D.numpy().max())

        # add to our list of "eaten" nodes, update the list of dangling edges
        eaten_nodes.add(node)
        for edge in inp_edges:
            dangling_edges.remove(edge)
        dangling_edges = out_edges + dangling_edges

    # compute value of network
    tn.contractors.naive(net)
    log_net_val = np.log(net.get_final_node().tensor.numpy())

    net_prob = np.exp(2 * (log_net_val - log_net_norm))
    return net_prob, log_net_norm, max_qubits
