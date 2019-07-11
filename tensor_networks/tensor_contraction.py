#!/usr/bin/env python3

import os
import numpy as np

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn
from tensornetwork.contractors import greedy_contractor

# return an indexed pure state of given number of qubits
def idx_state(index, qubits, dtype = tf.float64):
    qubits = max(0,qubits)
    dim = 2**qubits
    return tf.reshape(tf.one_hot(index % dim, dim, dtype = dtype), (2,)*qubits)

# return the trivial (zero) state of a given number of qubits
def zero_state(qubits, dtype = tf.float64):
    return idx_state(0, qubits, dtype)

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

# convert an isometry T : A --> B into a unitary operator U : A_B --> A_B
# where the dimensions |A_B| = |B| and A_B = A \otimes ancillas
def to_unitary(isometry):
    assert(len(isometry.shape) == 2)
    trgt_dim, base_dim = isometry.shape # dimensions of base and target spaces A and B
    if trgt_dim == base_dim: return isometry

    # identity operator on the target space B, and projector onto T(A)
    identity = tf.eye(trgt_dim, dtype = isometry.dtype)
    trgt_projector = tf.linalg.matmul(isometry, isometry, adjoint_b = True)

    # isometry V : A --> A_B defined by V(a) = a \otimes ancillas,
    # and a projector onto V(A)
    base_vecs = tf.eye(*isometry.shape, dtype = isometry.dtype)
    base_projector = tf.linalg.matmul(base_vecs, base_vecs, transpose_b = True)

    # construct a minimal unitary rotating V(A) into *T(A),
    # where * denotes an embedding of B into A_B
    trgt_reflector = identity - 2 * trgt_projector
    base_reflector = identity - 2 * base_projector
    minimal_unitary = tf.linalg.sqrtm(tf.linalg.matmul(trgt_reflector, base_reflector))

    # the minimal unitary U* above only gives us U : A_B --> A_B
    #   up to some rotation R of the standard basis on A; identify the basis rotation R
    base_rot = tf.matmul(minimal_unitary, isometry, adjoint_a = True)[:base_dim,:base_dim]

    # lift the rotation R in A to a rotation R* in A_B via R* = V \circ R
    base_num, trgt_num = base_dim.bit_length()-1, trgt_dim.bit_length()-1
    ancilla_num = trgt_num - base_num
    ancilla_dim = 2**ancilla_num
    ancilla_identity = tf.eye(ancilla_dim, dtype = isometry.dtype)
    ancilla_identity = tf.reshape(ancilla_identity, (2,2)*ancilla_num)
    trgt_rot = tf.tensordot(ancilla_identity, base_rot, axes = 0) # <-- this is R*

    # rearrange the indices in R* properly
    idx_perm = list(range(ancilla_num)) \
             + list(range(2*ancilla_num, 2*ancilla_num + base_num)) \
             + list(range(ancilla_num, 2*ancilla_num)) \
             + list(range(2*ancilla_num + base_num, 2*ancilla_num + 2*base_num))
    trgt_rot = tf.reshape(trgt_rot, (2,2)*trgt_num)
    trgt_rot = tf.transpose(trgt_rot, idx_perm)
    trgt_rot = tf.reshape(trgt_rot, (trgt_dim,)*2)

    # return U = U* R*
    return tf.matmul(minimal_unitary, trgt_rot)

# attach (remove) some qubits to (from) a state
def attach_qubits(state, add_num):
    if add_num <= 0: return state
    return tf.tensordot(zero_state(add_num, dtype = state.dtype), state, axes = 0)
def remove_qubits(state, del_num):
    if del_num <= 0: return state
    return tf.tensordot(zero_state(del_num, dtype = state.dtype), state,
                        axes = [ list(range(del_num)) ]*2)

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
    state = zero_state(0, dtype = dtype)

    eaten_nodes = set()
    dangling_edges = []
    for node_idx in bubbler:
        node = nodes[node_idx]
        inp_edges, inp_op_idx, out_edges, out_op_idx = get_edge_info(node, eaten_nodes)

        # identify auxiliary dangling edges that do not participate in swallowing this node
        aux_edges = [ edge for edge in dangling_edges if edge not in inp_edges ]

        # identify the numbers of different kinds of qubits
        inp_num = len(inp_edges) # number of input qubits to the swallowing operator
        out_num = len(out_edges) # number of output qubits to the swallowing operator
        aux_num = len(aux_edges) # number of auxiliary (bystander) qubits
        assert(aux_num == len(state.shape) - inp_num) # sanity check

        # rearrange the order of qubits in the state
        inp_state_idx = [ dangling_edges.index(edge) for edge in inp_edges ]
        aux_state_idx = [ dangling_edges.index(edge) for edge in aux_edges ]
        state = tf.transpose(state, inp_state_idx + aux_state_idx)

        # get the tensor/matrix associated with this node, reordering axes as necessary
        swallow_tensor = tf.transpose(tf.cast(node.get_tensor(), dtype),
                                      out_op_idx + inp_op_idx)
        swallow_matrix = tf.reshape(swallow_tensor, (2**out_num, 2**inp_num))

        # perform singular-value decomposition of swallowing_matrix:
        #   S = V_L @ diag(D) @ V_R^\dag,
        # where V_L, V_R are isometric; and D is positive semi-definite
        vals_D, mat_V_L, mat_V_R = tf.linalg.svd(swallow_matrix)

        # convert isometries into unitaries on an extension of the domain
        #   achieved by attaching trivial (|0>) states of extra qubits
        mat_V_L = to_unitary(mat_V_L)
        mat_V_R = to_unitary(mat_V_R)

        # normalize D by its operator norm, keeping track of the norm independently
        norm_D = max(vals_D.numpy())
        normed_vals_D = vals_D/norm_D
        log_net_norm += np.log(norm_D)

        # construct the unitary action of D
        mat_U_D = tf.tensordot(tf_Z, tf.linalg.diag(normed_vals_D), axes = 0) \
                + tf.tensordot(tf_X, tf.linalg.diag(tf.sqrt(1-normed_vals_D**2)), axes = 0)

        # rotate into the right-diagonal basis of the swallowing operator
        state = tf.reshape(state, (2**inp_num,) + (2,)*aux_num)
        state = tf.tensordot(tf.transpose(mat_V_R, conjugate = True), state, axes = [[1],[0]])
        state = tf.reshape(state, (2,)*(inp_num+aux_num))

        # if we lose qubits upon swallowing this operator, then project them out now
        state = remove_qubits(state, max(0, inp_num - out_num))

        # attach ancilla, act on the state by the unitary U_D, and project out the ancilla
        state = tf.reshape(state, (2**min(inp_num,out_num),) + (2,)*aux_num)
        state = attach_qubits(state, 1)
        state = tf.tensordot(mat_U_D, state, axes = [ [1,3], [0,1] ])
        state = remove_qubits(state, 1)

        # if we gain qubits upon swallowing this tensor, then attach them now
        state = attach_qubits(state, max(0, out_num - inp_num))

        # rotate back from the left-diagonal basis of the swallowing operator
        state = tf.reshape(state, (2**out_num,) + (2,)*aux_num)
        state = tf.tensordot(mat_V_L, state, axes = [ [1], [0] ])
        state = tf.reshape(state, (2,)*(out_num+aux_num))

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
    if bubbler is None: bubbler = nodes.key()
    log_net_norm = 0
    eaten_nodes = set()
    dangling_edges = []

    for node_idx in bubbler:
        node = nodes[node_idx]
        inp_edges, inp_op_idx, out_edges, out_op_idx = get_edge_info(node, eaten_nodes)

        # get the dimensions of the input/output spaces
        node_shape = node.get_tensor().shape
        inp_dim = np.prod([ node_shape[idx] for idx in inp_op_idx ], dtype = int)
        out_dim = np.prod([ node_shape[idx] for idx in out_op_idx ], dtype = int)

        # get the tensor associated with this node, reordering axes as necessary
        swallow_tensor = tf.transpose(node.get_tensor(), out_op_idx + inp_op_idx)
        swallow_matrix = tf.reshape(swallow_tensor, (out_dim, inp_dim))
        vals_D = tf.linalg.svd(swallow_matrix)[0].numpy()
        ### the tensorflow's svd algorithm sometimes gives a segfault...
        ### if a segfault occur, use numpy's svd algorithm instead
        # from numpy.linalg import svd
        # _, vals_D, _ = svd(swallow_matrix.numpy())
        norm_D = max(vals_D)
        log_net_norm += np.log(vals_D.max())

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
