#!/usr/bin/env python3

import tensorflow as tf

# return a pure state of given number of qubits
def idx_state(index, qubits, tf_dtype):
    qubits = max(0,qubits)
    dim = 2**qubits
    return tf.reshape(tf.one_hot(index % dim, dim, dtype = tf_dtype), (2,)*qubits)

# simulate the contracting of a tensor network using a computer
# uses the method described in arxiv.org/abs/0805.0040
# accepts: a list of TensorNetwork nodes in bubbling order
# returns:
# (i) the value of the tensor network
# (ii) probability of "success", i.e. finding all ancillas in |0>,
# (iii) the number of qubits necessary to run the computation, and
# (iv) the maximum number of qubits addressed by a single unitary
def quantum_contract(nodes, print_status = False, fake = False, tf_dtype = tf.float64):
    zero_state = lambda qubits : idx_state(0, qubits, tf_dtype)
    tf_eye = tf.eye(2, dtype = tf_dtype)
    tf_iY = tf.constant([[0,1],[-1,0]], dtype = tf_dtype)

    max_qubits_mem = 0
    max_qubits_op = 0
    net_norm = 1
    state = zero_state(0)

    eaten_nodes = set()
    dangling_edges = []

    for node_idx, node in enumerate(nodes):
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
        act_num = max(inp_num, out_num) # number of qubits the swallowing operator acts on
        aux_num = len(aux_edges) # number of auxiliary (bystander) qubits
        assert(aux_num == len(state.shape) - inp_num) # sanity check

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
        vals_D, op_V_L, op_V_R = tf.svd(op_matrix)

        # rotate into the diagonal basis of D
        state = tf.reshape(state, (2**act_num,) + (2,)*(len(state.shape)-act_num))
        state = tf.tensordot(tf.transpose(op_V_R, conjugate = True), state, axes = [[1],[0]])

        # normalize D by its operator norm, keeping track of the norm independently
        norm_D = max(vals_D.numpy())
        normed_vals_D = vals_D/norm_D
        net_norm *= norm_D

        # construct the unitary action of D
        op_U_D = tf.tensordot(tf_eye, tf.diag(normed_vals_D), axes = 0) \
               + tf.tensordot(tf_iY, tf.diag(tf.sqrt(1-normed_vals_D**2)), axes = 0)

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

        # number of qubits required for this step,
        # and the number of qubits the current unitarized swallowing operators acts on
        node_qubits_mem = len(state.shape) + max(0, inp_num-out_num) + 1
        node_qubits_op = op_U_D.shape[-1].bit_length()-1

        max_qubits_mem = max(max_qubits_mem, node_qubits_mem)
        max_qubits_op = max(max_qubits_op, node_qubits_op)

        # print status info
        if print_status:
            # the node we just swallowed
            print("node:", node_idx, node)

            # the number of memory qubits,
            # and the number of qubits the unitary swallowing operator acts on
            print("qubits (mem, op):", node_qubits_mem, node_qubits_op)

            # the norm of the state after swallowing this node and projecting out ancillas
            state_norm = tf.tensordot(tf.transpose(state, conjugate = True), state,
                                  axes = [ list(range(len(state.shape))) ]*2).numpy()
            print("norm:", state_norm)
            print("-"*10)

    assert(state.shape == ()) # we should have contracted out the entire state

    net_prob = state.numpy()**2
    net_val = net_norm * state.numpy()
    return net_val, net_prob, max_qubits_mem, max_qubits_op
