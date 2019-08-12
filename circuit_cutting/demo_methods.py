#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

# construct circuit of random local 2-qubit gates that we can cut
def random_circuit(qubits, layers, seed = None):

    qreg = qs.QuantumRegister(qubits, "q")
    circuit = qs.QuantumCircuit(qreg)

    if seed is not None: np.random.seed(seed)
    def random_unitary():
        return qs.quantum_info.random.utils.random_unitary(4)

    for idx, qubit in enumerate(qreg):
        circuit.u0(idx, qubit)
    circuit.barrier()

    for layer in range(layers):
        for odd_links in range(2):
            for jj in range(odd_links, qubits-1, 2):
                circuit.append(random_unitary(), [ qreg[jj], qreg[jj+1] ])

    circuit.barrier()
    for idx, qubit in enumerate(qreg):
        circuit.u0(idx, qubit)

    return circuit

# human-readable info about a circuit and fragments
def fragment_info(fragments, wire_path_map):
    text = ""
    for jj, fragment in enumerate(fragments):
        text += f"fragment index: {jj}\n"
        text += str(fragment.draw()) + "\n"
        text += "-"*50 + "\n"

    text += "wire paths:\n\n"
    for wire, path in wire_path_map.items():
        text += f"{wire} -->"
        if len(path) == 1:
            text += " {} {}".format(*path[0])
        if len(path) > 1:
           for frag_wire in path:
               text += "\n {} {}".format(*frag_wire)
        text += "\n"
    return text

# convert a 1-D array of values into a string
def to_str(vals):
    return "".join([ str(val) for val in vals ])

# text to print a probability distribution
def dist_text(dist):
    if type(dist) is tf.SparseTensor:
        text_generator = ( f"{to_str(idx.numpy())} {val.numpy()}"

                           for idx, val in zip(dist.indices, dist.values) )
    else:
        dist = dist.numpy()
        text_generator = ( f"{to_str(idx)} {dist[idx]}"
                           for idx in np.ndindex(dist.shape) )
    return "\n".join(text_generator)

# fidelity of two distribution functions: tr( sqrt(rho_0 * rho_1) )
def distribution_fidelity(dist_0, dist_1):
    if type(dist_0) is tf.SparseTensor and type(dist_1) is not tf.SparseTensor:
        dist_1 = dist_1.numpy()
        return sum( np.sqrt(complex(value) * complex(dist_1[tuple(idx)]))
                    for idx, value in zip(dist_0.indices, dist_0.values) )

    if type(dist_1) is tf.SparseTensor and type(dist_0) is not tf.SparseTensor:
        return distribution_fidelity(dist_1, dist_0)

    dist_prod = tf.math.multiply(dist_0, dist_1)
    sqrt_prod = tf.sqrt(tf.cast(dist_prod, tf.complex128))
    return tf.reduce_sum(sqrt_prod).numpy()

# relative entropy S( P | Q ) \equiv tr( P log(P/Q) ) in bits
# interpretation: information gained upon using Q (the "actual distribution")
#                 rather than P (the "estimate" of Q)
def relative_entropy(approx_dist, actual_dist):
    if tf.SparseTensor in [ type(approx_dist), type(actual_dist) ]: return None
    dist_to_sum = approx_dist * tf.math.log(approx_dist/actual_dist)
    return tf.reduce_sum( dist_to_sum ).numpy() / np.log(2)

def uniform_dist(dist):
    num_vals = np.prod(dist.shape)
    return tf.constant([1/num_vals] * num_vals, shape = dist.shape, dtype = dist.dtype)
