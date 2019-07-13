#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from circuit_cutter import cut_circuit
from fragment_simulator import get_circuit_distribution, get_fragment_probabilities, \
    combine_fragment_probabilities

##########################################################################################
# construct circuit of random local 2-qubit gates that we can cut

qubits = 3
layers = 2

qreg = qs.QuantumRegister(qubits, "q")
circ = qs.QuantumCircuit(qreg)

np.random.seed(0)
def random_unitary():
    return qs.quantum_info.random.utils.random_unitary(4)

for idx, qubit in enumerate(qreg):
    circ.u0(idx, qubit)

for layer in range(layers):
    for odd_links in range(2):
        for jj in range(odd_links, qubits-1, 2):
            random_gate = random_unitary()
            circ.append(random_gate, [ qreg[jj], qreg[jj+1] ])

for idx, qubit in enumerate(qreg):
    circ.u0(idx, qubit)

cuts = [ (qreg[qubits//2], op+1) for op in range(1,2*layers) ]
fragments, frag_wiring, frag_stitches = cut_circuit(circ, *cuts)


print("original circuit:")
print(circ)

print()
for jj, fragment in enumerate(fragments):
    print("fragment index:", jj)
    print(fragment)
    print("--------------------")

print()
print("fragment wiring:")
for old_wire, new_wire in sorted(frag_wiring.items()):
    print(old_wire, "-->", *new_wire)

print()
print("fragment stitches:")
for old_wire, new_wire in sorted(frag_stitches.items()):
    print(*old_wire, "-->", *new_wire)

##########################################################################################
# get distribution functions over measurement outcomes and print results

# convert a 1-D array of values into a string
def to_str(vals):
    return "".join([ str(val) for val in vals ])

# print a probability distribution
def print_dist(distribution):
    if type(distribution) is tf.SparseTensor:
        for idx, val in zip(distribution.indices.numpy(), distribution.values.numpy()):
            print(to_str(idx), val)

    else:
        distribution = distribution.numpy()
        for idx in np.ndindex(distribution.shape):
            print(to_str(idx), distribution[idx])

# fidelity of two distribution functions: tr( sqrt(rho_0 * rho_1) )
def distribution_fidelity(dist_0, dist_1):
    if type(dist_0) is tf.SparseTensor and type(dist_1) is not tf.SparseTensor:
        dist_1 = dist_1.numpy()
        return sum( np.sqrt(complex(value) * complex(dist_1[tuple(idx)]))
                    for idx, value in zip(dist_0.indices, dist_0.values) )

    if type(dist_1) is tf.SparseTensor and type(dist_0) is not tf.SparseTensor:
        return distribution_fidelity(dist_1, dist_0)

    return tf.reduce_sum(tf.sqrt(tf.math.multiply(dist_0, dist_1))).numpy()


circ_dist = get_circuit_distribution(circ)

frag_probs = get_fragment_probabilities(fragments, frag_stitches)
# frag_probs = get_fragment_probabilities(fragments, frag_stitches,
                                        # backend_simulator = "qasm_simulator", shots = 1000)

frag_qubits = [ fragment.qubits for fragment in fragments ]
combined_dist = combine_fragment_probabilities(frag_probs, frag_stitches,
                                               frag_wiring, frag_qubits, circ.qubits)

print()
print("full circuit probability distribution:")
print_dist(circ_dist)

print()
print("reconstructed probability distribution:")
print_dist(combined_dist)

print()
print("fidelity:", distribution_fidelity(circ_dist, combined_dist))
