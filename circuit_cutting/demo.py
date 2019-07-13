#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from circuit_cutter import cut_circuit
from fragment_simulator import get_circuit_distribution, \
    get_fragment_distributions, combine_fragment_distributions


backend_simulator = "statevector_simulator"
# backend_simulator = "qasm_simulator"
shots = 10**5

# print status updates during recombination?
status_updates = False

##########################################################################################
# construct circuit of random local 2-qubit gates that we can cut

qubits = 3
layers = 2

qreg = qs.QuantumRegister(qubits, "q")
circuit = qs.QuantumCircuit(qreg)

np.random.seed(0)
def random_unitary():
    return qs.quantum_info.random.utils.random_unitary(4)

for idx, qubit in enumerate(qreg):
    circuit.u0(idx, qubit)

for layer in range(layers):
    for odd_links in range(2):
        for jj in range(odd_links, qubits-1, 2):
            random_gate = random_unitary()
            circuit.append(random_gate, [ qreg[jj], qreg[jj+1] ])

for idx, qubit in enumerate(qreg):
    circuit.u0(idx, qubit)

cuts = [ (qreg[qubits//2], op+1) for op in range(1,2*layers) ]
fragments, frag_wiring, frag_stitches = cut_circuit(circuit, *cuts)

circ_wires = circuit.qubits
frag_wires = [ fragment.qubits for fragment in fragments ]

print("original circuit:")
print(circuit)

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


circuit_distribution = get_circuit_distribution(circuit)

frag_distributions \
    = get_fragment_distributions(fragments, frag_stitches, backend_simulator, shots = shots)
reconstructed_distribution \
    = combine_fragment_distributions(frag_distributions, frag_stitches, frag_wiring,
                                     frag_wires, circ_wires, status_updates = status_updates)

print()
print("full circuit probability distribution:")
print_dist(circuit_distribution)

print()
print("reconstructed probability distribution:")
print_dist(reconstructed_distribution)

print()
print("fidelity:", distribution_fidelity(circuit_distribution, reconstructed_distribution))
