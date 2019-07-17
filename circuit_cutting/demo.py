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
seed = 0

print_circuits = False
print_distributions = False
print_recombination_updates = False

# throw out negative terms when reconstructing a probability distribution?
discard_negative_terms = False

##########################################################################################
# construct circuit of random local 2-qubit gates that we can cut

qubits = 20
layers = 3

qreg = qs.QuantumRegister(qubits, "q")
circuit = qs.QuantumCircuit(qreg)

if seed is not None: np.random.seed(seed)
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
fragments, frag_stitches, frag_wiring = cut_circuit(circuit, *cuts)

circ_wires = circuit.qubits
frag_wires = [ fragment.qubits for fragment in fragments ]

if print_circuits:
    print("original circuit:")
    print(circuit)

    print()
    for jj, fragment in enumerate(fragments):
        print("fragment index:", jj)
        print(fragment)
        print("--------------------")

    print()
    print("fragment stitches:")
    for old_wire, new_wire in sorted(frag_stitches.items()):
        print(*old_wire, "-->", *new_wire)

    print()
    print("fragment wiring:")
    for old_wire, new_wire in sorted(frag_wiring.items()):
        print(old_wire, "-->", *new_wire)

##########################################################################################
# get distribution functions over measurement outcomes and print results

# convert a 1-D array of values into a string
def to_str(vals):
    return "".join([ str(val) for val in vals ])

# text to print a probability distribution
def dist_text(dist):
    if type(dist) is tf.SparseTensor:
        text_generator = ( f"{to_str(idx.numpy())} {val.numpy()}"
                           for idx, val in zip(dist.indicses, dist.values) )
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

    return abs( tf.reduce_sum(tf.sqrt(tf.math.multiply(dist_0, dist_1))).numpy() )**2

# relative entropy S( P | Q ) \equiv tr( P log(P/Q) ) in bits
# interpretation: information gained upon using Q (the "actual distribution")
#                 rather than P (the "estimate" of Q)
def relative_entropy(approx_dist, actual_dist):
    if tf.SparseTensor in [ type(approx_dist), type(actual_dist) ]: return None
    dist_to_sum = approx_dist * tf.math.log(approx_dist/actual_dist)
    return tf.reduce_sum( dist_to_sum ).numpy() / np.log(2)


circuit_distribution = get_circuit_distribution(circuit)

frag_distributions \
    = get_fragment_distributions(fragments, frag_stitches, backend_simulator, shots = shots)
reconstructed_distribution \
    = combine_fragment_distributions(frag_distributions, frag_stitches, frag_wiring,
                                     frag_wires, circ_wires,
                                     discard_negative_terms = discard_negative_terms,
                                     status_updates = print_recombination_updates)

if print_distributions:
    print()
    print("full circuit probability distribution:")
    print(dist_text(circuit_distribution))

    print()
    print("reconstructed probability distribution:")
    print(dist_text(reconstructed_distribution))
    print()
print("fidelity:", distribution_fidelity(reconstructed_distribution, circuit_distribution))
print("relative entropy:", relative_entropy(reconstructed_distribution, circuit_distribution))
