#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from circuit_cutter import cut_circuit
from fragment_simulator import get_circuit_probabilities, \
    get_fragment_distributions, combine_fragment_distributions

from demo_methods import *

backend_simulator = "statevector_simulator"
# backend_simulator = "qasm_simulator"

# number of shots per condition on each fragment
# irrelevant if using the statevector simulator
shots = 10**4

# some printing options
print_circuits = True
print_distributions = False
print_recombination_updates = True

# throw out negative terms when reconstructing a probability distribution?
discard_negative_terms = False

# convert amplitudes to probabilities?
# only relevant if using the statevector simulator
force_probabilities = False

# random number seed
seed = 0

# numer of qubits and layers in the random unitary circuit
qubits = 20
layers = 3

##########################################################################################
# build and cut a random unitary circuit
circuit = random_circuit(qubits, layers, seed)
cuts = [ (circuit.qubits[qubits//2], op+1) for op in range(1,2*layers) ]
fragments, wire_path_map = cut_circuit(circuit, cuts)

circ_wires = tuple(circuit.qubits)
frag_wires = tuple([ tuple(fragment.qubits) for fragment in fragments ])

if print_circuits:
    print("original circuit:")
    print(circuit)
    print()
    print(fragment_info(fragments, wire_path_map))

##########################################################################################
# simulate circuit, simulate fragments, recombine fragments

circuit_distribution = get_circuit_probabilities(circuit, seed_simulator = seed)

frag_distributions \
    = get_fragment_distributions(fragments, wire_path_map, backend_simulator, shots = shots,
                                 seed_simulator = seed, seed_transpiler = seed)
reconstructed_distribution \
    = combine_fragment_distributions(frag_distributions, wire_path_map, circ_wires, frag_wires,
                                     discard_negative_terms = discard_negative_terms,
                                     status_updates = print_recombination_updates)

if print_distributions:
    print("full circuit probability distribution:")
    print(dist_text(circuit_distribution))

    print()
    print("reconstructed probability distribution:")
    print(dist_text(reconstructed_distribution))
    print()

print("fidelity:", distribution_fidelity(reconstructed_distribution, circuit_distribution))
print("relative entropy:", relative_entropy(reconstructed_distribution, circuit_distribution))

# num_vals = np.prod(circuit_distribution.shape)
# uniform_dist = tf.constant([1/num_vals] * num_vals,
#                            shape = circuit_distribution.shape,
#                            dtype = circuit_distribution.dtype)
# print()
# print(distribution_fidelity(uniform_dist, circuit_distribution))
# print(relative_entropy(uniform_dist, circuit_distribution))
