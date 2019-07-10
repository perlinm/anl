#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from circuit_cutter import cut_circuit
from fragment_simulator import get_circuit_distribution, simulate_and_combine

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

# convert a distribution function into dictionary format
def dist_vec_to_dict(distribution, cutoff = 1e-10):
    if type(distribution) is dict: return distribution

    if type(distribution) is tf.SparseTensor:
        idx_val_zip = zip(distribution.indices, distribution.values)
        dist_dict = { to_str(idx.numpy()) : val.numpy()
                      for idx, val in idx_val_zip
                      if abs(val.numpy()) > cutoff }
        return dist_dict

    if type(distribution) is not np.ndarray:
        return dist_vec_to_dict(np.array(distribution))

    return { to_str(idx) : distribution[idx]
             for idx in np.ndindex(distribution.shape)
             if abs(distribution[idx]) > cutoff }

# fidelity of two distribution functions: tr( sqrt(rho_0 * rho_1) )
def distribution_fidelity(dist_0, dist_1):
    if type(dist_0) is tf.SparseTensor and type(dist_1) is not tf.SparseTensor:
        return sum( np.sqrt(complex(value) * complex(dist_1[tuple(idx)]))
                    for idx, value in zip(dist_0.indices, dist_0.values) )

    if type(dist_1) is tf.SparseTensor and type(dist_0) is not tf.SparseTensor:
        return distribution_fidelity(dist_1, dist_0)

    return tf.reduce_sum(tf.sqrt(tf.math.multiply(dist_0, dist_1))).numpy()


circ_dist = get_circuit_distribution(circ)

combined_dist = simulate_and_combine(fragments, frag_stitches, frag_wiring, circ.qubits)
# combined_dist = simulate_and_combine(fragments, frag_stitches, frag_wiring, circ.qubits,
                                     # backend_simulator = "qasm_simulator", shots = 1000)


print()
print("full circuit probability distribution")
for key, val in dist_vec_to_dict(circ_dist).items():
    print(key, val)

print()
print("reconstructed probability distribution")
for key, val in dist_vec_to_dict(combined_dist).items():
    print(key, val)

print()
print("fidelity:", distribution_fidelity(circ_dist, combined_dist))
