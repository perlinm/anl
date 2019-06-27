#!/usr/bin/env python3

import numpy as np
import qiskit as qs

from circuit_cutter import cut_circuit
from fragment_simulator import get_circuit_distribution, simulate_and_combine

##########################################################################################
# construct circuit of random local 2-qubit gates that we can cut

qubits = 3
layers = 2

qreg = qs.QuantumRegister(qubits, "q")
circ = qs.QuantumCircuit(qreg)

for layer in range(layers):
    for odd_links in range(2):
        for jj in range(odd_links, qubits-1, 2):
            random_gate = qs.quantum_info.random.utils.random_unitary(4)
            circ.append(random_gate, [ qreg[jj], qreg[jj+1] ])

cuts = [ (qreg[qubits//2], op) for op in range(1,2*layers) ]
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

circ_dist = get_circuit_distribution(circ)
combined_dist = simulate_and_combine(fragments, frag_stitches, frag_wiring, circ.qubits)

# convert a distribution function into dictionary format
def dist_vec_to_dict(distribution):
    return { "".join([ str(bb) for bb in idx ]) : distribution[idx]
             for idx in np.ndindex(distribution.shape)
             if distribution[idx] != 0 }

# fidelity of two distribution functions: tr( sqrt(rho_0 * rho_1) )
def distribution_fidelity(dist_0, dist_1):
    return sum( np.sqrt(dist_0[idx] * dist_1[idx])
                for idx in np.ndindex(dist_0.shape) )

print()
print("full circuit probability distribution")
circ_wires = circ.qubits
for key, val in dist_vec_to_dict(circ_dist).items():
    print(key, val)

print()
print("reconstructed probability distribution")
for key, val in dist_vec_to_dict(combined_dist).items():
    print(key, val)

print()
print("fidelity:", distribution_fidelity(abs(circ_dist), abs(combined_dist)))
