#!/usr/bin/env python3

import numpy as np
import qiskit as qs

from itertools import product as set_product
from functools import reduce

from circuit_cutter import cut_circuit
from fragment_simulator import get_circuit_distribution, get_fragment_distribution

qreg = qs.QuantumRegister(4, "q")
circ = qs.QuantumCircuit(qreg)
circ.h(qreg[0])
circ.cx(qreg[0], qreg[1])
circ.cx(qreg[1], qreg[2])
circ.cx(qreg[2], qreg[3])

circ.barrier()
circ.x(qreg[3])
circ.barrier()
for idx, qubit in enumerate(qreg):
    circ.u0(idx, qubit)

fragments, frag_wiring, frag_stitches = cut_circuit(circ, (qreg[1],1), (qreg[2],1))

print("original circuit:")
print(circ)

print()
for jj, fragment in enumerate(fragments):
    print("fragment index:", jj)
    print(fragment)
    print("--------------------")

print()
print("fragment wiring:")
for old_wire, new_wire in frag_wiring.items():
    print(old_wire, "-->", *new_wire)

print()
print("fragment stitches:")
for old_wire, new_wire in frag_stitches.items():
    print(*old_wire, "-->", *new_wire)


# convert a distribution vector to a dictionary taking bistrings to values
# note that the *first* bit in the bitstring corresponds to the *last* bit in a register
def dist_vec_to_dict(distribution):
    return { "".join([ str(bb) for bb in idx ]) : distribution[idx]
             for idx in np.ndindex(distribution.shape)
             if distribution[idx] != 0 }

# identify initialization and exit wires for each fragment
def identify_wires(frag_stitches, num_fragments):
    exit_wires_list, init_wires_list = zip(*frag_stitches.items())
    exit_wires = tuple([ [ wire[1] for wire in exit_wires_list if wire[0] == frag_idx ]
                              for frag_idx in range(num_fragments) ])
    init_wires = tuple([ [ wire[1] for wire in init_wires_list if wire[0] == frag_idx ]
                              for frag_idx in range(num_fragments) ])
    return exit_wires, init_wires

# return a dictionary mapping fragment output wires to corresponding wires of a circuit
def original_wires(original_wires, fragment_wiring, fragment_stitches):
    wire_map = {}
    for original_wire in original_wires:
        frag_wire = fragment_wiring[original_wire]
        while frag_wire in fragment_stitches:
            frag_wire = fragment_stitches[frag_wire]
        wire_map[frag_wire] = original_wire
    return wire_map

stitch_assigments = [ ("+Z",)*2, ("-Z",)*2 ] + [ ( dir_M + op, dir_S + op)
                                                 for op in [ "X", "Y" ]
                                                 for dir_M in [ "+", "-" ]
                                                 for dir_S in [ "+", "-" ] ]

def simulate_and_combine(fragments, frag_wiring, frag_stitches, wire_order):
    frag_wires = [ fragment.qubits for fragment in fragments ]
    exit_wires, init_wires = identify_wires(frag_stitches, len(fragments))
    frag_dists = [ get_fragment_distribution(ff, ii, ee)
                   for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

    combined_dist = np.zeros((2,)*len(circ.qubits))
    for assignment in set_product(stitch_assigments, repeat = len(frag_stitches)):
        frag_exit_keys = [ set() for _ in range(len(fragments)) ]
        frag_init_keys = [ set() for _ in range(len(fragments)) ]
        for stitch_idx, ( exit_frag_wire, init_frag_wire ) in enumerate(frag_stitches.items()):
            exit_frag_idx, exit_wire = exit_frag_wire
            init_frag_idx, init_wire = init_frag_wire
            frag_exit_keys[exit_frag_idx].add(( exit_wire, assignment[stitch_idx][0] ))
            frag_init_keys[init_frag_idx].add(( init_wire, assignment[stitch_idx][1] ))

        dist_factors = [ frag_dist[frozenset(init_keys), frozenset(exit_keys)]
                         for frag_dist, init_keys, exit_keys
                         in zip(frag_dists, frag_init_keys, frag_exit_keys) ]

        scalar_factor = np.product([ ( 1 if fst[0] == snd[0] else -1 ) *
                                     ( 1 if fst[1] == "Z" else 1/2 )
                                     for fst, snd in assignment ])

        combined_dist += scalar_factor * reduce(np.multiply.outer, dist_factors[::-1])

    combined_wires = [ ( frag_idx, qubit ) for frag_idx, fragment in enumerate(fragments)
                       for qubit in fragment.qubits
                       if ( frag_idx, qubit ) not in frag_stitches.keys() ]

    terminal_wire_map = original_wires(wire_order, frag_wiring, frag_stitches)
    current_wire_order = [ terminal_wire_map[wire] for wire in combined_wires ]
    wire_permutation = [ current_wire_order.index(wire) for wire in wire_order ]
    axis_permutation = [ len(wire_order) - 1 - idx for idx in wire_permutation ][::-1]

    return combined_dist.transpose(*axis_permutation)


def distribution_fidelity(dist_0, dist_1):
    fidelity = 0
    for idx in np.ndindex(dist_0.shape):
        fidelity += np.sqrt(abs(dist_0[idx] * dist_1[idx]))
    return fidelity


circ_dist = get_circuit_distribution(circ)
combined_dist = simulate_and_combine(fragments, frag_wiring, frag_stitches, circ.qubits)


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
print("distribution fidelity:", distribution_fidelity(circ_dist, combined_dist))
