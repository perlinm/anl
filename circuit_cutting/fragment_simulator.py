#!/usr/bin/env python3

import numpy as np
import qiskit as qs

from itertools import product as set_product
from functools import reduce

##########################################################################################
# this script contains methods to simulate circuit fragments
# and combine fragment simulation results
##########################################################################################

##########################################################################################
# methods to simulate circuit fragments and collect conditional probability distributions
# over measurement outcomes
##########################################################################################

# return an empty circuit on the registers of an existing circuit
def empty_circuit(circuit):
    registers = set( wire[0] for wire in circuit.qubits + circuit.clbits )
    return qs.QuantumCircuit(*registers)

# act with the given gates acting on the given qubits
def act_gates(circuit, gates, *qubits):
    new_circuit = empty_circuit(circuit)
    for gate in gates:
        new_circuit.append(gate, qargs = list(qubits))
    return new_circuit

# get a distribution over measurement outcomes for a circuit
def get_circuit_distribution(circuit, backend = "statevector_simulator"):
    simulator = qs.Aer.get_backend(backend)

    if backend == "statevector_simulator":
        result = qs.execute(circuit, simulator).result()
        state_vector = result.get_statevector(circuit)
        return np.reshape(abs(state_vector)**2, (2,)*(len(state_vector).bit_length()-1))
    else:
        print("backend not supported:", backend)
        return None

# sequences of gates to (i) prepare initial states, (ii) measure in different bases
gates = qs.extensions.standard
frag_init_preps = [ ( "+Z", [ ] ),
                    ( "-Z", [ gates.XGate() ]  ),
                    ( "+X", [ gates.HGate() ] ),
                    ( "-X", [ gates.XGate(), gates.HGate() ] ),
                    ( "+Y", [ gates.HGate(), gates.SGate() ] ),
                    ( "-Y", [ gates.XGate(), gates.HGate(), gates.SGate() ] ) ]
frag_exit_apnds = [ ( "Z", [ ] ),
                    ( "X", [ gates.HGate() ] ),
                    ( "Y", [ gates.SGate().inverse(), gates.HGate() ] ) ]

# get a distributions over measurement outcomes for a circuit fragment
# accepts a list of fragments, and lists of wires (i.e. in that fragment) that are
#   respectively "initialization" wires (on which to prepare states)
#   and "exit wires" (on which to make measurements)
# returns an (unnormalized) conditional distribution over measurement outcomes
#   in dictionary format:
# { ( set( ( < initialized wire >, < initialized state >   ) ),
#     set( ( < measured wire >,    < measurement outcome > ) ) :
#   < distribution over non-exit-wire measurement outcomes,
#     conditional on the exit-wire measurement outcomes > }
def get_fragment_distribution(fragment, init_wires = None, exit_wires = None):
    if init_wires: # if we have wires to initialize into various states

        # pick the first init wire for state initialization
        init_wire, other_wires = init_wires[0], init_wires[1:]

        frag_dist = {} # the overall conditional distribution over measurement outcomes
        for state, prepend_op in frag_init_preps: # for each state we will prepare

            # construct the circuit to prepare the state we want
            prep_circuit = act_gates(fragment, prepend_op, init_wire)

            # get the distribution over measurement outcomes for each prepared state
            init_frag_dist = get_fragment_distribution(prep_circuit + fragment,
                                                       other_wires, exit_wires)

            # add to our collection of conditional distributions,
            # indexing init_frag_dist by state that we prepared
            for all_keys, dist in init_frag_dist.items():
                init_keys, exit_keys = all_keys
                new_init_keys = init_keys.union({ ( init_wire, state ) })
                frag_dist[new_init_keys, exit_keys] = dist

        return frag_dist

    if exit_wires: # if we have wires to measure in various bases

        # sort exit wires by index to ensure proper deletion of axes
        #   for example, if we wish to delete axes 0 and 3,
        #   then we need to delete axis 3 before deleting axis 0,
        #   as otherwise the bit at axis 3 will change position
        exit_wires = sorted(exit_wires, key = lambda wire : fragment.qubits.index(wire) )

        # pick the first exit wire for measurement
        exit_wire, other_wires = exit_wires[0], exit_wires[1:]

        # determine axis to delete (i.e. project out) when constructing
        # a distribution over all measurement outcomes that is conditional
        # on a particular measurement outcome
        del_axis = -fragment.qubits.index(exit_wire)-1

        frag_dist = {} # the overall conditional distribution over measurement outcomes
        for measurement, append_op in frag_exit_apnds: # for each measurement basis

            # construct the circuit to measure in the desired basis
            apnd_circuit = act_gates(fragment, append_op, exit_wire)

            # get the conditional distribution for each measurement basis
            exit_frag_dist = get_fragment_distribution(fragment + apnd_circuit,
                                                       init_wires, other_wires)

            # add to our collection of conditional distributions, indexing exit_frag_dist
            # by the measured basis and measurement outcome on the exit wire
            for all_keys, dist in exit_frag_dist.items():
                init_keys, exit_keys = all_keys
                for outcome, bit_state in [ ( "+", 0 ), ( "-", 1 ) ]:
                    new_exit_keys = exit_keys.union({ ( exit_wire, outcome + measurement ) })

                    # project onto a given exit-wire measurement outcome
                    new_dist = np.delete(dist, 1-bit_state, axis = del_axis)
                    new_dist = np.reshape(new_dist, dist.shape[:-1])

                    frag_dist[init_keys, new_exit_keys] = new_dist

        return frag_dist

    # if no init_frag_wires and no exit_frag_wires
    distribution = get_circuit_distribution(fragment)
    return { ( frozenset(), frozenset() ) : distribution }

##########################################################################################
# methods to combine conditional probability distributions from fragments,
# and reconstruct the probability distribution of a stitched-together circuit
##########################################################################################

# identify initialization and exit wires for each fragment
# accepts dictionary of stitches and total number of fragments
# returns two lists: one for initialization wires and one exit wires
# the frag_idx element of each list is itself a list of init/exit wires for that fragment
def sort_init_exit_wires(frag_stitches, num_fragments):
    # all exit/init wires: ( frag_index, wire )
    exit_frag_wires_list, init_frag_wires_list = zip(*frag_stitches.items())

    # collect exit/init wires into list indexed by a fragment number,
    # removing the fragment index from the wire itself
    init_wires = tuple([ [ frag_wire[1] for frag_wire in init_frag_wires_list
                           if frag_wire[0] == frag_idx ]
                         for frag_idx in range(num_fragments) ])
    exit_wires = tuple([ [ frag_wire[1] for frag_wire in exit_frag_wires_list
                           if frag_wire[0] == frag_idx ]
                         for frag_idx in range(num_fragments) ])
    return init_wires, exit_wires

# return a dictionary mapping fragment output wires to corresponding wires of a circuit
def frag_wire_map(circuit_wires, fragment_wiring, fragment_stitches):
    wire_map = {}
    for original_wire in circuit_wires:
        frag_wire = fragment_wiring[original_wire]
        while frag_wire in fragment_stitches:
            frag_wire = fragment_stitches[frag_wire]
        wire_map[frag_wire] = original_wire
    return wire_map

# rearrange the ordering of wires in a distribution over measurement outcomes
def rearranged_wires(distribution, old_wire_order, new_wire_order, wire_map = None):
    if wire_map is None: wire_map = { wire : wire for wire in old_wire_order }
    current_wire_order = [ wire_map[wire] for wire in old_wire_order ]
    wire_permutation = [ current_wire_order.index(wire) for wire in new_wire_order ]
    axis_permutation = [ len(new_wire_order) - 1 - idx for idx in wire_permutation ][::-1]
    return distribution.transpose(*axis_permutation)

# all allowed assignments of measurement outcomes and states to the ends of a stitch
stitch_assigments = [ ("+Z",)*2, ("-Z",)*2 ] \
                  + [ ( dir_exit + op, dir_init + op)
                      for op in [ "X", "Y" ]
                      for dir_exit in [ "+", "-" ]
                      for dir_init in [ "+", "-" ] ]

# simulate fragments and stitch together results
# accepts a list of circuit fragments and a dictionary for how to stitch them together
# return a distribution over measurement outcomes on the stitched-together circuit
def simulate_and_combine(fragments, frag_stitches,
                         frag_wiring = None, wire_order = None):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = sort_init_exit_wires(frag_stitches, len(fragments))

    # compute conditional distributions over measurement outcomes for all fragments
    frag_dists = [ get_fragment_distribution(ff, ii, ee)
                   for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

    # identify the order of wires in a combined/reconstructed distribution over outcomes
    combined_wire_order = [ ( frag_idx, qubit )
                            for frag_idx, fragment in enumerate(fragments)
                            for qubit in fragment.qubits
                            if ( frag_idx, qubit ) not in frag_stitches.keys() ]

    # initialize an empty combined distribution over outcomes
    combined_dist = np.zeros((2,)*len(combined_wire_order))

    # loop over all assigments of measurement outcomes / initialized states
    #   at the exit / init wires of all stitches
    for assignment in set_product(stitch_assigments, repeat = len(frag_stitches)):

        # collect the assignments of exit/init outcomes/states for each fragment
        frag_exit_keys = [ set() for _ in range(len(fragments)) ]
        frag_init_keys = [ set() for _ in range(len(fragments)) ]
        for stitch_idx, ( exit_frag_wire, init_frag_wire ) in enumerate(frag_stitches.items()):
            exit_frag_idx, exit_wire = exit_frag_wire
            init_frag_idx, init_wire = init_frag_wire
            frag_exit_keys[exit_frag_idx].add(( exit_wire, assignment[stitch_idx][0] ))
            frag_init_keys[init_frag_idx].add(( init_wire, assignment[stitch_idx][1] ))

        # get the conditional probability distribution at each fragment
        dist_factors = [ frag_dist[frozenset(init_keys), frozenset(exit_keys)]
                         for frag_dist, init_keys, exit_keys
                         in zip(frag_dists, frag_init_keys, frag_exit_keys) ]

        # get the scalar factor associated with this assignment of exit/init outcomes/states
        scalar_factor = np.product([ ( 1 if fst[0] == snd[0] else -1 ) *
                                     ( 1 if fst[1] == "Z" else 1/2 )
                                     for fst, snd in assignment ])

        # add to the combined distribution over measurement outcomes
        combined_dist += scalar_factor * reduce(np.multiply.outer, dist_factors[::-1])

    # if we did not provide wiring info,
    #   return the combined distribution over measurement outcomes,
    #   together with the order of wires
    if frag_wiring is None and wire_order is None:
        return combined_dist, combined_wire_order

    # otherwise, return the combined distribution over measurement outcomes
    #   with wires sorted in the provided order
    else:
        assert( frag_wiring is not None and wire_order is not None )
        wire_map = frag_wire_map(wire_order, frag_wiring, frag_stitches)
        return rearranged_wires(combined_dist, combined_wire_order, wire_order, wire_map)
