#!/usr/bin/env python3

import numpy as np
import qiskit as qs

from itertools import product as set_product
from functools import reduce

##########################################################################################
# this script contains methods to simulate circuit fragments
# and combine fragment simulation results
##########################################################################################

# choose whether to prepare states in the ZXY or SIC basis
state_prep_basis = "SIC"

state_str_ZXY = [ "+Z", "-Z", "+X", "+Y" ]
state_vecs_ZYX =  [ (1,0,0), (-1,0,0), (0,1,0), (0,0,1) ]

state_idx_SIC = [ 0, 1, 2, 3 ]
state_vecs_SIC = [ [ 1, 1, 1 ], [ 1, -1, -1 ], [ -1, 1, -1 ], [ -1, -1, 1 ] ]
state_vecs_SIC = [ np.array(vec) / np.linalg.norm(vec) for vec in state_vecs_SIC ]

# class for a conditional distribution function
class conditional_distribution:
    def __init__(self, empty_dist = None):
        self.dist_dict = {}
        if empty_dist is not None:
            self.dist_dict[frozenset(), frozenset()] = empty_dist

    def add(self, init_keys, exit_keys, dist):
        self.dist_dict[frozenset(init_keys), frozenset(exit_keys)] = dist

    def items(self):
        return self.dist_dict.items()

    def __getitem__(self, key):
        state_preps, measure_ops = frozenset(key[0]), frozenset(key[1])
        try: return self.dist_dict[state_preps, measure_ops]
        except: None

        for wire, op in state_preps:
            vacancy = state_preps.difference({(wire,op)})

            if state_prep_basis == "ZXY":

                if op == "I":
                    state_up = vacancy.union({(wire,"+Z")})
                    state_dn = vacancy.union({(wire,"-Z")})
                    return self[state_up, measure_ops] + self[state_dn, measure_ops]

                if op in [ "-X", "-Y" ]:
                    state_I = vacancy.union({(wire,"I")})
                    state_XY = vacancy.union({(wire,f"+{op[1]}")})
                    return self[state_I, measure_ops] - self[state_XY, measure_ops]

                if type(op) is int:
                    vec = tuple(state_vecs_SIC[op])
                    state_vec = vacancy.union({(wire,vec)})
                    return self[state_vec, measure_ops]

                if type(op) is tuple:
                    assert(len(op) == 3)
                    dist_I = self[vacancy.union({(wire,"I")}), measure_ops]
                    dist_Z = self[vacancy.union({(wire,"+Z")}), measure_ops]
                    dist_X = self[vacancy.union({(wire,"+X")}), measure_ops]
                    dist_Y = self[vacancy.union({(wire,"+Y")}), measure_ops]
                    dist_ZXY = [ dist_Z, dist_X, dist_Y ]
                    return ( sum( val * dist for val, dist in zip(op, dist_ZXY ) ) +
                             dist_I * ( 1 - sum(op) ) / 2 )

            else: # state_prep_basis == "SIC"

                if op == "I":  return self[vacancy.union({(wire, (0,0,0))}), measure_ops]
                if op == "+Z": return self[vacancy.union({(wire,(+1,0,0))}), measure_ops]
                if op == "-Z": return self[vacancy.union({(wire,(-1,0,0))}), measure_ops]
                if op == "+X": return self[vacancy.union({(wire,(0,+1,0))}), measure_ops]
                if op == "-X": return self[vacancy.union({(wire,(0,-1,0))}), measure_ops]
                if op == "+Y": return self[vacancy.union({(wire,(0,0,+1))}), measure_ops]
                if op == "-Y": return self[vacancy.union({(wire,(0,0,-1))}), measure_ops]

                if type(op) is tuple:
                    fac = lambda vec : ( 1 + 3 * np.dot(op, vec) )
                    dist = lambda idx : self[vacancy.union({(wire,idx)}), measure_ops]
                    return 1/2 * sum( fac(vec) * dist(idx)
                                      for idx, vec in zip(state_idx_SIC, state_vecs_SIC) )

        for wire, op in measure_ops:
            vacancy = measure_ops.difference({(wire,op)})

            if op == "I":
                measure_up = vacancy.union({(wire,"+Z")})
                measure_dn = vacancy.union({(wire,"-Z")})
                return self[state_preps, measure_up] + self[state_preps, measure_dn]

            if type(op) is int:
                vec = tuple(state_vecs_SIC[op])
                measure_op = vacancy.union({(wire,vec)})
                return self[state_preps, measure_op]

            if type(op) is tuple:
                assert(len(op) == 3)
                dist_I = self[state_preps, vacancy.union({(wire,"I")})]
                dist_Z = self[state_preps, vacancy.union({(wire,"+Z")})]
                dist_X = self[state_preps, vacancy.union({(wire,"+X")})]
                dist_Y = self[state_preps, vacancy.union({(wire,"+Y")})]
                dist_ZXY = [ dist_Z, dist_X, dist_Y ]
                return ( sum( val * dist for val, dist in zip(op, dist_ZXY ) ) +
                         dist_I * ( 1 - sum(op) ) / 2 )

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
    if not gates: return new_circuit
    if type(gates) is not list: gates = [ gates ]
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

# return a gate that rotates |0> into a state pointing along the given vector
def vec_gate(vec):
    vec /= np.linalg.norm(vec)
    theta = np.arccos(vec[0]) # angle down from north pole, i.e. from |0>
    phi = np.arctan2(vec[2],vec[1]) # angle in the X-Y plane
    return qs.extensions.standard.U3Gate(theta, phi, 0)

# sequences of gates to (i) prepare initial states, (ii) measure in different bases
if state_prep_basis == "ZXY":
    init_states = state_str_ZXY
    init_vecs = state_vecs_ZYX
else: # state_prep_basis == "SIC"
    init_states = state_idx_SIC
    init_vecs = state_vecs_SIC
frag_init_preps = [ ( state, vec_gate(vec) )
                      for state, vec in zip(init_states, init_vecs) ]
frag_exit_apnds = [ ( "Z", None ),
                    ( "X", [ qs.extensions.standard.HGate() ] ),
                    ( "Y", [ qs.extensions.standard.SGate().inverse(),
                             qs.extensions.standard.HGate() ] ) ]

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

        # build the overall conditional distribution over measurement outcomes
        frag_dist = conditional_distribution()
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
                frag_dist.add(new_init_keys, exit_keys, dist)

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

        # build the overall conditional distribution over measurement outcomes
        frag_dist = conditional_distribution()
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

                    frag_dist.add(init_keys, new_exit_keys, new_dist)

        return frag_dist

    # if no init_frag_wires and no exit_frag_wires
    distribution = get_circuit_distribution(fragment)
    return conditional_distribution(distribution)

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

# compute conditional probability distributions for all circuit fragments
# accepts a list of fragments and stitch data
# returns a list of conditional probability distributions over measurement outcomes
def get_fragment_distributions(fragments, frag_stitches):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = sort_init_exit_wires(frag_stitches, len(fragments))

    # compute conditional distributions over measurement outcomes for all fragments
    return [ get_fragment_distribution(ff, ii, ee)
             for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

##########################################################################################
# methods to combine conditional probability distributions from fragments,
# and reconstruct the probability distribution of a stitched-together circuit
##########################################################################################

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

# all operators to insert on either end of a stitch
if state_prep_basis == "ZXY":
   stitch_assignments = [ f"{ss}{BB}"
                          for BB in [ "Z", "X", "Y" ]
                          for ss in [ "+", "-" ] ]
else: # state_prep_basis == "SIC"
    stitch_assignments = state_idx_SIC
stitch_assignments += [ "I" ]

# simulate fragments and stitch together results
# accepts a list of circuit fragments and a dictionary for how to stitch them together
# return a distribution over measurement outcomes on the stitched-together circuit
def simulate_and_combine(fragments, frag_stitches,
                         frag_wiring = None, wire_order = None):
    # get conditional probability distributions for of all circuit fragments
    frag_dists = get_fragment_distributions(fragments, frag_stitches)

    # identify the order of wires in a combined/reconstructed distribution over outcomes
    combined_wire_order = [ ( frag_idx, qubit )
                            for frag_idx, fragment in enumerate(fragments)
                            for qubit in fragment.qubits
                            if ( frag_idx, qubit ) not in frag_stitches.keys() ]

    # initialize an empty combined distribution over outcomes
    combined_dist = np.zeros((2,)*len(combined_wire_order))

    # loop over all assigments of stitch operators at all cut locations
    for assignment in set_product(stitch_assignments, repeat = len(frag_stitches)):

        # collect the assignments of exit/init outcomes/states for each fragment
        frag_exit_keys = [ set() for _ in range(len(fragments)) ]
        frag_init_keys = [ set() for _ in range(len(fragments)) ]
        for stitch_idx, ( exit_frag_wire, init_frag_wire ) in enumerate(frag_stitches.items()):
            exit_frag_idx, exit_wire = exit_frag_wire
            init_frag_idx, init_wire = init_frag_wire
            frag_exit_keys[exit_frag_idx].add(( exit_wire, assignment[stitch_idx] ))
            frag_init_keys[init_frag_idx].add(( init_wire, assignment[stitch_idx] ))

        # get the conditional probability distribution at each fragment
        dist_factors = [ frag_dist[init_keys, exit_keys]
                         for frag_dist, init_keys, exit_keys
                         in zip(frag_dists, frag_init_keys, frag_exit_keys) ]

        # get the scalar factor associated with this assignment of stitch operators
        if state_prep_basis == "ZXY":
            scalar_factor = (-1)**np.sum( op == "I" for op in assignment )
        else: # state_prep_basis == "SIC"
            scalar_factor = np.product([ -1 if op == "I" else 3/2 for op in assignment ])

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
