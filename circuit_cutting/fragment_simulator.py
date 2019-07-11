#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from tensorflow_extension import *

from itertools import product as set_product
from functools import reduce
from copy import deepcopy

##########################################################################################
# this script contains methods to simulate circuit fragments
# and combine fragment simulation results
##########################################################################################

# choose whether to prepare states in the SIC or ZXY basis
SIC, ZXY = "SIC", "ZXY" # define these to protect against typos
state_prep_basis = SIC
stitching_basis = SIC

# ZXY basis to use for statevector simulations
basis_ZXY_statevector = [ "-Z", "+Z", "+X", "+Y" ]

# define state vectors for SIC basis
state_vecs_SIC = [ ( +1, +1, +1 ), ( +1, -1, -1 ), ( -1, +1, -1 ), ( -1, -1, +1 ) ]
state_vecs_SIC = { f"SIC-{jj}" : tuple( np.array(vec) / np.linalg.norm(vec) )
                   for jj, vec in enumerate(state_vecs_SIC) }

# define state vectors for ZXY basis
state_vecs_ZXY = { "+Z" : (+1,0,0),
                   "-Z" : (-1,0,0),
                   "+X" : (0,+1,0),
                   "-X" : (0,-1,0),
                   "+Y" : (0,0,+1),
                   "-Y" : (0,0,-1) }

state_vecs = dict(state_vecs_ZXY, **state_vecs_SIC)

# operators to insert on either end of a stitch
if stitching_basis == SIC:
    stitch_ops = list(state_vecs_SIC.keys()) + [ "I" ]
else: # stitching_basis == ZXY
    stitch_ops = list(state_vecs_ZXY.keys()) + [ "I" ]

# return a gate that rotates |0> into a state pointing along the given vector
def prep_gate(state):
    vec = state_vecs[state]
    theta = np.arccos(vec[0]) # angle down from north pole, i.e. from |0>
    phi = np.arctan2(vec[2],vec[1]) # angle in the X-Y plane
    return qs.extensions.standard.U3Gate(theta, phi, 0)

# gates to measure in different bases
basis_gates = { basis : prep_gate(f"+{basis}").inverse()
                for basis in [ "Z", "X", "Y" ] }

# class for a conditional distribution function
# TODO: add flexibility for different init/exit bases
#       nominal plan: explicitly allow for [I,+Z,+X,+Y], [-Z,+Z,+X,+Y] and [SIC-n] bases
#       optional: allow for *any* informationally-complete basis
#       we will probably spin off the conditional_distribution into a separate file
class conditional_distribution:

    # initialize an empty conditional distribution function
    def __init__(self, empty_condition_dist = None):
        self.dist_dict = {}
        if empty_condition_dist is not None:
            self.dist_dict[frozenset(), frozenset()] = empty_condition_dist

    # if asked to print, print the dictionary
    def __repr__(self):
        return self.dist_dict.__repr__()

    # provide an iterator over conditions / distribution functions
    def items(self):
        return self.dist_dict.items()

    # add data to the conditional distribution object
    def add(self, init_keys, exit_keys, dist):
        key = ( frozenset(init_keys), frozenset(exit_keys) )
        try:
            self.dist_dict[key] += dist
        except:
            self.dist_dict[key] = deepcopy(dist)

    # retrieve a distribution function with given conditions
    def __getitem__(self, conditions):
        state_preps, measure_ops = frozenset(conditions[0]), frozenset(conditions[1])

        # return a distribution function if we have it
        try: return self.dist_dict[state_preps, measure_ops]
        except: None

        # use available data to compute the distribution function for the given conditions

        for wire, op in state_preps:
            vacancy = state_preps.difference({(wire,op)})
            dist = lambda op : self[vacancy.union({(wire,op)}), measure_ops]

            if state_prep_basis == SIC:

                if op == "I":
                    return dist((0,0,0)) * 2 # I = 2 * ( maximally mixed state )

                if op in state_vecs_ZXY.keys():
                    return dist(state_vecs_ZXY[op])

                if type(op) is tuple:
                    assert( len(op) == 3 )
                    return 1/4 * sum( ( 1 + 3 * np.dot(op, vec) ) * dist(idx)
                                      for idx, vec in state_vecs_SIC.items() )

            else: # state_prep_basis == ZXY

                if op == "-Z": return dist("I") - dist("+Z")
                if op == "-X": return dist("I") - dist("+X")
                if op == "-Y": return dist("I") - dist("+Y")

                if op in state_vecs_SIC.keys():
                    return dist(state_vecs_SIC[op])

                if type(op) is tuple:
                    assert( len(op) == 3 )
                    bases = [ "+Z", "+X", "+Y" ]
                    init_keys, exit_keys = zip(*self.dist_dict.keys())
                    dist_ZXY = sum( val * dist(basis) for val, basis in zip(op, bases) )
                    return dist_ZXY + dist("I") * ( 1 - sum(op) ) / 2

        for wire, op in measure_ops:
            vacancy = measure_ops.difference({(wire,op)})
            dist = lambda op : self[state_preps, vacancy.union({(wire,op)})]

            if op == "-Z": return dist("I") - dist("+Z")
            if op == "-X": return dist("I") - dist("+X")
            if op == "-Y": return dist("I") - dist("+Y")

            if op in state_vecs_SIC.keys():
                return dist(state_vecs_SIC[op])

            if type(op) is tuple:
                assert( len(op) == 3 )
                bases = [ "+Z", "+X", "+Y" ]
                dist_ZXY = sum( val * dist(basis) for val, basis in zip(op, bases ) )
                return dist_ZXY + dist("I") * ( 1 - sum(op) ) / 2

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
def get_circuit_distribution(circuit, backend_simulator = "statevector_simulator",
                             return_amplitudes = False, dtype = tf.float64, **kwargs):
    if return_amplitudes: assert( backend_simulator == "statevector_simulator" )

    simulator = qs.Aer.get_backend(backend_simulator)

    if backend_simulator == "statevector_simulator":
        result = qs.execute(circuit, simulator).result()
        state_vector = result.get_statevector(circuit)
        qubits = len(state_vector).bit_length()-1
        amplitudes = tf.constant(state_vector, shape = (2,)*qubits)
        if return_amplitudes: return amplitudes
        else: return abs(amplitudes)**2

    if backend_simulator == "qasm_simulator":
        # identify current registers in the circuit
        qubit_registers = [ wire[0] for wire in circuit.qubits if wire[1] == 0 ]
        clbit_registers = [ wire[0] for wire in circuit.clbits if wire[1] == 0 ]
        all_registers = qubit_registers + clbit_registers

        # add measurements for all quantum registers
        name_prefix = "_".join( register.prefix for register in all_registers )
        for rr, register in enumerate(qubit_registers):
            bits = len(register)
            name = name_prefix + f"_c{rr}"
            measurement_register = qs.ClassicalRegister(bits, name)
            circuit.add_register(measurement_register)
            circuit.measure(register, measurement_register)

        # simulate!
        result = qs.execute(circuit, simulator, **kwargs).result()
        state_counts = result.get_counts(circuit)

        # collect results into a sparse tensor
        indices = [ tuple( int(bit) for bit in state ) for state in state_counts.keys() ]
        values = list(state_counts.values())
        dense_shape = (2,)*len(circuit.clbits)
        sparse_counts = tf.SparseTensor(indices, values, dense_shape)
        return tf.cast(tf.sparse.reorder(sparse_counts), dtype) / kwargs["shots"]

    else:
        print("backend not supported:", backend_simulator)
        return None

def dist_terms_ZXY_statevector(state):
    if state == "+Z": return [ "+Z", "I" ]
    if state == "-Z": return [ "I" ]
    else: return [ state ]

def dist_terms_ZXY_shots(state):
    if state[0] == "-": return [ "I" ]
    else: return [ state, "I" ]

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
def get_fragment_distribution(fragment, init_wires = None, exit_wires = None,
                              backend_simulator = "statevector_simulator",
                              dtype = tf.float64, **kwargs):
    # decide what states to prepare on each individual init wire
    if state_prep_basis == SIC:
        qubit_init_states = state_vecs_SIC.keys()
    else: # state_prep_basis == ZXY
        if backend_simulator != "statevector_simulator":
            qubit_init_states = state_vecs_ZXY.keys()
        else:
            qubit_init_states = basis_ZXY_statevector

    # identify the axes we will project out for the exit wires
    exit_axes = { wire : -fragment.qubits.index(wire)-1 for wire in exit_wires }

    # sort exit wires in reverse order from their indices
    exit_wires = sorted(exit_wires, key = lambda wire : exit_axes[wire] )

    # initialize an empty conditional distribution over measurement outcomes
    frag_dist = conditional_distribution()

    # for every choice of states prepared on all on init wires
    for init_states in set_product(qubit_init_states, repeat = len(init_wires)):
        if state_prep_basis == SIC:
            init_dist = frag_dist
        else:
            init_dist = conditional_distribution()

        if backend_simulator != "statevector_simulator":
            init_shots = kwargs["shots"]
            if state_prep_basis != SIC:
                init_shots /= 3**sum( state[0] == "-" for state in init_states )

        # build circuit with the "input" states prepared appropriately
        init_keys = frozenset(zip(init_wires, init_states))
        prep_circuits = [ act_gates(fragment, prep_gate(state), wire)
                          for wire, state in init_keys ]
        init_circuit = reduce(lambda x, y : x + y, prep_circuits + [ fragment ])

        if backend_simulator == "statevector_simulator":

            # get the state vector for the circuit (i.e. with amplitudes)
            all_amplitudes \
                = get_circuit_distribution(init_circuit, backend_simulator,
                                           return_amplitudes = True, dtype = dtype)

            # for every set of measurement outcomes on all on exit wires
            for exit_states in set_product(basis_ZXY_statevector,
                                           repeat = len(exit_wires)):

                # project onto the given measured states on exit wires
                projected_amplitudes = deepcopy(all_amplitudes)
                for exit_wire, exit_state in zip(exit_wires, exit_states):
                    exit_vec = prep_gate(exit_state).to_matrix()[:,0].conj()
                    axes = [ [ 0 ], [ exit_axes[exit_wire] ] ]
                    projected_amplitudes \
                        = tf.tensordot(exit_vec, projected_amplitudes, axes = axes)

                # probability distribution on non-exit output wires
                dist = abs(projected_amplitudes)**2

                exit_terms = [ dist_terms_ZXY_statevector(state)
                               for state in exit_states ]
                for exit_ops in set_product(*exit_terms):
                    exit_keys = zip(exit_wires, exit_ops)
                    init_dist.add(init_keys, exit_keys, dist)

        else: # backend_simulator != "statevector_simulator"

            # for every choice of measurement bases on all on exit wires
            for exit_bases in set_product(basis_gates.keys(), repeat = len(exit_wires)):

                # if we are simulating with initial states polarized in - Z/X/Y
                # then we are actually collecting data for an insertion of I,
                # so we only need a third of the number of shots (per such state)
                total_shots \
                    = init_shots / 3**sum( state[0] == "-" for state in init_states )
                if total_shots < 1: continue

                # build a circuit to measure in the correct bases
                measurement_circuit = [ act_gates(fragment, basis_gates[basis], wire)
                                        for wire, basis in zip(exit_wires, exit_bases) ]
                circuit = reduce(lambda x, y : x + y,
                                 [ init_circuit ] + measurement_circuit)

                # get probability distribution over measurement outcomes
                full_dist = get_circuit_distribution(circuit, backend_simulator,
                                                     dtype = dtype, shots = int(total_shots))

                # project onto given exit-wire measurement outcomes
                for exit_bits in set_product(range(2), repeat = len(exit_wires)):
                    exit_states = [ ( "+" if bit == 0 else "-" ) + basis
                                    for bit, basis in zip(exit_bits, exit_bases) ]

                    projected_dist = deepcopy(full_dist)
                    for wire, bit_state in zip(exit_wires, exit_bits):
                        qubits = len(projected_dist.shape)
                        begin = [ 0 ] * qubits
                        size = [ 2 ] * qubits
                        begin[exit_axes[wire]] = bit_state
                        size[exit_axes[wire]] = 1
                        projected_dist = tf.sparse.slice(projected_dist, begin, size)
                        projected_dist = tf.sparse.reshape(projected_dist, (2,)*(qubits-1))

                    # loop over all assignments of + Z/X/Y and I at exit wires
                    exit_terms = [ dist_terms_ZXY_shots(state) for state in exit_states ]
                    for exit_states in set_product(*exit_terms):
                        # divide distribution by 3 for each identity operator (I)
                        #   to average over measurements in 3 different bases
                        iden_fac = 3**sum( state == "I" for state in exit_states )
                        exit_keys = zip(exit_wires, exit_states)
                        init_dist.add(init_keys, exit_keys, projected_dist / iden_fac)

            ### end construction of init_dist

        if init_dist is frag_dist: continue

        for ( init_keys, exit_keys ), dist in init_dist.items():
            if init_keys == frozenset():
                frag_dist.add(init_keys, exit_keys, dist)
                continue

            if backend_simulator == "statevector_simulator":
                init_terms = [ dist_terms_ZXY_statevector(state)
                               for state in init_states ]
                for init_ops in set_product(*init_terms):
                    init_keys = zip(init_wires, init_ops)
                    frag_dist.add(init_keys, exit_keys, dist)

            else: # backend_simulator != "statevector_simulator"

                init_terms = [ dist_terms_ZXY_shots(state)
                               for state in init_states ]
                for init_ops in set_product(*init_terms):
                    iden_fac = 3**sum( op == "I" for op in init_ops )
                    init_keys = zip(init_wires, init_ops)
                    frag_dist.add(init_keys, exit_keys, dist / iden_fac)

    return frag_dist

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
def get_fragment_distributions(fragments, frag_stitches,
                               backend_simulator = "statevector_simulator",
                               dtype = tf.float64, **kwargs):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = sort_init_exit_wires(frag_stitches, len(fragments))

    # compute conditional distributions over measurement outcomes for all fragments
    return [ get_fragment_distribution(ff, ii, ee, backend_simulator,
                                       dtype = dtype, **kwargs)
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
    if type(distribution) is not tf.SparseTensor:
        return tf.transpose(distribution, axis_permutation)
    else:
        return tf.sparse.transpose(distribution, axis_permutation)

# simulate fragments and stitch together results
# accepts a list of circuit fragments and a dictionary for how to stitch them together
# return a distribution over measurement outcomes on the stitched-together circuit
def simulate_and_combine(fragments, frag_stitches,
                         frag_wiring = None, wire_order = None,
                         backend_simulator = "statevector_simulator",
                         dtype = tf.float64, **kwargs):
    # get conditional probability distributions for of all circuit fragments
    frag_dists = get_fragment_distributions(fragments, frag_stitches,
                                            backend_simulator, dtype, **kwargs)

    # identify the order of wires in a combined/reconstructed distribution over outcomes
    combined_wire_order = [ ( frag_idx, qubit )
                            for frag_idx, fragment in enumerate(fragments)
                            for qubit in fragment.qubits
                            if ( frag_idx, qubit ) not in frag_stitches.keys() ]

    # initialize an empty combined distribution over outcomes
    dist_shape = (2,)*len(combined_wire_order)
    if backend_simulator == "statevector_simulator":
        combined_dist = tf.zeros(dist_shape, dtype = dtype)
    else:
        indices = np.empty((0,len(dist_shape)))
        values = tf.constant([], dtype = dtype)
        combined_dist = tf.SparseTensor(indices, values, dist_shape)

    # loop over all assigments of stitch operators at all cut locations
    for op_assignment in set_product(stitch_ops, repeat = len(frag_stitches)):

        # collect the assignments of exit/init outcomes/states for each fragment
        frag_exit_keys = [ set() for _ in range(len(fragments)) ]
        frag_init_keys = [ set() for _ in range(len(fragments)) ]
        for stitch_idx, ( exit_frag_wire, init_frag_wire ) in enumerate(frag_stitches.items()):
            exit_frag_idx, exit_wire = exit_frag_wire
            init_frag_idx, init_wire = init_frag_wire
            frag_exit_keys[exit_frag_idx].add(( exit_wire, op_assignment[stitch_idx] ))
            frag_init_keys[init_frag_idx].add(( init_wire, op_assignment[stitch_idx] ))

        # get the conditional probability distribution at each fragment
        dist_factors = [ frag_dist[init_keys, exit_keys]
                         for frag_dist, init_keys, exit_keys
                         in zip(frag_dists, frag_init_keys, frag_exit_keys) ]

        # get the scalar factor associated with this assignment of stitch operators
        if stitching_basis == SIC:
            scalar_factor = np.product([ -1 if op == "I" else 3/2 for op in op_assignment ])
        else: # stitching_basis == ZXY
            scalar_factor = (-1)**np.sum( op == "I" for op in op_assignment )

        # add to the combined distribution over measurement outcomes
        combined_dist += scalar_factor * reduce(tf_outer_product, dist_factors[::-1])

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
