#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from tensorflow_extension import tf_outer_product

from itertools import product as set_product
from functools import reduce
from copy import deepcopy

from fragment_distributions import SIC, IZXY, ZZXY, op_basis, \
    state_vecs_SIC, state_vecs_ZXY, state_vecs, \
    FragmentProbabilities, FragmentAmplitudes

##########################################################################################
# this script contains methods to
# (i) simulate circuit fragments to collect conditional distributions,
# (ii) combine conditional distributions of fragments into circuit distributions, and
# (iii) sample circuit measurements from the conditional distributions of its fragments
# NOTE: (iii) is not yet implemented
##########################################################################################

# return a gate that rotates |0> into a state pointing along the given vector
def prep_gate(state):
    vec = state_vecs[state]
    theta = np.arccos(vec[0]) # angle down from north pole, i.e. from |0>
    phi = np.arctan2(vec[2],vec[1]) # angle in the X-Y plane
    return qs.extensions.standard.U3Gate(theta, phi, 0)

# gates to measure in different bases
basis_gates = { basis : prep_gate(f"+{basis}").inverse()
                for basis in [ "Z", "X", "Y" ] }

def sort_init_exit_wires(frag_stitches, num_fragments):
    '''
    identify initialization and exit wires for each fragment

    accepts:
      (i) a dictionary of stitches
      (ii) a total number of fragments

    returns:
      (i) a list of lists of "init_wires" (one list of wires for each fragment)
      (ii) a list of lists of "exit_wires" (one list of wires for each fragment)
    '''

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

##########################################################################################
# methods to simulate individual circuits
##########################################################################################

# get the output quantum state of a circuit
def get_circuit_amplitudes(circuit, **kwargs):
    simulator = qs.Aer.get_backend("statevector_simulator")
    result = qs.execute(circuit, simulator, **kwargs).result()
    state_vector = result.get_statevector(circuit)
    qubits = len(state_vector).bit_length()-1
    return tf.constant(state_vector, shape = (2,)*qubits)

# get a distribution over measurement outcomes for a circuit
def get_circuit_distribution(circuit, backend_simulator = "statevector_simulator",
                             dtype = tf.float64, **kwargs):
    if backend_simulator == "statevector_simulator":
        amplitudes = get_circuit_amplitudes(circuit, **kwargs)
        return tf.cast(abs(amplitudes)**2, dtype = dtype)

    simulator = qs.Aer.get_backend(backend_simulator)

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

##########################################################################################
# fragment simulation methods
##########################################################################################

# get amplitudes for a single fragment
def get_single_fragment_amplitudes(fragment, init_wires = None, exit_wires = None,
                                   **kwargs):
    '''
    this method accepts:
      (i) a circuit fragment,
      (ii) a list of "input" wires (`init_wires`),
      (iii) a list of "output" wires (`exit_wires`),
      (iv) additional keyword arguments passed to the Qiskit statevector simulator.

    this method returns "conditional amplitudes" of the fragment, which are
      (i) conditional on computational basis states prepared at `init_wires`,
      (ii) projected onto computational basis states at `exit_wires`.
    '''

    # initialize an empty conditional distribution over measurement outcomes
    frag_amps = FragmentAmplitudes()

    # identify computational basis states
    def _state_ZXY(bit): return "+Z" if not bit else "-Z"
    def _state_vec(bit):
        vec = (1,0) if not bit else (0,1)
        return np.array(vec, dtype = complex)

    # identify the axes we will project out for the exit wires
    # and sort exit wires according to these axes
    exit_axes = { wire : -fragment.qubits.index(wire)-1 for wire in exit_wires }
    exit_wires = sorted(exit_wires, key = lambda wire : exit_axes[wire] )

    # for every choice of states prepared on all on init wires
    for init_states in set_product(range(2), repeat = len(init_wires)):
        init_conds = frozenset(zip(init_states, init_wires))

        # build circuit with the "input" states prepared appropriately
        prep_circuits = [ act_gates(fragment, prep_gate(_state_ZXY(bit_state)), wire)
                          for bit_state, wire in init_conds ]
        init_circuit = reduce(lambda x, y : x + y, prep_circuits + [ fragment ])

        # get the quantum state vector for the circuit
        all_amplitudes = get_circuit_amplitudes(init_circuit, *kwargs)

        # for every set of projections on all on exit wires
        for exit_states in set_product(range(2), repeat = len(exit_wires)):
            exit_conds = zip(exit_states, exit_wires)

            # project onto the given measured states on exit wires
            projected_amplitudes = deepcopy(all_amplitudes)
            for exit_state, exit_wire in zip(exit_states, exit_wires):
                exit_vec = tf.constant(_state_vec(exit_state))
                axes = [ [ 0 ], [ exit_axes[exit_wire] ] ]
                projected_amplitudes \
                    = tf.tensordot(exit_vec, projected_amplitudes, axes = axes)

            frag_amps.add(init_conds, exit_conds, projected_amplitudes)

    return frag_amps

# get probability distributions for a single fragment
def get_single_fragment_probabilities(fragment, init_wires = None, exit_wires = None,
                                      backend_simulator = "statevector_simulator",
                                      init_op_basis = SIC, dtype = tf.float64, **kwargs):
    assert( init_op_basis in [ SIC, IZXY ] ) # todo(?): add support to ZZXY
    '''
    this method accepts:
      (i) a circuit fragment,
      (ii) a list of "input" wires (`init_wires`),
      (iii) a list of "output" wires (`exit_wires`),
      (iv) additional keyword arguments passed to the Qiskit statevector simulator.

    this method returns probability distributions for non-exit-wires of the fragment
    '''

    if backend_simulator == "statevector_simulator":
        frag_amps = get_single_fragment_amplitudes(fragment, init_wires, exit_wires)
        return frag_amps.to_probabilities(dtype = dtype)

    def _dist_terms_IZXY(state):
        if state[0] == "-": return [ "I" ]
        else: return [ state, "I" ]

    # initialize an empty conditional distribution over measurement outcomes
    frag_dist = FragmentProbabilities(init_op_basis)

    # identify the axes we will project out for the exit wires
    # and sort exit wires according to these axes
    exit_axes = { wire : -fragment.qubits.index(wire)-1 for wire in exit_wires }
    exit_wires = sorted(exit_wires, key = lambda wire : exit_axes[wire] )

    if init_op_basis == SIC:
        init_state_basis = list(state_vecs_SIC.keys())
    else:
        init_state_basis = list(state_vecs_ZXY.keys())

    # for every choice of states prepared on all on init wires
    for init_states in set_product(init_state_basis, repeat = len(init_wires)):
        init_shots = kwargs["shots"]

        if init_op_basis == SIC:
            init_dist = frag_dist
        else:
            init_dist = FragmentProbabilities(init_op_basis)
            init_shots /= 3**sum( state[0] == "-" for state in init_states )

        # build circuit with the "input" states prepared appropriately
        init_conds = frozenset(zip(init_states, init_wires))
        prep_circuits = [ act_gates(fragment, prep_gate(state), wire)
                          for state, wire in init_conds ]
        init_circuit = reduce(lambda x, y : x + y, prep_circuits + [ fragment ])

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
                for bit_state, wire in zip(exit_bits, exit_wires):
                    qubits = len(projected_dist.shape)
                    begin = [ 0 ] * qubits
                    size = [ 2 ] * qubits
                    begin[exit_axes[wire]] = bit_state
                    size[exit_axes[wire]] = 1
                    projected_dist = tf.sparse.slice(projected_dist, begin, size)
                    projected_dist = tf.sparse.reshape(projected_dist, (2,)*(qubits-1))

                # loop over all assignments of + Z/X/Y and I at exit wires
                exit_terms = [ _dist_terms_IZXY(state) for state in exit_states ]
                for exit_states in set_product(*exit_terms):
                    # divide distribution by 3 for each identity operator (I)
                    #   to average over measurements in 3 different bases
                    iden_fac = 3**sum( state == "I" for state in exit_states )
                    exit_conds = zip(exit_states, exit_wires)
                    init_dist.add(init_conds, exit_conds, projected_dist / iden_fac)

        ### end construction of init_dist

        if init_dist is frag_dist: continue

        for conditions, dist in init_dist:
            init_conds = set( cond for cond in conditions if cond[0] )
            exit_conds = set( cond for cond in conditions if not cond[0] )
            if not init_conds:
                frag_dist.add(conditions, dist)
                continue

            init_terms = [ _dist_terms_IZXY(state)
                           for state in init_states ]
            for init_ops in set_product(*init_terms):
                iden_fac = 3**sum( op == "I" for op in init_ops )
                init_conds = zip(init_ops, init_wires)
                frag_dist.add(init_conds, exit_conds, dist / iden_fac)

    return frag_dist

# get amplitudes for a list of fragments
def get_fragment_amplitudes(fragments, frag_stitches, **kwargs):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = sort_init_exit_wires(frag_stitches, len(fragments))

    # compute conditional distributions over measurement outcomes for all fragments
    return [ get_single_fragment_amplitudes(ff, ii, ee, **kwargs)
             for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

# get probabilities for a list of fragments
def get_fragment_probabilities(fragments, frag_stitches,
                               backend_simulator = "statevector_simulator",
                               init_op_basis = SIC, dtype = tf.float64, **kwargs):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = sort_init_exit_wires(frag_stitches, len(fragments))

    def _get_frag_probs(ff, ii, ee):
        return get_single_fragment_probabilities(ff, ii, ee, backend_simulator,
                                                 init_op_basis, dtype, **kwargs)

    # compute conditional distributions over measurement outcomes for all fragments
    return [ _get_frag_probs(ff, ii, ee)
             for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

##########################################################################################
# methods to combine conditional fragment distributions
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
def rearrange_wires(distribution, old_wire_order, new_wire_order, wire_map = None):
    if wire_map is None: wire_map = { wire : wire for wire in old_wire_order }
    current_wire_order = [ wire_map[wire] for wire in old_wire_order ]
    wire_permutation = [ current_wire_order.index(wire) for wire in new_wire_order ]
    axis_permutation = [ len(new_wire_order) - 1 - idx for idx in wire_permutation ][::-1]
    if type(distribution) is not tf.SparseTensor:
        return tf.transpose(distribution, axis_permutation)
    else:
        return tf.sparse.transpose(distribution, axis_permutation)

# sort the qubits of a reconstructed probability distrubion
def sort_reconstructed_distribution(distribution, frag_stitches, frag_wiring,
                                    frag_qubits, circuit_wires):

    # identify the order of wires in the reconstructed probability distribution
    combined_wire_order = [ ( frag_idx, qubit )
                            for frag_idx, qubits in enumerate(frag_qubits)
                            for qubit in qubits
                            if ( frag_idx, qubit ) not in frag_stitches.keys() ]

    # determine the map from fragment wires to full-circuit wires
    wire_map = frag_wire_map(circuit_wires, frag_wiring, frag_stitches)
    return rearrange_wires(distribution, combined_wire_order, circuit_wires, wire_map)

# combine conditional probability distributions of fragments
def combine_fragment_probabilities(frag_probs, frag_stitches, frag_wiring = None,
                                   frag_qubits = None, circuit_wires = None,
                                   stitch_basis = SIC, dtype = tf.float64, updates = False):

    assert(( frag_wiring and frag_qubits and circuit_wires ) or
           ( not frag_wiring and not frag_qubits and not circuit_wires ))

    # determine which operators to assign to stitches
    if stitch_basis == SIC:
        stitch_ops = list(state_vecs_SIC.keys()) + [ "I" ]
    else:
        stitch_ops = list(state_vecs_ZXY.keys()) + [ "I" ]

    # determine the type and shape of the combined probability distributon
    combined_shape = ()
    for frag_prob in frag_probs:
        for _, dist in frag_prob:
            dist_type = type(dist)
            combined_shape += tuple(dist.shape)
            break

    # initialize an empty probability distribution
    if dist_type is tf.SparseTensor:
        indices = np.empty((0,len(combined_shape)))
        values = tf.constant([], dtype = dtype)
        combined_dist = tf.SparseTensor(indices, values, combined_shape)
    else:
        combined_dist = tf.zeros(combined_shape, dtype = dtype)

    # loop over all assigments of stitch operators at all cut locations
    for op_assignment in set_product(stitch_ops, repeat = len(frag_stitches)):
        if updates: print(op_assignment)

        # collect the assignments of exit/init outcomes/states for each fragment
        frag_exit_conds = [ set() for _ in range(len(frag_probs)) ]
        frag_init_conds = [ set() for _ in range(len(frag_probs)) ]
        for stitch_idx, ( exit_frag_wire, init_frag_wire ) in enumerate(frag_stitches.items()):
            exit_frag_idx, exit_wire = exit_frag_wire
            init_frag_idx, init_wire = init_frag_wire
            frag_exit_conds[exit_frag_idx].add(( op_assignment[stitch_idx], exit_wire ))
            frag_init_conds[init_frag_idx].add(( op_assignment[stitch_idx], init_wire ))

        dist_factors = [ frag_dist[init_conds, exit_conds]
                         for frag_dist, init_conds, exit_conds
                         in zip(frag_probs, frag_init_conds, frag_exit_conds) ]

        # get the scalar factor associated with this assignment of stitch operators
        if stitch_basis == SIC:
            scalar_factor = np.product([ -1 if op == "I" else 3/2 for op in op_assignment ])
        else:
            scalar_factor = (-1)**np.sum( op == "I" for op in op_assignment )

        # add to the combined probability distribution over measurement outcomes
        combined_dist += scalar_factor * reduce(tf_outer_product, dist_factors[::-1])

    # if we did not provide wiring info, simply return the combined probability distribution
    if frag_wiring is None:
        return combined_dist

    # otherwise, sort qubits appropriately
    else:
        return sort_reconstructed_distribution(combined_dist, frag_stitches,
                                               frag_wiring, frag_qubits, circuit_wires)
