#!/usr/bin/env python3

##########################################################################################
# methods simulate circuit fragments and collect conditional distributions
##########################################################################################

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from itertools import product as set_product
from functools import reduce
from copy import deepcopy

from fragment_distributions import SIC, IZXY, pauli, \
    state_vecs_SIC, state_vecs_ZXY, state_vecs, \
    FragmentAmplitudes, FragmentProbabilities

# read a wire path map to identify init / exit wires for all fragments
def identify_init_exit_wires(wire_path_map, num_fragments):
    # collect all exit/init wires in format ( frag_index, wire )
    all_init_wires = set()
    all_exit_wires = set()

    # loop over all paths to identify init/exit wires
    for path in wire_path_map.values():
        all_init_wires.update(path[1:])
        all_exit_wires.update(path[:-1])

    # collect init/exit wires within each fragment, sorting them to fix their order
    init_wires = tuple([ { wire for idx, wire in all_init_wires if idx == frag_idx }
                         for frag_idx in range(num_fragments) ])
    exit_wires = tuple([ { wire for idx, wire in all_exit_wires if idx == frag_idx }
                         for frag_idx in range(num_fragments) ])
    return init_wires, exit_wires

# return a gate that rotates |0> into a state pointing along the given vector
def _prep_gate(state):
    vec = state_vecs[state]
    theta = np.arccos(vec[0]) # angle down from north pole, i.e. from |0>
    phi = np.arctan2(vec[2],vec[1]) # angle in the X-Y plane
    return qs.extensions.standard.U3Gate(theta, phi, 0)

# gates to measure in different bases
_basis_gates = { basis : _prep_gate(f"+{basis}").inverse()
                 for basis in [ "Z", "X", "Y" ] }

# return an empty circuit that acts with the given gate on the given qubits
def _act_gate(circuit, gate, *wires):
    registers = set( wire[0] for wire in circuit.qubits + circuit.clbits )
    new_circuit = qs.QuantumCircuit(*registers)
    new_circuit.append(gate, qargs = list(wires))
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
def get_circuit_probabilities(circuit, backend_simulator = "statevector_simulator",
                              dtype = tf.float64, max_shots = 8192, **kwargs):
    if backend_simulator == "statevector_simulator":
        amplitudes = get_circuit_amplitudes(circuit, **kwargs)
        return tf.cast(abs(amplitudes)**2, dtype = dtype)

    simulator = qs.Aer.get_backend(backend_simulator)

    if backend_simulator == "qasm_simulator":
        # identify current registers in the circuit
        qubit_registers = [ register for register, bit in circuit.qubits if bit == 0 ]
        clbit_registers = [ register for register, bit in circuit.clbits if bit == 0 ]
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
        # In the case where we want to more shots than max_shots,
        #   we must iteratively build our counts
        total_shots = kwargs["shots"]
        del kwargs["shots"]

        state_counts = {}
        shots_remaining = total_shots
        while shots_remaining > 0:
            # fire as many shots as we can, and update the nubmer of shots remaining
            shots = min(max_shots, shots_remaining)
            result = qs.execute(circuit, simulator, shots = shots, **kwargs).result()
            shots_remaining -= shots

            # Tally results in the state_counts dictionary
            for state, counts in result.get_counts(circuit).items():
                try:
                    state_counts[state] += counts # add to state_counts if we can
                except KeyError: # if `state` was not already in `state_counts`
                    state_counts[state] = counts

        # collect results into a sparse tensor
        indices = [ tuple( int(bit) for bit in state )
                    for state in state_counts.keys() ]
        values = list(state_counts.values())
        dense_shape = (2,)*len(circuit.clbits)
        sparse_counts = tf.SparseTensor(indices, values, dense_shape)
        return tf.cast(tf.sparse.reorder(sparse_counts), dtype) / total_shots

    else:
        print("backend not supported:", backend_simulator)
        return None

##########################################################################################
# methods to simulate fragents and charactize the corresponding conditional distributions
##########################################################################################

# get amplitudes for a single fragment
def _get_single_fragment_amplitudes(fragment, init_wires = None, exit_wires = None,
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

    # sort init wires to fix their order
    init_wires = sorted(init_wires, key = lambda wire : fragment.qubits.index(wire))

    # for every choice of states prepared on all on init wires
    for init_states in set_product(range(2), repeat = len(init_wires)):
        init_conds = frozenset(zip(init_wires, init_states))

        # build circuit with the "input" states prepared appropriately
        prep_circuits = [ _act_gate(fragment, _prep_gate(_state_ZXY(bit_state)), wire)
                          for wire, bit_state in init_conds ]
        init_circuit = reduce(lambda x, y : x + y, prep_circuits + [ fragment ])

        # get the quantum state vector for the circuit
        all_amplitudes = get_circuit_amplitudes(init_circuit, **kwargs)

        # for every set of projections on all on exit wires
        for exit_states in set_product(range(2), repeat = len(exit_wires)):
            exit_conds = zip(exit_wires, exit_states)

            # project onto the given measured states on exit wires
            projected_amplitudes = deepcopy(all_amplitudes)
            for exit_wire, exit_state in zip(exit_wires, exit_states):
                exit_vec = tf.constant(_state_vec(exit_state))
                axes = [ [ 0 ], [ exit_axes[exit_wire] ] ]
                projected_amplitudes \
                    = tf.tensordot(exit_vec, projected_amplitudes, axes = axes)

            frag_amps.add(init_conds, exit_conds, projected_amplitudes)

    return frag_amps

# get probability distributions for a single fragment
def _get_single_fragment_probabilities(fragment, init_wires = None, exit_wires = None,
                                       backend_simulator = "statevector_simulator",
                                       init_basis = pauli, exit_basis = pauli,
                                       dtype = tf.float64, **kwargs):
    # save the bases in which we want to store init/exit conditions
    final_init_basis = init_basis
    final_exit_basis = exit_basis

    # we can only actually *compute* distributions in certain bases,
    # so if we weren't asked for one of those, choose one of them to use for now
    if init_basis not in [ SIC, IZXY ]:
        init_basis = SIC
    if exit_basis != IZXY:
        exit_basis = IZXY

    '''
    this method accepts:
      (i) a circuit fragment,
      (ii) a list of "input" wires (`init_wires`),
      (iii) a list of "output" wires (`exit_wires`),
      (iv) additional keyword arguments passed to the Qiskit statevector simulator.

    this method returns probability distributions for non-exit-wires of the fragment
    '''

    if backend_simulator == "statevector_simulator":
        frag_amps = _get_single_fragment_amplitudes(fragment, init_wires, exit_wires)
        return frag_amps.to_probabilities(dtype = dtype)

    def _dist_terms_IZXY(state):
        if state[0] == "-": return [ "I" ]
        else: return [ state, "I" ]

    # initialize an empty conditional distribution over measurement outcomes
    frag_dist = FragmentProbabilities(init_basis = init_basis)

    # identify the axes we will project out for the exit wires
    # and sort exit wires according to these axes
    exit_axes = { wire : -fragment.qubits.index(wire)-1 for wire in exit_wires }
    exit_wires = sorted(exit_wires, key = lambda wire : exit_axes[wire] )

    # sort init wires to fix their order
    init_wires = sorted(init_wires, key = lambda wire : fragment.qubits.index(wire))

    if init_basis == SIC:
        init_state_basis = list(state_vecs_SIC.keys())
    else: # init_basis == IZXY
        init_state_basis = list(state_vecs_ZXY.keys())

    # remember the number of shots we were told to run
    shots = kwargs["shots"]
    del kwargs["shots"]

    # for every choice of states prepared on all on init wires
    for init_states in set_product(init_state_basis, repeat = len(init_wires)):
        init_shots = shots # number of shots for this set of init states

        if init_basis == SIC:
            init_dist = frag_dist
        else: # init_basis == IZXY
            init_dist = FragmentProbabilities(init_basis)

            # if we are simulating with initial states polarized in - Z/X/Y
            # then we are actually collecting data for an insertion of I,
            # so we only need a third of the number of shots (per such state)
            init_shots /= 3**sum( state[0] == "-" for state in init_states )
            if init_shots < 1 : continue
            init_shots = int(init_shots + 0.5)

        # build circuit with the "input" states prepared appropriately
        init_conds = frozenset(zip(init_wires, init_states))
        prep_circuits = [ _act_gate(fragment, _prep_gate(state), wire)
                          for wire, state in init_conds ]
        init_circuit = reduce(lambda x, y : x + y, prep_circuits + [ fragment ])

        # for every choice of measurement bases on all on exit wires
        for exit_bases in set_product(_basis_gates.keys(), repeat = len(exit_wires)):

            # build a circuit to measure in the correct bases
            measurement_circuit = [ _act_gate(fragment, _basis_gates[basis], wire)
                                    for wire, basis in zip(exit_wires, exit_bases) ]
            circuit = reduce(lambda x, y : x + y,
                             [ init_circuit ] + measurement_circuit)

            # get probability distribution over measurement outcomes
            full_dist = get_circuit_probabilities(circuit, backend_simulator,
                                                  dtype = dtype, shots = init_shots,
                                                  **kwargs)

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
                exit_terms = [ _dist_terms_IZXY(state) for state in exit_states ]
                for exit_states in set_product(*exit_terms):
                    # divide distribution by 3 for each identity operator (I)
                    #   to average over measurements in 3 different bases
                    iden_fac = 3**sum( state == "I" for state in exit_states )
                    exit_conds = zip(exit_wires, exit_states)
                    init_dist.add(init_conds, exit_conds, projected_dist / iden_fac)

        ### end construction of init_dist

        if init_dist is frag_dist: continue

        for conditions, dist in init_dist:
            init_conds = set( cond for cond in conditions if cond[0] )
            exit_conds = set( cond for cond in conditions if not cond[0] )
            if not init_conds:
                frag_dist.add(conditions, dist)
                continue

            init_terms = [ _dist_terms_IZXY(state) for state in init_states ]
            for init_ops in set_product(*init_terms):
                iden_fac = 3**sum( op == "I" for op in init_ops )
                init_conds = zip(init_wires, init_ops)
                frag_dist.add(init_conds, exit_conds, dist / iden_fac)

    # if we computed distributions in the same init/exit bases as we were asked for,
    #   then just return the conditional distribution we computed
    # otherwise, change bases appropriately
    if init_basis == final_init_basis and exit_basis == final_exit_basis:
        return frag_dist
    else:
        return frag_dist.shuffle_bases(final_init_basis, final_exit_basis)

# get amplitudes for a list of fragments
def get_fragment_amplitudes(fragments, wire_path_map, **kwargs):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = identify_init_exit_wires(wire_path_map, len(fragments))

    # compute conditional distributions over measurement outcomes for all fragments
    return [ _get_single_fragment_amplitudes(ff, ii, ee, **kwargs)
             for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

# get probabilities for a list of fragments
def get_fragment_probabilities(fragments, wire_path_map,
                               backend_simulator = "statevector_simulator",
                               init_basis = pauli, exit_basis = pauli,
                               dtype = tf.float64, **kwargs):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = identify_init_exit_wires(wire_path_map, len(fragments))

    # compute conditional distributions over measurement outcomes for all fragments
    return [ _get_single_fragment_probabilities(ff, ii, ee, backend_simulator,
                                                init_basis, exit_basis, dtype, **kwargs)
             for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

# get conditional amplitudes or probabilities as appropriate
def get_fragment_distributions(fragments, wire_path_map,
                               backend_simulator = "statevector_simulator",
                               init_basis = pauli, exit_basis = pauli, dtype = tf.float64,
                               force_probs = False, **kwargs):
    if backend_simulator == "statevector_simulator":
        frag_amplitudes = get_fragment_amplitudes(fragments, wire_path_map, **kwargs)
        if not force_probs: return frag_amplitudes
        else: return [ amplitudes.to_probabilities(init_basis, exit_basis)
                       for amplitudes in frag_amplitudes ]
    else:
        return get_fragment_probabilities(fragments, wire_path_map, backend_simulator,
                                          init_basis, exit_basis, dtype, **kwargs)
