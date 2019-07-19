#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from tensorflow_extension import tf_outer_product, tf_transpose

from itertools import product as set_product
from itertools import combinations as set_combinations
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

# identify all stitches in a cut-up circuit
# note that circuit_wires is used to guarantee an order the stitches
def identify_stitches(wire_path_map, circuit_wires):
    stitches = {}
    for wire in circuit_wires:
        # identify all init/exit wires in the path of this wire
        init_wires = wire_path_map[wire][1:]
        exit_wires = wire_path_map[wire][:-1]
        # add the stitches in this path
        stitches.update({ exit_wire : init_wire
                          for init_wire, exit_wire in zip(init_wires,exit_wires) })
    return stitches

# act with the given gates on the given qubits
def act_gates(circuit, gates, *qubits):
    # build an empty circuit on the registers of an existing circuit
    registers = set( wire[0] for wire in circuit.qubits + circuit.clbits )
    new_circuit = qs.QuantumCircuit(*registers)
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
def get_circuit_probabilities(circuit, backend_simulator = "statevector_simulator",
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

    # sort init wires to fix their order
    init_wires = sorted(init_wires, key = lambda wire : fragment.qubits.index(wire))

    # for every choice of states prepared on all on init wires
    for init_states in set_product(range(2), repeat = len(init_wires)):
        init_conds = frozenset(zip(init_states, init_wires))

        # build circuit with the "input" states prepared appropriately
        prep_circuits = [ act_gates(fragment, prep_gate(_state_ZXY(bit_state)), wire)
                          for bit_state, wire in init_conds ]
        init_circuit = reduce(lambda x, y : x + y, prep_circuits + [ fragment ])

        # get the quantum state vector for the circuit
        all_amplitudes = get_circuit_amplitudes(init_circuit, **kwargs)

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

    # sort init wires to fix their order
    init_wires = sorted(init_wires, key = lambda wire : fragment.qubits.index(wire))

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
            full_dist = get_circuit_probabilities(circuit, backend_simulator,
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
def get_fragment_amplitudes(fragments, wire_path_map, **kwargs):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = identify_init_exit_wires(wire_path_map, len(fragments))

    # compute conditional distributions over measurement outcomes for all fragments
    return [ get_single_fragment_amplitudes(ff, ii, ee, **kwargs)
             for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

# get probabilities for a list of fragments
def get_fragment_probabilities(fragments, wire_path_map,
                               backend_simulator = "statevector_simulator",
                               init_op_basis = SIC, dtype = tf.float64, **kwargs):
    # collect lists of initialization / exit wires for all fragments
    init_wires, exit_wires = identify_init_exit_wires(wire_path_map, len(fragments))

    def _get_frag_probs(fragment, init_wires, exit_wires):
        return get_single_fragment_probabilities(fragment, init_wires, exit_wires,
                                                 backend_simulator, init_op_basis,
                                                 dtype, **kwargs)

    # compute conditional distributions over measurement outcomes for all fragments
    return [ _get_frag_probs(ff, ii, ee)
             for ff, ii, ee in zip(fragments, init_wires, exit_wires) ]

# get conditional amplitudes or probabilities as appropriate
def get_fragment_distributions(fragments, wire_path_map,
                               backend_simulator = "statevector_simulator",
                               init_op_basis = SIC, dtype = tf.float64,
                               force_probs = False, **kwargs):
    if backend_simulator == "statevector_simulator":
        frag_amplitudes = get_fragment_amplitudes(fragments, wire_path_map, **kwargs)
        if not force_probs: return frag_amplitudes
        else: return [ amplitudes.to_probabilities() for amplitudes in frag_amplitudes ]
    else:
        return get_fragment_probabilities(fragments, wire_path_map, backend_simulator,
                                          init_op_basis, dtype, **kwargs)

##########################################################################################
# methods to combine conditional fragment distributions
##########################################################################################

# get metadata for a reconstructed distribution:
# (i) shape of the reconstructed distribution
# (ii) type of the reconstructed distribution
# (iii) type of the data contained in the reconstructed distribution
def get_reconstruction_metadata(frag_dists):
    reconstructed_dist_shape = ()
    for frag_dist in frag_dists:
        for _, dist in frag_dist:
            reconstructed_dist_shape += tuple(dist.shape)
            dist_obj_type = type(dist) # type of dist *itself*
            dist_dat_type = dist.dtype # type of the data stored in dist
            break
    return reconstructed_dist_shape, dist_obj_type, dist_dat_type

# collect all conditional distributions from fragments
# with a given assignment of operators at every stitch
def collect_distribution_factors(frag_dists, stitches, op_assignment,
                                 use_subtraction_scheme = False):
    frag_exit_conds = [ set() for _ in range(len(frag_dists)) ]
    frag_init_conds = [ set() for _ in range(len(frag_dists)) ]
    for stitch_idx, ( exit_frag_wire, init_frag_wire ) in enumerate(stitches.items()):
        exit_frag_idx, exit_wire = exit_frag_wire
        init_frag_idx, init_wire = init_frag_wire
        frag_exit_conds[exit_frag_idx].add(( op_assignment[stitch_idx], exit_wire ))
        frag_init_conds[init_frag_idx].add(( op_assignment[stitch_idx], init_wire ))

    dists = [ frag_dist[init_conds, exit_conds]
              for frag_dist, init_conds, exit_conds
              in zip(frag_dists, frag_init_conds, frag_exit_conds) ]

    if not use_subtraction_scheme: return dists

    # for every distribution f(M) with an init condition M,
    #   take f(M) --> f(M) - 1/3 * f(I)
    else:

        for frag_idx in range(len(frag_dists)):
            frag_dist = frag_dists[frag_idx]
            init_conds = frag_init_conds[frag_idx]
            exit_conds = frag_exit_conds[frag_idx]
            for iden_num in range(1,len(init_conds)+1):
                factor = (-1/3)**iden_num
                for comb in set_combinations(init_conds, iden_num):
                    new_comb = { ( "I", wire ) for _, wire in comb }
                    new_init_conds = init_conds.difference(comb).union(new_comb)
                    dists[frag_idx] += factor * frag_dist[new_init_conds, exit_conds]
        return dists

# return the order of output wires in a reconstructed distribution
def reconstructed_wire_order(wire_path_map, frag_wires):
    _, exit_wires = identify_init_exit_wires(wire_path_map, len(frag_wires))
    return [ ( frag_idx, wire )
             for frag_idx, wires in enumerate(frag_wires)
             for wire in wires if wire not in exit_wires[frag_idx] ]

# return a dictionary mapping fragment output wires to the output wires of a circuit
def frag_output_wire_map(wire_path_map):
    return { path[-1] : wire for wire, path in wire_path_map.items() }

# determine the permutation of tensor factors taking an old wire order to a new wire order
# old/new_wire_order are lists of wires in an old/desired order
# wire_map is a dictionary identifying wires in old_wire_order with those in new_wire_order
def axis_permutation(old_wire_order, new_wire_order, wire_map = None):
    if wire_map is None: wire_map = { wire : wire for wire in old_wire_order }
    output_wire_order = [ wire_map[wire] for wire in old_wire_order ]
    wire_permutation = [ output_wire_order.index(wire) for wire in new_wire_order ]
    return [ len(new_wire_order) - 1 - idx for idx in wire_permutation ][::-1]

# get the permutation to apply to the tensor factors of a reconstructed distribution
def reconstructed_axis_permutation(wire_path_map, circuit_wires, frag_wires):
    output_wires = reconstructed_wire_order(wire_path_map, frag_wires)
    output_wire_map = frag_output_wire_map(wire_path_map)
    return axis_permutation(output_wires, circuit_wires, output_wire_map)

# given the state at the end of a circuit,
# get the state at the end of the output wires on each fragment
def get_frag_states(state, wire_path_map, circuit_wires, frag_wires):
    frag_states = [ () for _ in range(len(frag_wires)) ]
    output_wires = reconstructed_wire_order(wire_path_map, frag_wires)
    output_wire_map = frag_output_wire_map(wire_path_map)
    for frag_wire in output_wires:
        state_idx = -circuit_wires.index(output_wire_map[frag_wire])-1
        frag_states[frag_wire[0]] = (state[state_idx],) + frag_states[frag_wire[0]]
    return frag_states

# combine conditional distributions of fragments into a circuit distribution
def combine_fragment_distributions(frag_dists, wire_path_map, circuit_wires, frag_wires,
                                   stitch_basis = SIC, return_probs = True,
                                   discard_negative_terms = False,
                                   use_subtraction_scheme = False,
                                   query_state = None, status_updates = False):
    reconstructing_distribution = not query_state
    include_negative_terms = not discard_negative_terms

    # sanity check: if we are subtracting the identity operator at each stitch,
    #   then we must be including negative terms
    assert( not ( use_subtraction_scheme and discard_negative_terms ) )

    # identify all cuts ("stitches" with a dictionary mapping exit wires to init wires
    stitches = identify_stitches(wire_path_map, circuit_wires)

    # determine metadata for the reconstructed distribution
    reconstructed_dist_shape, dist_obj_type, dist_dat_type \
        = get_reconstruction_metadata(frag_dists)

    def _is_complex(tf_dtype):
        return tf_dtype in [ tf.dtypes.complex64, tf.dtypes.complex128 ]

    # determine which operators to assign to stitches,
    # as well as the scalar factor associated with any given operator assignment
    if _is_complex(dist_dat_type):

        # the conditional distributions are amplitudes,
        # so we only need to assign |0> and |1> states to stitches
        stitch_ops = range(2)
        def _scalar_factor(_): return 1

        # sanity check: make sure that we are not both
        # (i) discarding negative terms and (ii) returning amplitudes
        # there are no "negative terms" if we are working with amplitudes
        assert( include_negative_terms or return_probs )

        # if we are returning probabilities, allow discarding negative terms when
        #   reconstructing the overall probability distribution
        # doing so requires converting all conditional amplitudes --> probabilities
        if return_probs and discard_negative_terms:
            frag_dists = [ amplitudes.to_probabilities() for amplitudes in frag_dists ]
            _, dist_obj_type, dist_dat_type = get_reconstruction_metadata(frag_dists)

    if not _is_complex(dist_dat_type):

        # the conditional distributions are probabilities

        # determien which operators to assign to each stitch
        if stitch_basis == SIC:
            stitch_ops = list(state_vecs_SIC.keys())
            def _scalar_factor(op_assignment):
                return np.product([ -1 if op == "I" else 3/2 for op in op_assignment ])
        else:
            stitch_ops = list(state_vecs_ZXY.keys())
            def _scalar_factor(op_assignment):
                return (-1)**np.sum( op == "I" for op in op_assignment )

        # if we are including negative terms and not using the identity operator
        # subtraction scheme, then add "I" to the set of stitch operators
        if include_negative_terms and not use_subtraction_scheme:
            stitch_ops += [ "I" ]

    if reconstructing_distribution:

        # initialize an empty probability distribution
        if dist_obj_type is tf.SparseTensor:
            indices = np.empty((0,len(reconstructed_dist_shape)))
            values = tf.constant([], dtype = dist_dat_type)
            reconstructed_dist = tf.SparseTensor(indices, values, reconstructed_dist_shape)
        else:
            reconstructed_dist = tf.zeros(reconstructed_dist_shape, dtype = dist_dat_type)

    else: # only performing a state query

        state_val = 0 # initialize a zero value for the query

        # figure out what state should be on the output of each fragment
        frag_states = get_frag_states(query_state, wire_path_map, circuit_wires, frag_wires)

    # if we are throwing out terms contributing to the overall probability distribution,
    # then we need to keep track of the overall normalization of the distribution we get
    if discard_negative_terms:
        overall_normalization = 0

    # loop over all assigments of stitch operators at all cut locations (stitches)
    for op_assignment in set_product(stitch_ops, repeat = len(stitches)):
        if status_updates: print(op_assignment)

        # identify the scalar factor associated with this assignment of stitch operators
        scalar_factor = _scalar_factor(op_assignment)

        # collect tensor factors of this term in the combined distribution
        dist_factors = collect_distribution_factors(frag_dists, stitches, op_assignment,
                                                    use_subtraction_scheme)

        # add to the reconstructed distribution
        if reconstructing_distribution:
            reconstructed_dist += scalar_factor * reduce(tf_outer_product, dist_factors[::-1])

        else: # only performing a state query
            dist_state_iter = zip(dist_factors, frag_states)
            state_val += scalar_factor * np.prod([ dist.numpy()[state]
                                                   for dist, state in dist_state_iter ])

        # keep trach of normalization if necessary
        if discard_negative_terms:
            overall_normalization \
                += scalar_factor * np.prod([ tf.reduce_sum(dist).numpy()
                                             for dist in dist_factors ])

    if reconstructing_distribution:

        # square amplitudes to get probabilities if appropriate
        if return_probs and _is_complex(dist_dat_type):
            reconstructed_dist = abs(reconstructed_dist)**2

        # normalize the overall distribution if appropriate
        if discard_negative_terms:
            reconstructed_dist /= overall_normalization

        # sort wires/qubits appropriately before returning the distribution
        perm = reconstructed_axis_permutation(wire_path_map, circuit_wires, frag_wires)
        return tf_transpose(reconstructed_dist, perm)

    else: # only performing a state query

        # square amplitudes to get probabilities if appropriate
        if return_probs and _is_complex(dist_dat_type):
            state_val = abs(state_val)**2

        # normalize the overall distribution if appropriate
        if discard_negative_terms:
            state_val /= overall_normalization

        return state_val

# todo:
# -- write separate method for state query; allow query of many states
# -- write method to pick samples from reconsturcted (full / positive) distributions
# -- write method to convert sample histogram into "modified" sample histogram
# -- write "identity subtractor" to minimize the weight of negative terms (?)

# -- MIT QASM <--> Intel QS circuit simulation backend
