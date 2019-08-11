#!/usr/bin/env python3

##########################################################################################
# methods to
# (i) unite conditional fragment distributions
# (ii) query united distributions
# (iii) sample united circuits (NOT YET IMPLEMENTED)
##########################################################################################

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from tensorflow_extension import tf_outer_product, tf_transpose

from itertools import product as set_product
from functools import reduce

from fragment_distributions import pauli_vecs
from fragment_simulator import identify_init_exit_wires

# determine whether a data type is complex
def _is_complex(tf_dtype):
    return tf_dtype in [ tf.dtypes.complex64, tf.dtypes.complex128 ]

# identify all stitches in a cut-up circuit
# note that circuit_wires is used to guarantee that stitches are ordered
def _identify_stitches(wire_path_map, circuit_wires):
    stitches = {}
    for wire in circuit_wires:
        # identify all init/exit wires in the path of this wire
        init_wires = wire_path_map[wire][1:]
        exit_wires = wire_path_map[wire][:-1]
        # add the stitches in this path
        stitches.update({ exit_wire : init_wire
                          for init_wire, exit_wire in zip(init_wires, exit_wires) })
    return stitches

# get metadata for a united distribution:
# (i) shape of the united distribution
# (ii) type of the united distribution
# (iii) type of the data contained in the united distribution
def _get_distribution_metadata(frag_dists):
    united_dist_shape = ()
    for frag_dist in frag_dists:
        for _, dist in frag_dist:
            united_dist_shape += tuple(dist.shape)
            dist_obj_type = type(dist) # type of dist *itself*
            dist_dat_type = dist.dtype # type of the data stored in dist
            break
    return united_dist_shape, dist_obj_type, dist_dat_type

# collect all conditional distributions from fragments
# with a given assignment of operators at every stitch
def _collect_tensor_factors(frag_dists, stitches, op_assignment):
    frag_exit_conds = [ set() for _ in range(len(frag_dists)) ]
    frag_init_conds = [ set() for _ in range(len(frag_dists)) ]
    for stitch_idx, ( exit_frag_wire, init_frag_wire ) in enumerate(stitches.items()):
        exit_frag_idx, exit_wire = exit_frag_wire
        init_frag_idx, init_wire = init_frag_wire
        frag_exit_conds[exit_frag_idx].add(( exit_wire, op_assignment[stitch_idx] ))
        frag_init_conds[init_frag_idx].add(( init_wire, op_assignment[stitch_idx] ))

    return [ frag_dist[init_conds, exit_conds]
             for frag_dist, init_conds, exit_conds
             in zip(frag_dists, frag_init_conds, frag_exit_conds) ]

# return the order of output wires in a united distribution
def _united_wire_order(wire_path_map, frag_wires):
    _, exit_wires = identify_init_exit_wires(wire_path_map, len(frag_wires))
    return [ ( frag_idx, wire )
             for frag_idx, wires in enumerate(frag_wires)
             for wire in wires if wire not in exit_wires[frag_idx] ]

# return a dictionary mapping fragment output wires to the output wires of a circuit
def _frag_output_wire_map(wire_path_map):
    return { path[-1] : wire for wire, path in wire_path_map.items() }

# determine the permutation of tensor factors taking an old wire order to a new wire order
# old/new_wire_order are lists of wires in an old/desired order
# wire_map is a dictionary identifying wires in old_wire_order with those in new_wire_order
def _axis_permutation(old_wire_order, new_wire_order, wire_map = None):
    if wire_map is None: wire_map = { wire : wire for wire in old_wire_order }
    output_wire_order = [ wire_map[wire] for wire in old_wire_order ]
    wire_permutation = [ output_wire_order.index(wire) for wire in new_wire_order ]
    return [ len(new_wire_order) - 1 - idx for idx in wire_permutation ][::-1]

# get the permutation to apply to the tensor factors of a united distribution
def _united_axis_permutation(wire_path_map, circuit_wires, frag_wires):
    output_wires = _united_wire_order(wire_path_map, frag_wires)
    output_wire_map = _frag_output_wire_map(wire_path_map)
    return _axis_permutation(output_wires, circuit_wires, output_wire_map)

# (i) initialize an empty distribution
# (ii) identify the operators to assign to each stitch
# (iii) determine the scalar factor at every stitch
def _get_uniting_objects(frag_dists, stitches, metadata = None):
    if not metadata: metadata = _get_distribution_metadata(frag_dists)
    united_dist_shape, dist_obj_type, dist_dat_type = metadata

    if _is_complex(dist_dat_type):

        # the conditional distributions are amplitudes
        stitch_ops = range(2)
        scalar_factor = 1

    if not _is_complex(dist_dat_type):

        # the conditional distributions are probabilities
        stitch_ops = list(pauli_vecs.keys())
        scalar_factor = 1/2**len(stitches)

    # initialize an empty probability distribution
    if dist_obj_type is tf.SparseTensor:
        indices = np.empty((0,len(united_dist_shape)))
        values = tf.constant([], dtype = dist_dat_type)
        empty_united_dist = tf.SparseTensor(indices, values, united_dist_shape)
    else:
        empty_united_dist = tf.zeros(united_dist_shape, dtype = dist_dat_type)

    return empty_united_dist, stitch_ops, scalar_factor

##########################################################################################

# unite fragment distributions to reconstruct an overall probability distribution
def unite_fragment_distributions(frag_dists, wire_path_map, circuit_wires, frag_wires,
                                 force_probs = True, status_updates = False):
    # identify all cuts ("stitches") with a dictionary mapping exit wires to init wires
    stitches = _identify_stitches(wire_path_map, circuit_wires)

    # determine metadata for the united distribution
    frag_metadata = _get_distribution_metadata(frag_dists)
    _, _, dist_dat_type = frag_metadata

    # initialize an empty distribution
    #   and identify the operators / scalar factor at each stitch
    united_dist, stitch_ops, scalar_factor \
        = _get_uniting_objects(frag_dists, stitches, frag_metadata)

    # loop over all assigments of stitch operators at all cut locations (stitches)
    for op_assignment in set_product(stitch_ops, repeat = len(stitches)):
        if status_updates: print(op_assignment)

        # collect tensor factors of this term in the combined distribution
        # and add to the united distribution
        dist_factors = _collect_tensor_factors(frag_dists, stitches, op_assignment)
        united_dist += reduce(tf_outer_product, dist_factors[::-1])

    if status_updates: print()

    # convert amplitudes to probabilities if appropriate
    if force_probs and _is_complex(dist_dat_type):
        united_dist = abs(united_dist)**2

    # sort wires/qubits appropriately before returning the distribution
    perm = _united_axis_permutation(wire_path_map, circuit_wires, frag_wires)
    return scalar_factor * tf_transpose(united_dist, perm)

##########################################################################################

# for each state in list of states at the end of a circuit,
# get the state at the end of the output wires on each fragment
def get_frag_states(states, wire_path_map, circuit_wires, frag_wires):
    frag_states = [ [ () for _ in range(len(frag_wires)) ]
                    for state in states ]
    output_wires = _united_wire_order(wire_path_map, frag_wires)
    output_wire_map = _frag_output_wire_map(wire_path_map)
    for frag_wire in output_wires:
        wire_idx = -circuit_wires.index(output_wire_map[frag_wire])-1
        for state_idx, state in enumerate(states):
            frag_states[state_idx][frag_wire[0]] \
                = (state[wire_idx],) + frag_states[state_idx][frag_wire[0]]
    return frag_states

# query the value of a united distribution at particular states
def query_united_distribution(frag_dists, wire_path_map, circuit_wires, frag_wires,
                              query_states = [], force_probs = True):
    # identify all cuts ("stitches") with a dictionary mapping exit wires to init wires
    stitches = _identify_stitches(wire_path_map, circuit_wires)

    # determine metadata for the united distribution
    frag_metadata = _get_distribution_metadata(frag_dists)
    _, dist_obj_type, dist_dat_type = frag_metadata

    # determine how to query distributions
    if dist_obj_type is tf.SparseTensor:
        def _query(dist, query_state):
            value = 0
            for idx, state in enumerate(dist.indices.numpy()):
                if all(np.equal(state, query_state)):
                    value = dist.values.numpy()[idx]
                    break
            return value
    else:
        def _query(dist, state): return dist.numpy()[state]

    # identify the operators and scalar factor at each stitch
    _, stitch_ops, scalar_factor \
        = _get_uniting_objects(frag_dists, stitches, frag_metadata)

    # initialize the values we are querying,
    # and identify the state on each fragment for each query state
    state_vals = np.zeros(len(query_states))
    frag_query_states = get_frag_states(query_states, wire_path_map,
                                        circuit_wires, frag_wires)

    # loop over all assigments of stitch operators at all cut locations (stitches)
    for op_assignment in set_product(stitch_ops, repeat = len(stitches)):
        # collect tensor factors of this term in the combined distribution
        # and add to the values we are querying
        dist_factors = _collect_tensor_factors(frag_dists, stitches, op_assignment)
        for query_idx, frag_states in enumerate(frag_query_states):
            dist_state_iter = zip(dist_factors, frag_states)
            state_vals[query_idx] \
                += np.prod([ _query(dist,state) for dist, state in dist_state_iter ])

    # convert amplitudes to probabilities if appropriate
    if force_probs and _is_complex(dist_dat_type):
        state_vals = abs(state_vals)**2

    return scalar_factor * state_vals

##########################################################################################

def sample_united_distribution(frag_dists, wire_path_map, circuit_wires, frag_wires):
    frag_metadata = _get_distribution_metadata(frag_dists)
    united_dist_shape, dist_obj_type, dist_dat_type = frag_metadata

# todo:
# -- write method to pick samples from reconsturcted (full / positive) distributions
# -- write method to convert "actual" sample histogram into "corrected" sample histogram
# -- write "identity subtractor" to minimize the weight of negative terms (?)

# -- MIT QASM <--> Intel QS circuit simulation backend
