#!/usr/bin/env python3

import numpy as np
import qiskit as qs

##########################################################################################
# this script simulates circuit fragments
##########################################################################################

# return an empty circuit on the registers of an existing circuit
def empty_circuit(circuit):
    registers = set( wire[0] for wire in circuit.qubits + circuit.clbits )
    return qs.QuantumCircuit(*registers)

# act with the given gates acting on the given qubits
def act_gates(circuit, gates, *qubits):
    new_circuit = empty_circuit(circuit)
    for gate in gates[::-1]:
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
frag_exit_apnds = [ ( "Z", [ gates.IdGate() ] ),
                    ( "X", [ gates.HGate() ] ),
                    ( "Y", [ gates.SGate().inverse(), gates.HGate() ] ) ]

# get a distributions over measurement outcomes for a circuit fragment
# accepts a list of fragments, and lists of wires (i.e. in that fragment) that are
#   respectively "initialization" wires (on which to prepare states)
#   and "exit wires" (on which to make measurements)
# returns a conditional distribution function in dictionary format:
# { ( set( ( < initialized wire >, < initialized state >   ) ),
#     set( ( < measured wire >,    < measurement outcome > ) ) :
#   < measurement outcome distribution function, projected onto the exit-wire outcomes > }
def get_fragment_distribution(fragment, init_wires = None, exit_wires = None):
    if init_wires: # if we have wires to initialize into various states

        # pick the first init wire for state initialization
        init_wire, other_wires = init_wires[0], init_wires[1:]

        frag_dist = {} # distribution function we will return
        for state, prepend_op in frag_init_preps: # for each state we will prepare

            # construct the circuit to prepare the state we want
            prep_circuit = act_gates(fragment, prepend_op, init_wire)

            # get the distribution function for each prepared state
            init_frag_dist = get_fragment_distribution(prep_circuit + fragment,
                                                       other_wires, exit_wires)

            # combine the distribution functions
            for all_keys, dist in init_frag_dist.items():
                init_keys, exit_keys = all_keys
                new_init_keys = init_keys.union({ ( init_wire, state ) })
                frag_dist[new_init_keys, exit_keys] = dist

        return frag_dist

    if exit_wires: # if we have wires to measure in various bases

        # pick the first exit wire for measurement
        exit_wire, other_wires = exit_wires[0], exit_wires[1:]

        # axis to delete when projecting the distribution function (stored as a numpy array),
        #   and the shape of the final distribution function
        del_axis = len(fragment.qubits) - 1 - fragment.qubits.index(exit_wire)
        new_dist_shape = (2,)*(len(fragment.qubits)-1)

        frag_dist = {} # distribution function we will return
        for measurement, append_op in frag_exit_apnds: # for each measurement basis

            # construct the circuit to measure in the right basis
            apnd_circuit = act_gates(fragment, append_op, exit_wire)

            # get the distribution function for each measurement
            exit_frag_dist = get_fragment_distribution(fragment + apnd_circuit,
                                                       init_wires, other_wires)

            # combine the distribution functions
            for all_keys, dist in exit_frag_dist.items():
                init_keys, exit_keys = all_keys
                for outcome, bit_state in [ ( "+", 0 ), ( "-", 1 ) ]:
                    new_exit_keys = exit_keys.union({ ( exit_wire, outcome + measurement ) })
                    new_dist = np.delete(dist, 1-bit_state, axis = del_axis)
                    new_dist = np.reshape(new_dist, new_dist_shape)
                    frag_dist[init_keys, new_exit_keys] = new_dist

        return frag_dist

    # if no init_frag_wires and no exit_frag_wires
    distribution = get_circuit_distribution(fragment)
    return { ( frozenset(), frozenset() ) : distribution }
