#!/usr/bin/env python3

import qiskit as qs

from circuit_cutter import cut_circuit

# construct a circuit that makes two bell pairs
qubits = qs.QuantumRegister(3, "q")
circ = qs.QuantumCircuit(qubits)
circ.h(qubits[0])
circ.cx(qubits[0], qubits[1])
circ.cx(qubits[1], qubits[2])
circ.barrier()
for qubit in qubits:
    circ.u0(qubit[1], qubit)

subcircs, subcirc_wiring, subcirc_stitches = cut_circuit(circ, (qubits[1],1))

print("original circuit:")
print(circ)

print()
for jj, subcirc in enumerate(subcircs):
    print("subcircuit index:", jj)
    print(subcirc)
    print("--------------------")

print()
print("subcircuit wiring:")
for old_wire, new_wire in subcirc_wiring.items():
    print(old_wire, "-->", *new_wire)

print()
print("subcircuit stitches:")
for old_wire, new_wire in subcirc_stitches.items():
    print(*old_wire, "-->", *new_wire)
