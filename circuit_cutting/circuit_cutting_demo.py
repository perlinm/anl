#!/usr/bin/env python3

import networkx as nx
import qiskit as qs
import copy, random

##########################################################################################
# this script demonstrates "automatic" cutting of a quantum circuit in qiskit
# cutting is performed using method described in arxiv.org/abs/1904.00102
# developed using qiskit version 0.8.1
##########################################################################################

# return a random character string of length k
def random_string(k = 100):
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k = k))

# get the terminal node of a qubit in a graph
def terminal_node(graph, qubit, termination_type):
    assert( termination_type in [ "in", "out" ] )
    for node in graph._multi_graph.nodes():
        if node.type == termination_type and node.wire == qubit:
            return node

# accept a circuit graph (i.e. in DAG form), and return a list of tuples:
# [ (<subgraph>, <list of wires used in this subgraph>) ]
# note that the subgraph circuits act on the full registers of the original graph circuit
def disjoint_subgraphs(graph, zip_output = True):
    # identify all subgraphs of nodes
    nx_subgraphs = nx.connected_component_subgraphs(graph.to_networkx().to_undirected())

    # convert subgraphs of nodes to circuit graphs
    subgraphs = []
    subgraph_wires = []
    for nx_subgraph in nx_subgraphs:
        # make a copy of the full graph, and remove nodes not in this subgraph
        subgraph = copy.deepcopy(graph)
        for node in subgraph.op_nodes():
            if not any( qs.dagcircuit.DAGNode.semantic_eq(node, nx_node)
                        for nx_node in nx_subgraph.nodes() ):
                subgraph.remove_op_node(node)

        # identify wires used in this subgraph circuit
        wires = { node.wire for node in nx_subgraph.nodes() if node.type == "in" }

        subgraphs.append(subgraph)
        subgraph_wires.append(wires)

    if zip_output: return zip(subgraphs, subgraph_wires)
    else: return subgraphs, subgraph_wires

# "trim" a circuit graph (i.e. in DAG form) by eliminating unused bits
# optionally accept a set of all used wires (with a promise that the set is correct)
# return trimmed graph, as well as a dictionary mapping old wires to new ones
def trimmed_graph(graph, graph_wires = None):
    # if we were not told which wires are used, figure it out
    if graph_wires is None:
        graph_wires = set()

        # identify all subgraphs
        nx_subgraphs = nx.connected_component_subgraphs(graph.to_networkx().to_undirected())
        for nx_subgraph in nx_subgraphs:
            # if there is only one edge in this subgraph, ignore it; it is an empty wire
            if len(nx_subgraph.edges()) == 1: continue

            # otherwise, add all wires from input nodes
            graph_wires.update({ node.wire for node in nx_subgraph if node.type == "in" })

    # construct map from old bits to new ones
    # qiskit refuses to construct empty registers, so we have to cover a few possible cases...
    old_qubits = [ wire for wire in graph_wires
                   if type(wire[0]) == qs.circuit.quantumregister.QuantumRegister ]
    old_clbits = [ wire for wire in graph_wires
                   if type(wire[0]) == qs.circuit.classicalregister.ClassicalRegister ]
    if len(old_qubits) > 0 and len(old_clbits) > 0:
        new_qubits = qs.QuantumRegister(len(old_qubits),"q")
        new_clbits = qs.ClassicalRegister(len(old_clbits),"c")
        trimmed_circuit = qs.QuantumCircuit(new_qubits, new_clbits)
    elif len(old_qubits) > 0 and len(old_clbits) == 0:
        new_qubits = qs.QuantumRegister(len(old_qubits),"q")
        new_clbits = []
        trimmed_circuit = qs.QuantumCircuit(new_qubits)
    elif len(old_qubits) == 0 and len(old_clbits) > 0:
        new_qubits = []
        new_clbits = qs.ClassicalRegister(len(old_clbits),"c")
        trimmed_circuit = qs.QuantumCircuit(new_clbits)
    else:
        trimmed_circuit = qs.QuantumCircuit()

    register_map = list(zip(old_qubits, new_qubits)) + list(zip(old_clbits, new_clbits))
    register_map = { old_bit : new_bit for old_bit, new_bit in register_map }

    # add all operations to the trimmed circuit
    for node in graph.topological_op_nodes():
        new_qargs = [ register_map[qubit] for qubit in node.qargs ]
        new_cargs = [ register_map[clbit] for clbit in node.cargs ]
        trimmed_circuit.append(node.op, qargs = new_qargs, cargs = new_cargs)

    return qs.converters.circuit_to_dag(trimmed_circuit), register_map

# accepts a circuit and cuts (qubit, op_number), where op_number is
#   the number of operations performed on the qubit before the cut; returns:
# (i) a list of subcircuits,
# (ii) a list of "stitches" in the format ( ( <index of output subcircuit>, <output wire> ),
#                                           ( <index of input subcircuit>,  <input wire> ) ),
# (iii) a dictionary taking input wires of the original circuit to input wires of subcircuits:
#       { <input wire in the original circuit> :
#         ( <index of subcircuit>, <corresponding input wire in subcircuit> ) }
def cut_circuit(circuit, *cuts):
    if len(cuts) == 0: return circuit.copy()

    # initialize new qubit register and construct total circuit graph
    new_quantum_register = qs.QuantumRegister(len(cuts),random_string())
    new_qubits = iter(new_quantum_register)
    graph = qs.converters.circuit_to_dag(circuit.copy())
    graph.add_qreg(new_quantum_register)

    # TODO: deal with barriers properly
    # barriers currently interfere with splitting a graph into subgraphs
    graph.remove_all_ops_named("barrier")

    # collect a dictionary summarizing cuts with { < qubit > : < set of cuts > }
    qubit_cuts = {}
    for qubit, op_number in cuts:
        if qubit_cuts.get(qubit) is None:
            qubit_cuts[qubit] = { op_number }
        else:
            qubit_cuts[qubit].add(op_number)

    # convert the sets of cuts in qubit_cuts to ordered lists
    for qubit in qubit_cuts.keys():
        qubit_cuts[qubit] = sorted(list(qubit_cuts[qubit]), reverse = True)

    # keep track of how many operations have been performed on each qubit
    qubit_op_count = { qubit : 0 for qubit in qubit_cuts.keys() }

    # tuples identifying which old/new qubits to stitch together
    stitches = set()

    # loop over all gates in this cirtuit, looking for places to cut
    for op_node in graph._multi_graph.nodes():
        if op_node.type != "op": continue

        # for each qubit that we have yet to cut...
        for cut_qubit, cut_pos in list(qubit_cuts.items()):

            # identify input/output nodes for this qubit
            cut_qubit_in = terminal_node(graph, cut_qubit, "in")
            cut_qubit_out = terminal_node(graph, cut_qubit, "out")

            # if we have reached an operation count that matches a cut we need to make...
            if qubit_op_count[cut_qubit] == cut_pos[-1]:

                # identify the new qubit we will insert
                new_qubit = next(new_qubits)
                new_qubit_in = terminal_node(graph, new_qubit, "in")
                new_qubit_out = terminal_node(graph, new_qubit, "out")
                graph._multi_graph.remove_edge(new_qubit_in, new_qubit_out)

                # replace the cut qubit in this node by a new qubit
                op_node.qargs[op_node.qargs.index(cut_qubit)] = new_qubit

                # identify all nodes downstream of this one
                op_descendants = nx.descendants(graph._multi_graph, op_node)

                for edge in list(graph._multi_graph.edges(data = True)):
                    # remove and replace incoming edges from a node with the cut cubit
                    if cut_qubit in edge[0].qargs and edge[1] == op_node:
                        graph._multi_graph.remove_edge(*edge[:2])
                        graph._multi_graph.add_edge(edge[0], cut_qubit_out,
                                                    name = f"{cut_qubit[0].name}[{cut_qubit[1]}]",
                                                    wire = cut_qubit)
                        graph._multi_graph.add_edge(new_qubit_in, edge[1],
                                                    name = f"{new_qubit[0].name}[{new_qubit[1]}]",
                                                    wire = new_qubit)

                    # fix downstream references to the cut cubit (in all edges)
                    if edge[1] in op_descendants and edge[2]["wire"] == cut_qubit:
                        graph._multi_graph.remove_edge(*edge[:2])
                        graph._multi_graph.add_edge(*edge[:2],
                                                    name = f"{new_qubit[0].name}[{new_qubit[1]}]",
                                                    wire = new_qubit)

                    # replace terminal node of the cut qubit by that of the new qubit
                    if edge[1] == cut_qubit_out:
                        graph._multi_graph.remove_edge(*edge[:2])
                        graph._multi_graph.add_edge(edge[0], new_qubit_out,
                                                    name = f"{new_qubit[0].name}[{new_qubit[1]}]",
                                                    wire = new_qubit)

                # fix downstream references to the cut cubit (in all nodes)
                for node in op_descendants:
                    if node.type == "op" and cut_qubit in node.qargs:
                        node.qargs[node.qargs.index(cut_qubit)] = new_qubit

                # identify the old/new qubit for stitching together
                stitches.add((cut_qubit, new_qubit))

                # remove this cut from our list of cuts
                cut_pos.pop()

            # remove this qubit if we do not need to cut it anymore
            if len(cut_pos) == 0: del qubit_cuts[cut_qubit]

        # update operation count for each qubit
        for node_qubit in op_node.qargs:
            try: qubit_op_count[node_qubit] += 1
            except: None

    # split the total circuit graph into subgraphs
    subgraphs, subgraph_wires = disjoint_subgraphs(graph, zip_output = False)

    # trim subgraphs, eliminating unused bits
    trimmed_subgraphs, wire_maps \
        = zip(*[ trimmed_graph(subgraph, wires)
                 for subgraph, wires in zip(subgraphs, subgraph_wires) ])

    # identify the subgraphs addressing the wires in each stitch
    subgraph_stitches = set()
    for wire_0, wire_1 in stitches:
        index_0, index_1 = None, None
        for subgraph_index, wires in enumerate(subgraph_wires):
            if not index_0 and wire_0 in wires: index_0 = subgraph_index
            if not index_1 and wire_1 in wires: index_1 = subgraph_index
            if index_0 and index_1: break
        wire_0 = wire_maps[index_0][wire_0]
        wire_1 = wire_maps[index_1][wire_1]
        subgraph_stitches.add( ( ( index_0, wire_0 ), ( index_1, wire_1 ) ) )

    # map each input wire in the original circuit to an input wire in subcircuits
    subcircuit_wiring = {}
    for subcircuit_index, wire_map in enumerate(wire_maps):
        for in_wire, out_wire in wire_map.items():
            if in_wire in circuit.qubits or in_wire in circuit.clbits:
                subcircuit_wiring[in_wire] = (subcircuit_index, out_wire)

    # convert the subgraphs into QuantumCircuit objects
    subcircuits = [ qs.converters.dag_to_circuit(graph)
                    for graph in trimmed_subgraphs ]
    return subcircuits, subgraph_stitches, subcircuit_wiring

##########################################################################################

# construct a circuit that makes two bell pairs
qubits = qs.QuantumRegister(4, "q")
circ = qs.QuantumCircuit(qubits)
circ.h(qubits[0])
circ.cx(qubits[0], qubits[1])
circ.barrier()
circ.h(qubits[2])
circ.cx(qubits[2], qubits[3])

# add identity operations for good measure (i.e. testing purposes)
circ.barrier()
for qubit in qubits:
    circ.iden(qubit)

subcircs, subcirc_stitches, subcirc_wiring = cut_circuit(circ, (qubits[0],1), (qubits[2],1))

print("original circuit:")
print(circ)

print()
print("subcircuits:")
for subcirc in subcircs:
    print(subcirc)

print()
print("subcircuit stitches:")
for stitch in subcirc_stitches:
    print(*stitch[0], "-->", *stitch[1])

print()
print("subcircuit wiring:")
for old_wire, new_wire in subcirc_wiring.items():
    print(old_wire, "-->", *new_wire)
