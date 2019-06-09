#!/usr/bin/env python3

import networkx as nx
import copy
import qiskit

##########################################################################################
# this script demonstrates "automatic" cutting of a quantum circuit in qiskit
# cutting is performed using method described in arxiv.org/abs/1904.00102
# developed using qiskit version 0.8.1
##########################################################################################

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
            if not any( qiskit.dagcircuit.DAGNode.semantic_eq(node, nx_node)
                        for nx_node in nx_subgraph.nodes() ):
                subgraph.remove_op_node(node)

        # identify wires used in this subgraph circuit
        wires = { node.wire for node in nx_subgraph.nodes() if node.type == "in" }

        subgraphs.append(subgraph)
        subgraph_wires.append(wires)

    if zip_output: return zip(subgraphs, subgraph_wires)
    else: return subgraphs, subgraph_wires

# accepts a circuit and cuts (qubit, op_number), where op_number is
#   the number of operations performed on the qubit before the cut
# returns (i) a list of subcircuits, and
# (ii) a list of "stitches" in the format ( ( <index of subcircuit>, <output wire> ),
#                                           ( <index of subcircuit>, <input wire> ) )
def cut_circuit(circuit, *cuts):
    if len(cuts) == 0: return circuit.copy()

    # initialize new qubit register and construct total circuit graph
    new_quantum_register = qiskit.QuantumRegister(len(cuts),"n")
    new_qubits = iter(new_quantum_register)
    graph = qiskit.converters.circuit_to_dag(circuit.copy())
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

    # identify the subgraphs addressing the wires in each stitch
    subgraph_stitches = set()
    for wire_0, wire_1 in stitches:
        index_0, index_1 = None, None
        for subgraph_index, wires in enumerate(subgraph_wires):
            if not index_0 and wire_0 in wires: index_0 = subgraph_index
            if not index_1 and wire_1 in wires: index_1 = subgraph_index
            if index_0 and index_1: break
        subgraph_stitches.add( ( ( index_0, wire_0 ), ( index_1, wire_1 ) ) )

    # convert the subgraphs into QuantumCircuit objects
    subcircuits = [ qiskit.converters.dag_to_circuit(graph) for graph in subgraphs ]
    return subcircuits, subgraph_stitches

##########################################################################################

# construct a circuit that makes two bell pairs
qubits = qiskit.QuantumRegister(4, "q")
circ = qiskit.QuantumCircuit(qubits)
circ.h(qubits[0])
circ.cx(qubits[0], qubits[1])
circ.barrier()
circ.h(qubits[2])
circ.cx(qubits[2], qubits[3])

# add identity operations for good measure (i.e. testing purposes)
circ.barrier()
for qubit in qubits:
    circ.iden(qubit)

subcircs, subcirc_stitches = cut_circuit(circ, (qubits[0],1), (qubits[2],1))

print("original circuit:")
print(circ)

print("subcircuits:")
for subcirc in subcircs:
    print(subcirc)

print("stitches:")
for stitch in subcirc_stitches:
    print(stitch)
