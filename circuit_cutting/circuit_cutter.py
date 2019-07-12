#!/usr/bin/env python3

import networkx as nx
import qiskit as qs
import copy

##########################################################################################
# this script cuts a quantum circuit built in qiskit
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
def trimmed_graph(graph, graph_wires = None, qreg_name = "q", creg_name = "c"):
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
                   if type(wire[0]) is qs.circuit.quantumregister.QuantumRegister ]
    old_clbits = [ wire for wire in graph_wires
                   if type(wire[0]) is qs.circuit.classicalregister.ClassicalRegister ]
    if len(old_qubits) > 0 and len(old_clbits) > 0:
        new_qubits = qs.QuantumRegister(len(old_qubits), qreg_name)
        new_clbits = qs.ClassicalRegister(len(old_clbits), creg_name)
        trimmed_circuit = qs.QuantumCircuit(new_qubits, new_clbits)
    elif len(old_qubits) > 0 and len(old_clbits) == 0:
        new_qubits = qs.QuantumRegister(len(old_qubits), qreg_name)
        new_clbits = []
        trimmed_circuit = qs.QuantumCircuit(new_qubits)
    elif len(old_qubits) == 0 and len(old_clbits) > 0:
        new_qubits = []
        new_clbits = qs.ClassicalRegister(len(old_clbits), creg_name)
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

# accepts a circuit and cuts (wire, op_number), where op_number is
#   the number of operations performed on the wire before the cut; returns:
# (i) a list of subcircuits
# (ii) a dictionary that identifies inputs to the original circuits with inputs of subcircuits:
#      { <wire in the original circuit> :
#        ( <index of subcircuit>, <wire in subcircuit> ) }
# (iii) a dictionary identifying how subcircuits should be stitched together;
#       this dictionary takes output wires of subcircuits to input wires of subcircuits:
#      { ( <index of subcircuit>, <wire in subcircuit> ) :
#        ( <index of subcircuit>, <wire in subcircuit> ) }
def cut_circuit(circuit, *cuts, qreg_name = "q", creg_name = "c"):
    if len(cuts) == 0: return circuit.copy()

    # assert that all cut wires are part of a quantum register
    assert(all( type(wire[0]) is qs.circuit.quantumregister.QuantumRegister
                for wire, _ in cuts ))

    # initialize new qubit register and construct total circuit graph
    new_reg_name = "_".join(set( wire[0].prefix
                                 for wire in circuit.qubits + circuit.clbits )) + "_new"
    new_register = qs.QuantumRegister(len(cuts),new_reg_name)
    new_wires = iter(new_register)
    graph = qs.converters.circuit_to_dag(circuit.copy())
    graph.add_qreg(new_register)

    # TODO: deal with barriers properly
    # barriers currently interfere with splitting a graph into subgraphs
    graph.remove_all_ops_named("barrier")

    # tuples identifying which old/new wires to stitch together
    stitches = set()

    # loop over all cuts from last to first
    for cut_wire, cut_location in sorted(cuts, key = lambda cut : -cut[1]):

        # identify terminal node of the wire we're cutting
        cut_wire_out = terminal_node(graph, cut_wire, "out")

        # identify the node before which to cut
        wire_nodes = [ node for node in graph.topological_op_nodes()
                       if cut_wire in node.qargs ]
        cut_node = wire_nodes[cut_location]

        # identify all nodes downstream of this one
        cut_descendants = nx.descendants(graph._multi_graph, cut_node)

        # identify the new wire to use
        new_wire = next(new_wires)
        new_wire_in = terminal_node(graph, new_wire, "in")
        new_wire_out = terminal_node(graph, new_wire, "out")
        graph._multi_graph.remove_edge(new_wire_in, new_wire_out)

        # replace all edges on this wire as appropriate
        for edge in [ edge[:2] for edge in graph._multi_graph.edges(data = True)
                      if edge[2]["wire"] == cut_wire ]:

            # if this edge ends at the node at which we're cutting, splice in the new wire
            if cut_wire in edge[0].qargs and edge[1] == cut_node:
                graph._multi_graph.remove_edge(*edge)
                graph._multi_graph.add_edge(edge[0], cut_wire_out,
                                            name = f"{cut_wire[0].name}[{cut_wire[1]}]",
                                            wire = cut_wire)
                graph._multi_graph.add_edge(new_wire_in, edge[1],
                                            name = f"{new_wire[0].name}[{new_wire[1]}]",
                                            wire = new_wire)
                continue # we are definitely done with this edge

            # fix downstream references to the cut wire (in all edges)
            if edge[1] in cut_descendants:
                # there may be multiple edges between the nodes in `edge`,
                # so first figure out which one we need to cut
                edge_data = graph._multi_graph.get_edge_data(*edge)
                for edge_key, data in edge_data.items():
                    if data["wire"] == cut_wire: break
                graph._multi_graph.remove_edge(*edge, edge_key)
                graph._multi_graph.add_edge(*edge,
                                            name = f"{new_wire[0].name}[{new_wire[1]}]",
                                            wire = new_wire)

            # replace downstream terminal node of the cut wire by that of the new wire
            if edge[1] == cut_wire_out:
                graph._multi_graph.remove_edge(*edge)
                graph._multi_graph.add_edge(edge[0], new_wire_out,
                                            name = f"{new_wire[0].name}[{new_wire[1]}]",
                                            wire = new_wire)

        ### end loop over edges

        # fix downstream references to the cut wire (in all nodes)
        for node in [ cut_node ] + list(cut_descendants):
            if node.type == "op" and cut_wire in node.qargs:
                node.qargs[node.qargs.index(cut_wire)] = new_wire

        # fix references to the cut wire in the set of stitches
        stitches = { ( start if start != cut_wire else new_wire, end )
                     for start, end in stitches }

        # identify the old/new wires to stitch together
        stitches.add((cut_wire, new_wire))

    ### end loop over cuts

    # split the total circuit graph into subgraphs
    subgraphs, subgraph_wires = disjoint_subgraphs(graph, zip_output = False)

    # trim subgraphs, eliminating unused bits
    trimmed_subgraphs, wire_maps \
        = zip(*[ trimmed_graph(subgraph, wires, qreg_name, creg_name)
                 for subgraph, wires in zip(subgraphs, subgraph_wires) ])

    # map each input wire in the original circuit to an input wire in subcircuits
    subcircuit_wiring = {}
    for subcircuit_index, wire_map in enumerate(wire_maps):
        for in_wire, out_wire in wire_map.items():
            if in_wire in circuit.qubits or in_wire in circuit.clbits:
                subcircuit_wiring[in_wire] = (subcircuit_index, out_wire)

    # identify the subgraphs addressing the wires in each stitch
    subgraph_stitches = {}
    for wire_0, wire_1 in stitches:
        index_0, index_1 = None, None
        for subgraph_index, wires in enumerate(subgraph_wires):
            if not index_0 and wire_0 in wires: index_0 = subgraph_index
            if not index_1 and wire_1 in wires: index_1 = subgraph_index
            if index_0 and index_1: break
        wire_0 = wire_maps[index_0][wire_0]
        wire_1 = wire_maps[index_1][wire_1]
        subgraph_stitches[( index_0, wire_0 )] = ( index_1, wire_1 )

    # convert the subgraphs into QuantumCircuit objects
    subcircuits = [ qs.converters.dag_to_circuit(graph)
                    for graph in trimmed_subgraphs ]
    return subcircuits, subcircuit_wiring, subgraph_stitches
