#!/usr/bin/env python3

import numpy as np
import tensornetwork as tn
from itertools import product as set_product

# add vectors on the periodic lattice
def _add(xx, yy, lattice_shape):
    return tuple( ( np.array(xx) + np.array(yy) ) % np.array(lattice_shape) )

# identify all occupied sites in a crystal
def _crystal_sites(lattice_shape, lattice_vectors):
    locations = [ (0,)*len(lattice_shape) ]
    for lattice_vector in lattice_vectors:
        for base_loc in locations:
            new_loc = _add(base_loc, lattice_vector, lattice_shape)
            if new_loc not in locations:
                locations.append(new_loc)
            else: break
    return locations

def _half(vec):
    return tuple( vv/2 for vv in vec )

##########################################################################################

# construct a tensor network with a crystal structure
def crystal_network(lattice_shape, lattice_vectors, tensor_bundle, link_tensors = False):

    lattice_dim = len(lattice_shape) # dimension of lattice
    net = tn.TensorNetwork() # initialize empty tensor network
    tensor_locations = _crystal_sites(lattice_shape, lattice_vectors)

    # make all nodes, indexed by lattice coorinates
    if not link_tensors:
        nodes = { node_loc : net.add_node(tensor_bundle(node_loc), name = str(node_loc))
                  for node_loc in tensor_locations }
    else:
        nodes = {}
        for node_loc in tensor_locations:
            nodes[node_loc] = net.add_node(tensor_bundle(node_loc), name = str(node_loc))
            for lattice_vector in lattice_vectors:
                link_loc = _add(node_loc, _half(lattice_vector), lattice_shape)
                nodes[link_loc] = net.add_node(tensor_bundle(link_loc),
                                               name = str(link_loc))

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for base_loc in tensor_locations: # for every tensor location

        # for each lattice vector
        for axis, lattice_vector in enumerate(lattice_vectors):

            # identify the neighboring tensor
            trgt_loc = _add(base_loc, lattice_vector, lattice_shape)

            # identify the contracted axes of these tensors
            base_axis, trgt_axis = axis, axis + len(lattice_vectors)

            if not link_tensors:

                # connect the neighboring nodes (tensors)
                edge = ( base_loc, trgt_loc )
                edges[edge] = net.connect(nodes[base_loc][base_axis],
                                          nodes[trgt_loc][trgt_axis],
                                          name = str(edge))

            else:

                link_loc = _add(base_loc, _half(lattice_vector), lattice_shape)

                edge = ( base_loc, link_loc )
                edges[edge] = net.connect(nodes[base_loc][base_axis],
                                          nodes[link_loc][0],
                                          name = str(edge))

                edge = ( link_loc, trgt_loc )
                edges[edge] = net.connect(nodes[link_loc][1],
                                          nodes[trgt_loc][trgt_axis],
                                          name = str(edge))

    return net, nodes, edges

# construct a "bubbler" for a crystal network by swallowing one node at a time
# the swallowing procedure minimizes, at each step:
# (i) the number of dangling edges, and
# (ii) the distance of the bubbled node from the "origin", or first node
def crystal_bubbler(lattice_shape, lattice_vectors, link_tensors = False):
    # identify all "free", un-bubbled nodes
    free_nodes = _crystal_sites(lattice_shape, lattice_vectors)

    # identify all nodes adjacent to a given node
    def _adjacent_nodes(node):
        return ( [ _add(node, step, lattice_shape)
                   for step in lattice_vectors ] +
                 [ _add(node, -np.array(step), lattice_shape)
                   for step in lattice_vectors ] )

    # define a distance from the origin
    def _dist(node):
        return sum( ( min(pos, size-pos) )**2 for pos, size in zip(node, lattice_shape) )

    # swallow a single (arbitrary) node
    bubbler = [ (0,)*len(lattice_shape) ]
    free_nodes.remove(bubbler[0])

    # keep track of all adjacent nodes and their distance from the "origin"
    adjacency_data = { adjacent_node : { "dist" : _dist(adjacent_node) }
                       for adjacent_node in _adjacent_nodes(bubbler[0]) }

    # return the tuple determining the order in which adjacent nodes should be swallowed
    def _bubbling_order(node):
        return ( adjacency_data[node]["gain"], adjacency_data[node]["dist"] )

    while free_nodes:
        # determine the numer of dangling edges we would gain
        # from absorbing each adjacent node
        for adj_node, adj_node_data in adjacency_data.items():
            adj_adj_nodes = [ node for node in _adjacent_nodes(adj_node) ]
            adj_node_data["gain"] = sum( -1 if node in bubbler else 1
                                         for node in adj_adj_nodes )

        # identify the node to swallow
        new_node = sorted(adjacency_data.keys(), key = _bubbling_order)[0]

        # swallow all links between bubbled nodes and new node
        if link_tensors:
            for lattice_vector in lattice_vectors:
                adj_to_new_node = _add(new_node, lattice_vector, lattice_shape)
                if adj_to_new_node in bubbler:
                    bubbler.append(_add(new_node, _half(lattice_vector), lattice_shape))

        # swallow new node and remove it from the list of free nodes
        bubbler.append(new_node)
        free_nodes.remove(new_node)
        del adjacency_data[new_node]

        # update adjacency data
        for node in _adjacent_nodes(new_node):
            if node not in bubbler:
                adjacency_data[node] = { "dist" : _dist(node) }

    return bubbler

##########################################################################################

### cubic lattice

def _cubic_lattice_vectors(lattice_shape):
    return [ tuple( 0 if jj != kk else 1
                    for jj in range(len(lattice_shape)) )
             for kk in range(len(lattice_shape)) ]

def cubic_network(lattice_shape, tensor_bundle, link_tensors = False):
    lattice_vectors = _cubic_lattice_vectors(lattice_shape)
    return crystal_network(lattice_shape, lattice_vectors, tensor_bundle, link_tensors)

def cubic_bubbler(lattice_shape, link_tensors = False):
    lattice_vectors = _cubic_lattice_vectors(lattice_shape)
    return crystal_bubbler(lattice_shape, lattice_vectors, link_tensors)

### checkerboard lattice

def _checkerboard_lattice_vectors(lattice_shape):
    assert(all([ num % 2 == 0 for num in lattice_shape ])) # even-sized lattices only
    signed_ones = set_product([+1,-1], repeat = len(lattice_shape)-1)
    return [ (1,) + vec for vec in signed_ones ]

def checkerboard_network(lattice_shape, tensor_bundle, link_tensors = False):
    lattice_vectors = _checkerboard_lattice_vectors(lattice_shape)
    return crystal_network(lattice_shape, lattice_vectors, tensor_bundle, link_tensors)

def checkerboard_bubbler(lattice_shape, link_tensors = False):
    lattice_vectors = _checkerboard_lattice_vectors(lattice_shape)
    return crystal_bubbler(lattice_shape, lattice_vectors, link_tensors)
