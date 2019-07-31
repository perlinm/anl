#!/usr/bin/env python3

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

# construct a tensor network with a translationally-invariant (crystal) structure
def crystal_network(lattice_shape, tensor_bundle, lattice_vectors):

    lattice_dim = len(lattice_shape) # dimension of lattice
    net = tn.TensorNetwork() # initialize empty tensor network

    # add vectors on the periodic lattice
    periodicity = np.array(lattice_shape)
    def _add_vecs(loc, shift):
        return tuple( ( np.array(loc) + np.array(shift) ) % periodicity )

    # identify all tensor locations
    tensor_locations = [ (0,)*len(lattice_shape) ]
    for lattice_vector in lattice_vectors:
        for base_loc in tensor_locations:
            new_loc = _add_vecs(base_loc, lattice_vector)
            if new_loc not in tensor_locations:
                tensor_locations.append(new_loc)
            else: break

    # make all nodes, indexed by lattice coorinates
    nodes = { node_loc : net.add_node(tensor_bundle(node_loc), name = str(node_loc))
              for node_loc in tensor_locations }

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for base_loc in tensor_locations: # for every tensor location

        # for each lattice vector
        for axis, lattice_vector in enumerate(lattice_vectors):

            # identify the neighboring tensor
            trgt_loc = _add_vecs(base_loc, lattice_vector)

            # identify the contracted axes of these tensors
            base_axis, trgt_axis = axis, axis + len(lattice_vectors)

            # connect the neighboring nodes (tensors)
            edge = (base_loc, trgt_loc)
            edges[edge] = net.connect(nodes[base_loc][base_axis],
                                      nodes[trgt_loc][trgt_axis],
                                      name = str(edge))

    return net, nodes, edges

# construct tensor network on a periodic cubic lattice
def cubic_network(lattice_shape, tensor_bundle):
    lattice_vectors = [ tuple( 0 if jj != kk else 1
                               for jj in range(len(lattice_shape)) )
                        for kk in range(len(lattice_shape)) ]
    return crystal_network(lattice_shape, tensor_bundle, lattice_vectors)

# construct tensor network on a checkerboard lattice
def checkerboard_network(lattice_shape, tensor_bundle):
    assert(len(lattice_shape) == 2) # we can only do 2-D checkerboard lattices
    assert(all([ num % 2 == 0 for num in lattice_shape ])) # even-sized lattices only
    lattice_vectors = [ (1,1), (1,-1) ]
    return crystal_network(lattice_shape, tensor_bundle, lattice_vectors)

# construct a "bubbler" for a cubic network by swallowing one node at a time
# the swallowing procedure minimizes, at each step:
# (i) the number of dangling edges, and
# (ii) the distance of the bubbled node from the "origin", or first node
def cubic_bubbler(lattice_shape):
    # identify all "free", un-bubbled nodes
    free_nodes = list(np.ndindex(lattice_shape))

    # add two node vectors
    def add(xx, yy):
        return tuple( ( pos_xx + pos_yy ) % size
                      for pos_xx, pos_yy, size in zip(xx, yy, lattice_shape) )

    # identify all nodes adjacent to a given node
    def adjacent_nodes(node):
        return [ add(node, step) for step in [ (0,1), (1,0), (0,-1), (-1,0) ] ]

    # define a distance from the origin
    def dist(node):
        return sum( ( min(pos, size-pos) )**2 for pos, size in zip(node, lattice_shape) )

    # swallow a single (arbitrary) node
    bubbler = [ (0,)*len(lattice_shape) ]
    free_nodes.remove(bubbler[0])

    # keep track of all adjacent nodes and their distance from the "origin"
    adjacency_data = { adjacent_node : { "dist" : dist(adjacent_node) }
                       for adjacent_node in adjacent_nodes(bubbler[0]) }

    # return the tuple determining the order in which adjacent nodes should be swallowed
    def bubbling_order(node):
        return ( adjacency_data[node]["gain"], adjacency_data[node]["dist"] )

    while free_nodes:
        # determine the numer of dangling edges we would gain
        # from absorbing each adjacent node
        for adj_node, adj_node_data in adjacency_data.items():
            adj_adj_nodes = [ node for node in adjacent_nodes(adj_node) ]
            adj_node_data["gain"] = sum( -1 if node in bubbler else 1
                                         for node in adj_adj_nodes )

        # identify the node to swallow
        new_node = sorted(adjacency_data.keys(), key = bubbling_order)[0]

        bubbler.append(new_node)
        free_nodes.remove(new_node)
        del adjacency_data[new_node]

        # update adjacency data
        for node in adjacent_nodes(new_node):
            if node not in bubbler:
                adjacency_data[node] = { "dist" : dist(node) }

    return bubbler
