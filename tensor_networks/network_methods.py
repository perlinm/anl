#!/usr/bin/env python3

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

# construct tensor network on a periodic primitive hypercubic lattice
def cubic_network(lattice_shape, tensor_bundle):

    lattice_dim = len(lattice_shape) # dimension of lattice
    net = tn.TensorNetwork() # initialize empty tensor network

    # make all nodes, indexed by lattice coorinates
    nodes = { node_idx : net.add_node(tensor_bundle(node_idx), name = str(node_idx))
              for node_idx in np.ndindex(lattice_shape) }

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for base_idx in np.ndindex(lattice_shape): # for vertex index
        # for each physical axis
        for axis in range(lattice_dim):
            # identify the tensor axes corresponding each direction along this physical axis
            base_axis, trgt_axis = axis, axis + lattice_dim

            # identify the neighboring vertex in the direction of base_axis
            trgt_idx = list(base_idx)
            trgt_idx[base_axis] = ( trgt_idx[base_axis] + 1 ) % lattice_shape[base_axis]
            trgt_idx = tuple(trgt_idx)

            # connect the neighboring nodes (tensors)
            edge = (base_idx, trgt_idx)
            edges[edge] = net.connect(nodes[base_idx][base_axis],
                                      nodes[trgt_idx][trgt_axis],
                                      name = str(edge))

    return net, nodes, edges

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
