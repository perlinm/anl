#!/usr/bin/env python3

import numpy as np
import tensornetwork as tn
from itertools import product as set_product

# add vectors on the periodic lattice
def _add(xx, yy, lattice_shape = None):
    vector_sum = np.array(xx) + np.array(yy)
    if lattice_shape:
        vector_sum %= np.array(lattice_shape)
    return tuple( int(vv) if vv % 1 == 0 else vv for vv in vector_sum )

# identify all occupied sites in a crystal
def _crystal_sites(lattice_shape, lattice_vectors, initial_location = None):
    if initial_location:
        locations = [ initial_location ]
    else:
        locations = [ (0,)*len(lattice_shape) ]
    for lattice_vector in lattice_vectors:
        for base_loc in locations:
            new_loc = _add(base_loc, lattice_vector, lattice_shape)
            if new_loc not in locations:
                locations.append(new_loc)
            else: break
    return locations

def _half(vec):
    return tuple( vv//2 if vv % 2 == 0 else vv/2 for vv in vec )

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

# return a bubbler that "scans" through a lattice,
#   bubbling adjacent tensors in a single row,
#   and subsequently bubbling adjacent rows likewise
def scanning_bubbler(lattice_shape, lattice_vectors, link_tensors = False):
    if not link_tensors:
        return _crystal_sites(lattice_shape, lattice_vectors)

    else:
        zero_pos = (0,) * len(lattice_shape)
        half_vecs = [ _half(vec) for vec in lattice_vectors ]
        sublattice_scans = [ _crystal_sites(lattice_shape, lattice_vectors, initial_pos)
                             for initial_pos in [ zero_pos ] + half_vecs ]
        return [ pos for cell_scan in zip(*sublattice_scans) for pos in cell_scan ]

# return an bubbler that "alternates" through the lattice
#   to maximize the number of 0 <--> X tensor flattenings
def alternating_bubbler(lattice_shape, lattice_vectors, link_tensors = False):
    if not link_tensors:
        base_vec, other_vecs = lattice_vectors[0], lattice_vectors[1:]
        signed_ones = set_product([+1,-1], repeat = len(lattice_shape)-1)
        new_vectors = [ _add(base_vec, sign*np.array(vec))
                        for signs in signed_ones
                        for sign, vec in zip(signs, other_vecs) ]

        half_scan = scanning_bubbler(lattice_shape, new_vectors)
        return half_scan + [ _add(pos,base_vec,lattice_shape) for pos in half_scan ]

    else:
        zero_pos = (0,) * len(lattice_shape)
        half_vecs = [ _half(vec) for vec in lattice_vectors ]
        sublattice_scans = [ _crystal_sites(lattice_shape, lattice_vectors, initial_pos)
                             for initial_pos in half_vecs + [ zero_pos ] ]
        return [ pos for sublattice_scan in sublattice_scans for pos in sublattice_scan ]

##########################################################################################

### cubic lattice

def _cubic_lattice_vectors(lattice_shape):
    return [ tuple( 0 if jj != kk else 1
                    for jj in range(len(lattice_shape)) )
             for kk in range(len(lattice_shape)) ]

def cubic_network(lattice_shape, tensor_bundle, link_tensors = False):
    lattice_vectors = _cubic_lattice_vectors(lattice_shape)
    return crystal_network(lattice_shape, lattice_vectors, tensor_bundle, link_tensors)

def cubic_bubbler(lattice_shape, link_tensors = False, scan = True):
    lattice_vectors = _cubic_lattice_vectors(lattice_shape)
    if scan:
        return scanning_bubbler(lattice_shape, lattice_vectors, link_tensors)
    else:
        return alternating_bubbler(lattice_shape, lattice_vectors, link_tensors)

### checkerboard lattice

def _checkerboard_lattice_vectors(lattice_shape):
    assert(all([ num % 2 == 0 for num in lattice_shape ])) # even-sized lattices only
    signed_ones = set_product([+1,-1], repeat = len(lattice_shape)-1)
    return [ (1,) + vec for vec in signed_ones ]

def checkerboard_network(lattice_shape, tensor_bundle, link_tensors = False):
    lattice_vectors = _checkerboard_lattice_vectors(lattice_shape)
    return crystal_network(lattice_shape, lattice_vectors, tensor_bundle, link_tensors)

def checkerboard_bubbler(lattice_shape, link_tensors = False, scan = True):
    lattice_vectors = _checkerboard_lattice_vectors(lattice_shape)
    if scan:
        return scanning_bubbler(lattice_shape, lattice_vectors, link_tensors)
    else:
        return alternating_bubbler(lattice_shape, lattice_vectors, link_tensors)
