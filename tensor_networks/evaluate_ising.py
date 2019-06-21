#!/usr/bin/env python3

import os
import numpy as np
np.set_printoptions(linewidth = 200)

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.enable_v2_behavior()

import tensornetwork as tn

from quantum_contract import quantum_contract


inv_temp = 1 # inv_temperature
lattice_shape = (3,3) # lattice sites per axis

# value of tensor at a given inv_temperature for particular indices
def tensor_val(idx, inv_temp):
    s_idx = 2 * np.array(idx) - 1
    return ( 1 + np.prod(s_idx) ) / 2 * np.exp(inv_temp * np.sum(s_idx)/2)

# for a given inv_temperature, return a single tensor in the (translationally-invariant) network
def make_tensor(inv_temp, shape):
    tensor_shape = (2,)*len(shape)*2 # dimension of space at each index of the tensor
    tensor_vals = [ tensor_val(idx, inv_temp) for idx in np.ndindex(tensor_shape) ]
    return tf.reshape(tf.constant(tensor_vals), tensor_shape)

# construct tensor network in a hyperrectangular lattice
def make_net(inv_temp, shape = lattice_shape):
    net = tn.TensorNetwork() # initialize empty tensor network

    # make all nodes, indexed by lattice coorinates
    tensor = make_tensor(inv_temp, shape)
    nodes = { idx : net.add_node(tensor, name = str(idx))
              for idx in np.ndindex(shape) }

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for axis in range(len(shape)): # for each axis of the lattice

        # choose the axes of tensors that we will contract
        dir_fst, dir_snd = axis, axis + len(shape)

        for idx_fst in nodes: # loop over all nodes

            # identify the "next" neighboring node along this lattice axis
            idx_snd = list(idx_fst)
            idx_snd[dir_fst] = ( idx_snd[dir_fst] + 1 ) % shape[dir_fst]
            idx_snd = tuple(idx_snd)

            # connect up the nodes
            edge = (idx_fst,idx_snd)
            edges[edge] = net.connect(nodes[idx_fst][dir_fst],
                                      nodes[idx_snd][dir_snd],
                                      name = str(edge))

    return net, nodes, edges

net, nodes, edges = make_net(inv_temp)
tn.contractors.naive(net)
net_val = net.get_final_node().tensor.numpy()
print(net_val)

net, nodes, edges = make_net(inv_temp)
net_prob, net_norm = quantum_contract(nodes.values())
net_val = net_norm * np.sqrt(net_prob)

print("probability:", net_prob)
print("network value:", net_val)

inv_temps = np.linspace(0, 1.6, 20)
vals_Z = np.zeros(len(inv_temps))
for jj, inv_temp in enumerate(inv_temps):
    net, nodes, edges = make_net(inv_temp)
    tn.contractors.naive(net)
    net_val = net.get_final_node().tensor.numpy()
    vals_Z[jj] = net_val

import matplotlib.pyplot as plt

plt.plot(inv_temps, np.log(vals_Z) / np.prod(lattice_shape), "k.")
plt.xlim(0, inv_temps.max())
plt.ylim(0, plt.gca().get_ylim()[1])
plt.xlabel(r"$\beta J$")
plt.ylabel(r"$\log(Z)/V$")
plt.tight_layout()
plt.show()
