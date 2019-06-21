#!/usr/bin/env python3

import os
import numpy as np
np.set_printoptions(linewidth = 200)

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.enable_v2_behavior()
import tensornetwork as tn

import matplotlib.pyplot as plt
from quantum_contract import quantum_contract

##########################################################################################
# method for constructing a tensor network that represents the partition function
#   of a uniform 2-D Ising model: H = -\sum_{<j,k>} s_j s_k
##########################################################################################

# value of tensor at a given inv_temperature for particular indices
def tensor_val(idx, inv_temp):
    s_idx = 2 * np.array(idx) - 1
    return ( 1 + np.prod(s_idx) ) / 2 * np.exp(inv_temp * np.sum(s_idx)/2)

# for a given inv_temperature, return a single tensor in the (translationally-invariant) network
def make_tensor(inv_temp, lattice_shape):
    tensor_shape = (2,)*len(lattice_shape)*2 # dimension of space at each index of the tensor
    tensor_vals = [ tensor_val(idx, inv_temp) for idx in np.ndindex(tensor_shape) ]
    return tf.reshape(tf.constant(tensor_vals), tensor_shape)

# construct tensor network in a hyperrectangular lattice
def make_net(inv_temp, lattice_shape):
    net = tn.TensorNetwork() # initialize empty tensor network

    # make all nodes, indexed by lattice coorinates
    tensor = make_tensor(inv_temp, lattice_shape)
    nodes = { idx : net.add_node(tensor, name = str(idx))
              for idx in np.ndindex(lattice_shape) }

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for axis in range(len(lattice_shape)): # for each axis of the lattice

        # choose the axes of tensors that we will contract
        dir_fst, dir_snd = axis, axis + len(lattice_shape)

        for idx_fst in nodes: # loop over all nodes

            # identify the "next" neighboring node along this lattice axis
            idx_snd = list(idx_fst)
            idx_snd[dir_fst] = ( idx_snd[dir_fst] + 1 ) % lattice_shape[dir_fst]
            idx_snd = tuple(idx_snd)

            # connect up the nodes
            edge = (idx_fst,idx_snd)
            edges[edge] = net.connect(nodes[idx_fst][dir_fst],
                                      nodes[idx_snd][dir_snd],
                                      name = str(edge))

    return net, nodes, edges

##########################################################################################
# compute various quantities and plot them
##########################################################################################

steps = 50
max_inv_temp = 1.5
lattice_shape = (4,4)
figsize = (4,3)

inv_temps = np.linspace(0, max_inv_temp, steps)
vals_Z = np.zeros(steps)
probs = np.zeros(steps)
for jj in range(steps):
    _, nodes, _ = make_net(inv_temps[jj], lattice_shape)
    net_prob, net_norm, qubits_mem, qubits_op = quantum_contract(nodes.values())
    probs[jj] = net_prob
    vals_Z[jj] = net_norm * np.sqrt(net_prob)

# log(Z) / V
plt.figure(figsize = figsize)
plt.plot(inv_temps, np.log(vals_Z) / np.prod(lattice_shape), "k.")
plt.xlim(0, inv_temps.max())
plt.ylim(0, plt.gca().get_ylim()[1])
plt.xlabel(r"$J/T$")
plt.ylabel(r"$\log(Z)/V$")
plt.tight_layout()

# probability of "acceptance" -- finding all ancillas in |0>
plt.figure(figsize = figsize)
plt.plot(inv_temps, probs, "k.")
plt.xlim(0, inv_temps.max())
plt.ylim(0, plt.gca().get_ylim()[1])
plt.xlabel(r"$J/T$")
plt.ylabel("acceptance probability")
plt.tight_layout()

plt.show()
