#!/usr/bin/env python3

import os
import numpy as np
np.set_printoptions(linewidth = 200)

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()
import tensornetwork as tn

import matplotlib.pyplot as plt
from tensor_contraction import quantum_contraction, classical_contraction

##########################################################################################
# methods for constructing a tensor network that represents the partition function
#   of a uniform 2-D Ising model on a periodic square lattice
# hamiltonian: H = -\sum_{<j,k>} s_j s_k - h \sum_j s_j
##########################################################################################

# compute a "bare" vertex tensor element
def bare_vertex_tensor_val(idx, field_over_temp):
    return int(np.sum(idx) in [ 0, 4 ]) * np.exp(field_over_temp * (2*idx[0]-1))

# construct the entire "bare" vertex tensor
def bare_vertex_tensor(field_over_temp):
    vertex_shape = (2,) * 4
    tensor_vals = [ bare_vertex_tensor_val(idx, field_over_temp)
                    for idx in np.ndindex(vertex_shape) ]
    return tf.reshape(tf.constant(tensor_vals), vertex_shape)

# compute a link tensor element
def link_tensor_val(idx, inv_temp):
    s_idx = 2 * np.array(idx) - 1
    return np.exp(inv_temp * np.prod(s_idx))

# construct the entire link tensor
def link_tensor(inv_temp):
    link_shape = (2,) * 2
    tensor_vals = [ link_tensor_val(idx, inv_temp)
                    for idx in np.ndindex(link_shape) ]
    return tf.reshape(tf.constant(tensor_vals), link_shape)

# construct the "full" vertex tensor by contracting the square root of the edge tensor
#   at each leg of the "bare" vertex tensor
def vertex_tensor(inv_temp, field):
    sqrt_T_L = tf.linalg.sqrtm(link_tensor(inv_temp))
    T_V = bare_vertex_tensor(field * inv_temp)
    for axis in range(len(T_V.shape)):
        T_V = tf.tensordot(sqrt_T_L, T_V, axes = [ [1], [axis] ])
    return T_V

# construct tensor network on a square lattice
def make_net(inv_temp, field, lattice_shape):
    net = tn.TensorNetwork() # initialize empty tensor network

    # make all nodes, indexed by lattice coorinates
    tensor = vertex_tensor(inv_temp, field)
    nodes = { idx : net.add_node(tensor, name = str(idx))
              for idx in np.ndindex(lattice_shape) }

    # make all edges, indexed by pairs of lattice coordinates
    edges = {}
    for base_idx in np.ndindex(lattice_shape): # for vertex index
        for base_axis in range(2): # for each axis in one of two directions
            trgt_axis = base_axis + 2 # identify axis in opposite direction

            # identify the position of the neighboring vertex in this direction
            trgt_idx = list(base_idx)
            trgt_idx[base_axis] = ( trgt_idx[base_axis] + 1 ) % lattice_shape[base_axis]
            trgt_idx = tuple(trgt_idx)

            # connect up the neighboring nodes (tensors)
            edge = (base_idx, trgt_idx)
            edges[edge] = net.connect(nodes[base_idx][base_axis],
                                      nodes[trgt_idx][trgt_axis],
                                      name = str(edge))

    return net, nodes, edges

##########################################################################################
# compute various quantities and plot them
##########################################################################################

font_size = 16
figsize = (6,5)

# set fonts and use latex packages
params = { "font.family" : "serif",
           "font.sans-serif" : "Computer Modern",
           "font.size" : font_size,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

steps = 101
small_value = 1e-6
max_inv_temp_val = 3
sizes = range(3,7)

inv_temp_crit = np.log(1+np.sqrt(2)) / 2
inv_temps = np.linspace(0, max_inv_temp_val, steps) * inv_temp_crit

log_Z = np.zeros(steps)
probs = np.zeros(steps)
log_norms = np.zeros(steps)
sqr_M = np.zeros(steps)
for size in sizes:
    lattice_shape = (size, size)
    volume = np.prod(lattice_shape)

    for jj in range(steps):
        net, nodes, _ = make_net(inv_temps[jj], 0, lattice_shape)
        probs[jj], log_norms[jj], max_qubits = classical_contraction(net, nodes.values())
        # probs[jj], log_norms[jj], max_qubits = quantum_contraction(nodes.values())
        log_Z[jj] = log_norms[jj] + 1/2 * np.log(probs[jj])

        if inv_temps[jj] == 0: continue

        net, nodes, _ = make_net(inv_temps[jj], small_value, lattice_shape)
        prob, log_norm, _ = classical_contraction(net, nodes.values())
        log_Z_small_field = log_norm + 1/2 * np.log(prob)
        sqr_M[jj] = 2 * ( log_Z_small_field - log_Z[jj] ) / small_value**2 / inv_temps[jj]**2

    print(f"size, qubits: {size}, {max_qubits}")

    # probability of "acceptance" -- finding all ancillas in |0>
    plt.figure("prob", figsize = figsize)
    plt.title(r"lattice size: $N\times N$")
    plt.semilogy(inv_temps / inv_temp_crit, probs, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.xlabel(r"$\beta / \beta_{\mathrm{crit}}$")
    plt.ylabel("acceptance probability")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # partition function
    plt.figure("log_Z", figsize = figsize)
    plt.title(r"lattice size: $N\times N$")
    plt.plot(inv_temps / inv_temp_crit, log_Z / volume, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.ylim(0, plt.gca().get_ylim()[-1])
    plt.xlabel(r"$\beta / \beta_{\mathrm{crit}}$")
    plt.ylabel(r"$\log Z/V$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # energy density
    mid_inv_temps = ( inv_temps[1:] + inv_temps[:-1] ) / 2
    energy = - ( log_Z[1:] - log_Z[:-1] ) / ( inv_temps[1:] - inv_temps[:-1] )
    plt.figure("energy", figsize = figsize)
    plt.title(r"lattice size: $N\times N$")
    plt.plot(mid_inv_temps / inv_temp_crit, energy / volume, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.xlabel(r"$\beta / \beta_{\mathrm{crit}}$")
    plt.ylabel(r"$\Braket{E}/V$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # squared magnetization density
    plt.figure("mag", figsize = figsize)
    plt.title(r"lattice size: $N\times N$")
    plt.plot(inv_temps / inv_temp_crit, sqr_M / volume**2, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.xlabel(r"$\beta / \beta_{\mathrm{crit}}$")
    plt.ylabel(r"$\Braket{S^2}/V^2$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

plt.show()
