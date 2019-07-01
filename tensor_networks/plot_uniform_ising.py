#!/usr/bin/env python3

import os, sys
import numpy as np

from ising_tensors import ising_network
from tensor_contraction import quantum_contraction, classical_contraction

import matplotlib.pyplot as plt

save_figures = "save" in sys.argv[1:]
show_figures = "show" in sys.argv[1:]

show_figures = show_figures or not save_figures

##########################################################################################
# compute and plot various quantities for the ising model
##########################################################################################

font_size = 16
figsize = (6,5)

steps = 101
small_value = 1e-6
max_inv_temp_val = 3
sizes = range(3,7)
quantum_backend = False

fig_dir = "figures/"

# set fonts and use latex packages
params = { "font.family" : "serif",
           "font.sans-serif" : "Computer Modern",
           "font.size" : font_size,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

inv_temp_crit = np.log(1+np.sqrt(2)) / 2
inv_temps = np.linspace(0, max_inv_temp_val, steps) * inv_temp_crit

log_Z = np.zeros(steps)
log_probs = np.zeros(steps)
log_norms = np.zeros(steps)
sqr_M = np.zeros(steps)
for size in sizes:
    print(f"size: {size}")

    lattice_shape = (size,)*2
    volume = np.prod(lattice_shape)

    for jj in range(steps):
        net, nodes, _, log_val_estimate \
            = ising_network(inv_temps[jj], 0, lattice_shape)
        if quantum_backend:
            log_probs[jj], log_norms[jj] = quantum_contraction(nodes.values())
        else:
            log_probs[jj], log_norms[jj] = classical_contraction(net, nodes.values())

        log_Z[jj] = log_norms[jj] + 1/2 * log_probs[jj] + log_val_estimate

        if inv_temps[jj] == 0: continue
        net, nodes, _, log_val_estimate \
            = ising_network(inv_temps[jj], small_value, lattice_shape)
        if quantum_backend:
            log_prob, log_norm = quantum_contraction(nodes.values())
        else:
            log_prob, log_norm = classical_contraction(net, nodes.values())

        log_Z_small_field = log_norm + 1/2 * log_prob + log_val_estimate
        sqr_M[jj] = 2 * ( log_Z_small_field - log_Z[jj] ) / small_value**2 / inv_temps[jj]**2

    temp_text = r"$\beta / \beta_{\mathrm{crit}}$"
    title_text = r"lattice size: ${}$".format(r"\times ".join(["N"]*len(lattice_shape)))

    # probability of "acceptance" -- finding all ancillas in |0>
    plt.figure("probs", figsize = figsize)
    plt.title(title_text)
    plt.semilogy(inv_temps / inv_temp_crit, np.exp(log_probs), ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.xlabel(temp_text)
    plt.ylabel("acceptance probability")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # partition function
    plt.figure("log_Z", figsize = figsize)
    plt.title(title_text)
    plt.plot(inv_temps / inv_temp_crit, log_Z / volume, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.ylim(0, plt.gca().get_ylim()[-1])
    plt.xlabel(temp_text)
    plt.ylabel(r"$\log Z/V$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # energy density
    mid_inv_temps = ( inv_temps[1:] + inv_temps[:-1] ) / 2
    energy = - ( log_Z[1:] - log_Z[:-1] ) / ( inv_temps[1:] - inv_temps[:-1] )
    plt.figure("energy", figsize = figsize)
    plt.title(title_text)
    plt.plot(mid_inv_temps / inv_temp_crit, energy / volume, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.xlabel(temp_text)
    plt.ylabel(r"$\Braket{E}/V$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # squared magnetization density
    plt.figure("mag", figsize = figsize)
    plt.title(title_text)
    plt.plot(inv_temps / inv_temp_crit, sqr_M / volume**2, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/inv_temp_crit))
    plt.xlabel(temp_text)
    plt.ylabel(r"$\Braket{S^2}/V^2$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

if save_figures:
    if not os.path.isdir(fig_dir): os.mkdir(fig_dir)
    for fig_name in [ "probs", "log_Z", "energy", "mag" ]:
        plt.figure(fig_name)
        plt.savefig(fig_dir + fig_name + f"_{size}.pdf")

if show_figures: plt.show()
