#!/usr/bin/env python3

import os, sys, scipy.optimize
import numpy as np

from network_methods import cubic_bubbler
from potts_network import potts_network
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
quantum_backend = False

spokes = 2
sizes = range(3,11)

fig_dir = "figures/"

# set fonts and use latex packages
params = { "font.family" : "serif",
           "font.sans-serif" : "Computer Modern",
           "font.size" : font_size,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}",
                                     r"\usepackage{braket}" ] }
plt.rcParams.update(params)

# identify known critical temperatures for different numbers of spokes
def inv_temp_crit_5_eqn(beta):
    return np.exp(5/4 * beta) / np.cosh(np.sqrt(5)/4 * beta) - ( 1 + np.sqrt(5) )
crit_inv_temp_5 = scipy.optimize.fsolve(inv_temp_crit_5_eqn, 1)[0]
crit_inv_temps = { 2 : np.log(1+np.sqrt(2)) / 2,
                   3 : np.log(1+np.sqrt(3)) * 2/3,
                   4 : np.log(1+np.sqrt(2)),
                   5 : crit_inv_temp_5 }

assert( spokes in crit_inv_temps.keys() )
crit_inv_temp = crit_inv_temps[spokes]
inv_temps = np.linspace(0, max_inv_temp_val, steps) * crit_inv_temp
temp_text = r"$\beta / \beta_{\mathrm{crit}}$"

log_Z = np.zeros(steps)
log_probs = np.zeros(steps)
log_norms = np.zeros(steps)
sqr_M = np.zeros(steps)
for size in sizes:
    print(f"size: {size}")

    lattice_shape = (size,)*2
    volume = np.prod(lattice_shape)

    for jj in range(steps):
        print(f" {size} : {jj} / {steps}")
        net, nodes, _, log_net_scale \
            = potts_network(lattice_shape, spokes, inv_temps[jj], 0)
        bubbler = cubic_bubbler(lattice_shape)

        if quantum_backend:
            log_probs[jj], log_norms[jj] = quantum_contraction(nodes, bubbler)
        else:
            log_probs[jj], log_norms[jj] = classical_contraction(net, nodes, bubbler)

        log_Z[jj] = log_norms[jj] + 1/2 * log_probs[jj] + log_net_scale

        if inv_temps[jj] == 0: continue
        small_field = small_value / inv_temps[jj]
        net, nodes, _, log_net_scale \
            = potts_network(lattice_shape, spokes, inv_temps[jj], small_field)
        bubbler = cubic_bubbler(lattice_shape)

        if quantum_backend:
            log_prob, log_norm = quantum_contraction(nodes, bubbler)
        else:
            log_prob, log_norm = classical_contraction(net, nodes, bubbler)

        log_Z_small_field = log_norm + 1/2 * log_prob + log_net_scale
        sqr_M[jj] = 2 * ( log_Z_small_field - log_Z[jj] ) / small_value**2

    title_text = r"$q={}$, $L=({})$".format(spokes, r",".join(["N"]*len(lattice_shape)))

    # probability of "acceptance" -- finding all ancillas in |0>
    plt.figure("probs", figsize = figsize)
    plt.title(title_text)
    plt.plot(inv_temps / crit_inv_temp, log_probs / volume, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/crit_inv_temp))
    plt.xlabel(temp_text)
    plt.ylabel(r"$\log p/V$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # partition function
    plt.figure("log_Z", figsize = figsize)
    plt.title(title_text)
    plt.plot(inv_temps / crit_inv_temp, log_Z / volume, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/crit_inv_temp))
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
    plt.plot(mid_inv_temps / crit_inv_temp, energy / volume, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/crit_inv_temp))
    plt.xlabel(temp_text)
    plt.ylabel(r"$\Braket{E}/V$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

    # squared magnetization density
    plt.figure("mag", figsize = figsize)
    plt.title(title_text)
    plt.plot(inv_temps / crit_inv_temp, sqr_M / volume**2, ".", label = f"$N={size}$")
    plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.xlim(*tuple(inv_temps[[0,-1]]/crit_inv_temp))
    plt.xlabel(temp_text)
    plt.ylabel(r"$\Braket{S^2}/V^2$")
    plt.legend(framealpha = 1)
    plt.tight_layout()

if save_figures:
    if not os.path.isdir(fig_dir): os.mkdir(fig_dir)
    for fig_name in [ "probs", "log_Z", "energy", "mag" ]:
        plt.figure(fig_name)
        plt.savefig(fig_dir + fig_name + f"_{spokes}_{size}.pdf")

if show_figures: plt.show()
