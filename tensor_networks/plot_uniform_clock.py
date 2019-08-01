#!/usr/bin/env python3

import os, sys, scipy.optimize
import numpy as np

from network_methods import cubic_bubbler, checkerboard_bubbler
from clock_methods import clock_network
from contraction_methods import quantum_contraction, classical_contraction

import matplotlib.pyplot as plt

save_figures = "save" in sys.argv[1:]
show_figures = "show" in sys.argv[1:]

show_figures = show_figures or not save_figures

print_steps = True

##########################################################################################
# compute and plot various quantities for the ising model
##########################################################################################

font_size = 16
figsize = (6,5)

steps = 101
small_value = 1e-6
max_inv_temp_val = 3
quantum_backend = False
use_vertex = True

spokes = 2
sizes = range(3,11)
dims = 2 # dimension of lattice

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

if spokes in crit_inv_temps.keys():
    crit_inv_temp = crit_inv_temps[spokes]
    temp_text = r"$\beta / \beta_{{\mathrm{{crit}}}}^{{({})}}$".format(spokes)
else:
    crit_inv_temp = 1
    temp_text = r"$\beta$"

title_text = r"$q={}$, $L=({})$".format(spokes, r",".join(["N"]*dims))
inv_temps = np.linspace(0, max_inv_temp_val, steps) * crit_inv_temp

log_Z = np.zeros(steps)
log_probs = np.zeros(steps)
log_norms = np.zeros(steps)
sqr_M = np.zeros(steps)
for size in sizes:
    print(f"size: {size}")

    lattice_shape = (size,)*dims
    volume = np.prod(lattice_shape)

    for jj in range(steps):
        if print_steps: print(f" {size} : {jj} / {steps}")
        net, nodes, _, log_net_scale \
            = clock_network(lattice_shape, spokes, inv_temps[jj], use_vertex = use_vertex)
        if use_vertex:
            bubbler = cubic_bubbler(lattice_shape)
        else:
            bubbler = checkerboard_bubbler(lattice_shape)

        if quantum_backend:
            log_probs[jj], log_norms[jj] = quantum_contraction(nodes, bubbler)
        else:
            log_probs[jj], log_norms[jj] = classical_contraction(net, nodes, bubbler)

        log_norms[jj] += log_net_scale
        log_Z[jj] = log_norms[jj] + 1/2 * log_probs[jj]

        if inv_temps[jj] == 0: continue
        small_field = small_value / inv_temps[jj]
        net, nodes, _, log_net_scale \
            = clock_network(lattice_shape, spokes, inv_temps[jj], small_field,
                            use_vertex = use_vertex)

        if quantum_backend:
            log_prob, log_norm = quantum_contraction(nodes, bubbler)
        else:
            log_prob, log_norm = classical_contraction(net, nodes, bubbler)

        log_norm += log_net_scale
        log_Z_small_field = log_norm + 1/2 * log_prob
        sqr_M[jj] = 2 * ( log_Z_small_field - log_Z[jj] ) / small_value**2

    # partition function
    plt.figure("log_Z")
    plt.plot(inv_temps / crit_inv_temp, log_Z / volume, ".", label = f"$N={size}$")
    plt.ylabel(r"$\log Z/V$")
    plt.ylim(0, plt.gca().get_ylim()[-1])

    def _diff(vals): return vals[1:] - vals[:-1]

    # energy density
    mid_inv_temps = ( inv_temps[1:] + inv_temps[:-1] ) / 2
    energy = - _diff(log_Z) / _diff(inv_temps) # d \log Z / d \beta
    plt.figure("energy", figsize = figsize)
    plt.plot(mid_inv_temps / crit_inv_temp, energy / volume, ".", label = f"$N={size}$")
    plt.ylabel(r"$\Braket{E}/V$")

    # heat_capacity
    mid_mid_inv_temps = ( mid_inv_temps[1:] + mid_inv_temps[:-1] ) / 2
    heat_capacity = _diff(energy) / _diff(1/mid_inv_temps) # d E / d T
    plt.figure("heat_capacity", figsize = figsize)
    plt.plot(mid_mid_inv_temps / crit_inv_temp, heat_capacity / volume, ".", label = f"$N={size}$")
    plt.ylabel(r"$C_V/V$")

    # tensor network norms
    plt.figure("norms")
    plt.plot(inv_temps / crit_inv_temp, log_norms / volume, ".", label = f"$N={size}$")
    plt.ylabel(r"$\log\mathcal{D}/V$")
    plt.ylim(0, plt.gca().get_ylim()[-1])

    # probability of "acceptance" -- i.e. of finding all ancillas in |0>
    plt.figure("probs")
    plt.plot(inv_temps / crit_inv_temp, log_probs / volume, ".", label = f"$N={size}$")
    plt.ylabel(r"$\log p/V$")

    # squared magnetization density
    plt.figure("mag")
    plt.plot(inv_temps / crit_inv_temp, sqr_M / volume**2, ".", label = f"$N={size}$")
    plt.ylabel(r"$\Braket{S^2}/V^2$")

# set miscellaneous figure properties
for fig_name in plt.get_figlabels():
    fig = plt.figure(fig_name)
    fig.set_size_inches(figsize)
    plt.title(title_text)
    plt.xlim(*tuple(inv_temps[[0,-1]]/crit_inv_temp))
    plt.xlabel(temp_text)
    if spokes in crit_inv_temps.keys():
        plt.axvline(1, color = "gray", linestyle = "--", linewidth = 1)
    plt.legend(framealpha = 1)
    plt.tight_layout()

if save_figures:
    if not os.path.isdir(fig_dir): os.mkdir(fig_dir)
    for fig_name in plt.get_figlabels():
        plt.figure(fig_name)
        plt.savefig(fig_dir + fig_name + f"_{spokes}_{size}.pdf")

if show_figures: plt.show()
