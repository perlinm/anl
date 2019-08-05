#!/usr/bin/env python3

import os, sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from clock_formats import dat_dir, fig_dir, dat_name_builder, fig_name_builder

from itertools import product as set_product

root_dir = os.path.dirname(sys.argv[0])
save_figures = "save" in sys.argv[1:]
show_figures = "show" in sys.argv[1:]

##########################################################################################
# plot various quantities for the clock model
##########################################################################################

spoke_vals = [ 2 ]
lattice_size_vals = range(3,11)
max_plot_inv_temp = 1.5

use_XY = False
use_vertex = True
dimensions = 2

crit_refline = True
normalize_to_crit = True
if use_XY and crit_refline:
    print("WARNING: critical temperatures for XY model not provided")
    crit_refline = False

font_size = 16
figsize = (6,5)

##########################################################################################
# collect derived info
##########################################################################################

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

if len(spoke_vals) == 1 and spoke_vals[0] in crit_inv_temps.keys():
    crit_inv_temp = crit_inv_temps[spoke_vals[0]]
else:
    crit_refline = False

##########################################################################################
# plot!
##########################################################################################

def _mid(vals): return ( vals[1:] + vals[:-1] ) / 2
def _diff(vals, axis = None):
    if axis == None:
        return vals[1:] - vals[:-1]
    if axis == 1:
        return vals[1:,:] - vals[:-1,:]
    if axis == 2:
        return vals[:,1:] - vals[:,:-1]

assert(len(spoke_vals) >= 1)
assert(len(lattice_size_vals) >= 1)
make_legend = True
if len(spoke_vals) > 1 and len(lattice_size_vals) > 1:
    title_text = ""
    def _label(spokes, lattice_size):
        return f"$q={spokes}$, $N={lattice_size}$"
elif len(spoke_vals) == 1 and len(lattice_size_vals) > 1:
    title_text = f"$q={spoke_vals[0]}$"
    def _label(spokes, lattice_size):
        return f"$N={lattice_size}$"
elif len(spoke_vals) > 1 and len(lattice_size_vals) == 1:
    title_text = f"$N={lattice_size_vals[0]}$"
    def _label(spokes, lattice_size):
        return f"$q={spokes}$"
else:
    title_text = f"$q={spoke_vals[0]}$, $N={lattice_size_vals[0]}$"
    def _label(*args): return None
    make_legend = False

for spokes, lattice_size in set_product(spoke_vals, lattice_size_vals):

    volume = lattice_size**dimensions
    label = _label(spokes, lattice_size)

    dat_base_dir = os.path.join(root_dir, dat_dir)
    dat_file_name = dat_name_builder(dat_base_dir, spokes, lattice_size,
                                     dimensions, use_vertex, use_XY)

    try:
        log_probs = np.loadtxt(dat_file_name("log_probs"))
        log_norms = np.loadtxt(dat_file_name("log_norms"))
    except OSError:
        print("data files not found:")
        print(dat_file_name("log_probs"))
        print(dat_file_name("log_norms"))
        exit()
    log_Z = 1/2 * log_probs + log_norms

    with open(dat_file_name("log_probs"), "r") as dat_file:
        for line in dat_file:
            if line[0] != "#": continue
            if "max_inv_temp" in line:
                max_inv_temp = float(line.split()[-1])
            if "max_field_val" in line:
                max_field_val = float(line.split()[-1])

    inv_temps = np.linspace(0, max_inv_temp, log_probs.shape[0])
    field_vals = np.linspace(0, max_field_val, log_probs.shape[1])

    # probability of "acceptance" -- i.e. of finding all ancillas in |0>
    plt.figure("probs")
    plt.plot(inv_temps, log_probs[:,0]/volume, ".", label = label)
    plt.ylabel(r"$\log p/V$")

    # energy density: E = -dZ / d\beta
    mid_inv_temps = _mid(inv_temps)
    energy = - _diff(log_Z[:,0]) / _diff(inv_temps)
    plt.figure("energy", figsize = figsize)
    plt.plot(mid_inv_temps, energy/volume, ".", label = label)
    plt.ylabel(r"$\Braket{E}/V$")

    # heat_capacity: C_V = -\beta^2 dE / d\beta
    mid_mid_inv_temps = _mid(mid_inv_temps)
    heat_capacity = - mid_mid_inv_temps**2 * _diff(energy) / _diff(mid_inv_temps)
    plt.figure("heat_capacity", figsize = figsize)
    plt.plot(mid_mid_inv_temps, heat_capacity/volume, ".", label = label)
    plt.ylabel(r"$C_V/V$")

    # squared magnetization density
    # M = dZ / d(\beta h); evaluated in the limit h --> 0+
    mid_field_vals = _mid(field_vals)
    mags = _diff(log_Z,2) / _diff(field_vals)
    mag_0 = 3/2 * mags[:,0] - 1/2 * mags[:,1] # \lim_{h \to 0+) m
    plt.figure("mag")
    plt.plot(inv_temps, mag_0/volume, ".", label = label)
    plt.ylabel(r"$\Braket{S}/V$")

    # squared magnetization density
    # M^2 = d^2 Z / d(\beta h)^2 / volume; evaluated at h = 0
    sqr_mag_0 = 2 * _diff(log_Z[:,:2],2) / _diff(field_vals[:2])**2
    sqr_mag_0 = sqr_mag_0.flatten()
    plt.figure("sqr_mag")
    plt.plot(inv_temps, sqr_mag_0/volume**2, ".", label = label)
    plt.ylabel(r"$\Braket{S^2}/V^2$")

# set miscellaneous figure properties
for fig_name in plt.get_figlabels():
    fig = plt.figure(fig_name)
    fig.set_size_inches(figsize)
    plt.xlabel(r"$\beta$")
    if title_text:
        plt.title(title_text)
    if max_plot_inv_temp:
        plt.xlim(0,max_plot_inv_temp)
    else:
        plt.xlim(0,plt.gca().get_xlim()[1])
    if crit_refline:
        plt.axvline(crit_inv_temp, color = "gray", linestyle = "--", linewidth = 1)
    if make_legend:
        plt.legend(framealpha = 1)
    plt.tight_layout()

if save_figures:
    fig_base_dir = os.path.join(root_dir, fig_dir)
    fig_file_name = fig_name_builder(fig_base_dir, dimensions, use_vertex, use_XY)
    if not os.path.isdir(fig_base_dir): os.mkdir(fig_base_dir)
    for fig_name in plt.get_figlabels():
        plt.figure(fig_name)
        plt.savefig(fig_file_name(fig_name))

if show_figures: plt.show()
