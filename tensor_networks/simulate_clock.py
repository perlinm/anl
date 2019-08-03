#!/usr/bin/env python3

import os, sys
import numpy as np

from contraction_methods import quantum_contraction, classical_contraction
from network_methods import cubic_bubbler, checkerboard_bubbler
from clock_methods import clock_network

##########################################################################################
# compute various quantities for the clock model
##########################################################################################

### set simulation options

spokes = 3
lattice_size = 5
dimensions = 2

inv_temp_steps = 100
diff_field_steps = 5

max_inv_temp = 5
small_value = 1e-6

quantum_backend = False
use_vertex = True
use_XY = False
print_steps = True

### collect derived info

lattice_shape = (lattice_size,)*dimensions
inv_temps = np.linspace(0, max_inv_temp, inv_temp_steps+1)
fields = np.linspace(0, diff_field_steps*small_value, diff_field_steps+1)

root_dir = os.path.dirname(sys.argv[0])
data_dir = "data"
base_dir = os.path.join(root_dir, data_dir)

text_XY = "_XY" if use_XY else ""
base_file_name = f"{{}}_bond{spokes}_N{lattice_size}_D{dimensions}{text_XY}.txt"
base_path = os.path.join(base_dir, base_file_name)

header = f"max_inv_temp: {inv_temps[-1]}\n"
header += f"max_field: {fields[-1]}"

if use_vertex:
    _bubbler = cubic_bubbler
else: # use checkerboard tensor
    _bubbler = checkerboard_bubbler

if quantum_backend:
    def _contraction_results(_, nodes, bubbler):
        return quantum_contraction(nodes, bubbler)
else:
    def _contraction_results(net, nodes, bubbler):
        return classical_contraction(net, nodes, bubbler)

##########################################################################################
# simulate!
##########################################################################################

data_shape = ( len(inv_temps), len(fields) )
log_probs = np.zeros(data_shape)
log_norms = np.zeros(data_shape)

for bb, inv_temp in enumerate(inv_temps):
    for hh, field in enumerate(fields):
        if print_steps: print(f"{bb}/{len(inv_temps)} {hh}/{len(fields)}")
        net, nodes, _, log_net_scale \
            = clock_network(lattice_shape, spokes, inv_temp, field,
                            use_vertex = use_vertex, use_XY = use_XY)
        bubbler = _bubbler(lattice_shape)
        log_probs[bb,hh], log_norms[bb,hh] = _contraction_results(net, nodes, bubbler)
        log_norms[bb,hh] += log_net_scale

if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
np.savetxt(base_path.format("log_probs"), log_probs, header = header)
np.savetxt(base_path.format("log_norms"), log_norms, header = header)
