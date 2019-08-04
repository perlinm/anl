#!/usr/bin/env python3

import os, sys
import numpy as np

from contraction_methods import quantum_contraction, classical_contraction
from network_methods import cubic_bubbler, checkerboard_bubbler
from clock_methods import clock_network

from clock_formats import dat_dir, name_builder

root_dir = os.path.dirname(sys.argv[0])

##########################################################################################
# compute various quantities for the clock model
##########################################################################################

### set simulation options

spokes = 3
lattice_size = 5
dimensions = 2
use_XY = False

max_inv_temp = 3
small_value = 1e-6

inv_temp_steps = 10
diff_field_steps = 2

quantum_backend = False
use_vertex = True
print_steps = True

##########################################################################################
# collect derived info
##########################################################################################

max_field_val = diff_field_steps * small_value
lattice_shape = (lattice_size,)*dimensions
inv_temps = np.linspace(0, max_inv_temp, inv_temp_steps+1)
field_vals = np.linspace(0, max_field_val, diff_field_steps+1)

header = f"max_inv_temp: {max_inv_temp}\n"
header += f"max_field_val: {max_field_val}"
base_dir = os.path.join(root_dir, dat_dir)
dat_file_name = name_builder(base_dir, spokes, lattice_size, dimensions, use_XY)

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

data_shape = ( len(inv_temps), len(field_vals) )
log_probs = np.zeros(data_shape)
log_norms = np.zeros(data_shape)

for bb, inv_temp in enumerate(inv_temps):
    for hh, field_val in enumerate(field_vals):
        if print_steps: print(f"{bb}/{len(inv_temps)} {hh}/{len(field_vals)}")
        field = field_val / inv_temp if inv_temp != 0 else 0
        net, nodes, _, log_net_scale \
            = clock_network(lattice_shape, spokes, inv_temp, field,
                            use_vertex = use_vertex, use_XY = use_XY)
        bubbler = _bubbler(lattice_shape)
        log_probs[bb,hh], log_norms[bb,hh] = _contraction_results(net, nodes, bubbler)
        log_norms[bb,hh] += log_net_scale

if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
np.savetxt(dat_file_name("log_probs"), log_probs, header = header)
np.savetxt(dat_file_name("log_norms"), log_norms, header = header)
