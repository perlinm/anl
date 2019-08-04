#!/usr/bin/env python3

import os

dat_dir = "data"
fig_dir = "figures"

def name_builder(base_dir, bond_dimension, lattice_size, dimensions, use_XY):
    text_XY = "_XY" if use_XY else ""
    def _build_method(data_tag):
        base_name = f"{data_tag}_bond{bond_dimension}_N{lattice_size}_D{dimensions}{text_XY}.txt"
        return os.path.join(base_dir, base_name)
    return _build_method
