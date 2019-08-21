#!/usr/bin/env python3

import os

dat_dir = "data"
fig_dir = "figures"

def dat_name_builder(base_dir, bond_dimension, lattice_size, dimensions,
                     network_type, test_run):
    test_text = "_test" if test_run else ""
    def _build_method(data_tag):
        base_name = f"{data_tag}_bond{bond_dimension}_N{lattice_size}" \
                  + f"_D{dimensions}_{network_type}{test_text}.txt"
        return os.path.join(base_dir, base_name)
    return _build_method

def fig_name_builder(base_dir, dimensions, network_type, test_run):
    test_text = "_test" if test_run else ""
    def _build_method(data_tag):
        base_name = f"{data_tag}_D{dimensions}_{network_type}{test_text}.pdf"
        return os.path.join(base_dir, base_name)
    return _build_method
