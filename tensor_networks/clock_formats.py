#!/usr/bin/env python3

import os

dat_dir = "data"
fig_dir = "figures"

def _bool_text(use_vertex, use_XY, test_run):
    return ( "_chkr" if not use_vertex else "" +
             "_XY" if use_XY else "" +
             "_test" if test_run else "" )

def dat_name_builder(base_dir, bond_dimension, lattice_size, dimensions,
                     use_vertex, use_XY, test_run):
    bool_text = _bool_text(use_vertex, use_XY, test_run)
    def _build_method(data_tag):
        base_name = f"{data_tag}_bond{bond_dimension}_N{lattice_size}" \
                  + f"_D{dimensions}{bool_text}.txt"
        return os.path.join(base_dir, base_name)
    return _build_method

def fig_name_builder(base_dir, dimensions, use_XY, test):
    bool_text = _bool_text(use_vertex, use_XY, test_run)
    def _build_method(data_tag):
        base_name = f"{data_tag}_D{dimensions}{bool_text}.pdf"
        return os.path.join(base_dir, base_name)
    return _build_method
