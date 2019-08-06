#!/usr/bin/env python3

import os

dat_dir = "data"
fig_dir = "figures"

def _bool_text(use_vertex, use_XY, test):
    text_vertex = "_chkr" if not use_vertex else ""
    text_XY = "_XY" if use_XY else ""
    text_test = "_test" if test else ""
    return test_vertex + text_XY + text_test

def dat_name_builder(base_dir, bond_dimension, lattice_size, dimensions, use_vertex, use_XY, test):
    bool_text = _bool_text(use_vertex, use_XY, test)
    def _build_method(data_tag):
        base_name = f"{data_tag}_bond{bond_dimension}_N{lattice_size}" \
                  + f"_D{dimensions}{bool_text}.txt"
        return os.path.join(base_dir, base_name)
    return _build_method

def fig_name_builder(base_dir, dimensions, use_XY, test):
    bool_text = _bool_text(use_vertex, use_XY, test)
    def _build_method(data_tag):
        base_name = f"{data_tag}_D{dimensions}{bool_text}.pdf"
        return os.path.join(base_dir, base_name)
    return _build_method
