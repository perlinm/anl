#!/usr/bin/env python3

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()


demo_matrix = np.loadtxt("tf_problem_demo.txt")

singular_vals = tf.linalg.svd(demo_matrix)[0].numpy()
print(np.isnan(singular_vals).any())

singular_vals = np.linalg.svd(demo_matrix)[1]
print(np.isnan(singular_vals).any())


### also try simulating the XY model with bond dimension 7
###   for inv_temps in np.arange(3,4.01,.05)

exit()

##########################################################################################
##########################################################################################
##########################################################################################

from clock_methods import vertex_tensor_XY

dimension = 2
bond_dimension = 7
field = 0
inv_temp = 3.6

mat_shape = (bond_dimension**dimension,)*2
T = vertex_tensor_XY(dimension, bond_dimension, inv_temp, field)
norm_T = tf.norm(T)
T /= norm_T

perm_0 = [ 1, 3, 0, 2 ]
perm_1 = [ 0, 2, 1, 3 ]

mats = {}
for jj, perm in enumerate([ perm_0, perm_1 ]):
    swallow_tensor = tf.transpose(T, perm)
    swallow_matrix = tf.reshape(swallow_tensor, mat_shape)
    singular_vals = tf.linalg.svd(swallow_matrix)[0].numpy()
    # singular_vals = np.linalg.svd(swallow_matrix)[1]
    matrix_norm = singular_vals.max()
    print(matrix_norm)

    mats[jj] = swallow_matrix.numpy()

print(abs((mats[1] - mats[0]).flatten()).max())
