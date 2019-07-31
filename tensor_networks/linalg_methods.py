#!/usr/bin/env python3

import numpy as np

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from functools import reduce

# outer product of two tensors
def tf_outer_product(tensor_a, tensor_b):
        return tf.tensordot(tensor_a, tensor_b, axes = 0)

# return the power-fold tensor power of a tensor
def tensor_power(tensor, power):
    return reduce(tf_outer_product, [tensor]*power)

# convert an isometry T : A --> B into a unitary operator U : A_B --> A_B
# where the dimensions |A_B| = |B| and A_B = A \otimes ancillas
def to_unitary(isometry):
    assert(len(isometry.shape) == 2)
    trgt_dim, base_dim = isometry.shape # dimensions of base and target spaces A and B
    if trgt_dim == base_dim: return isometry

    # identity operator on the target space B, and projector onto T(A)
    identity = tf.eye(trgt_dim, dtype = isometry.dtype)
    trgt_projector = tf.linalg.matmul(isometry, isometry, adjoint_b = True)

    # isometry V : A --> A_B defined by V(a) = a \otimes ancillas,
    # and a projector onto V(A)
    base_vecs = tf.eye(*isometry.shape, dtype = isometry.dtype)
    base_projector = tf.linalg.matmul(base_vecs, base_vecs, transpose_b = True)

    # construct a minimal unitary rotating V(A) into *T(A),
    # where * denotes an embedding of B into A_B
    trgt_reflector = identity - 2 * trgt_projector
    base_reflector = identity - 2 * base_projector
    minimal_unitary = tf.linalg.sqrtm(tf.linalg.matmul(trgt_reflector, base_reflector))

    # the minimal unitary U* above only gives us U : A_B --> A_B
    #   up to some rotation R of the standard basis on A; identify the basis rotation R
    base_rot = tf.matmul(minimal_unitary, isometry, adjoint_a = True)[:base_dim,:base_dim]

    # lift the rotation R in A to a rotation R* in A_B via R* = V \circ R
    ancilla_dim = trgt_dim // base_dim
    ancilla_identity = tf.eye(ancilla_dim, dtype = isometry.dtype)
    trgt_rot = tf.tensordot(ancilla_identity, base_rot, axes = 0) # <-- this is R*

    # rearrange the indices in R* properly
    trgt_rot = tf.transpose(trgt_rot, [ 0, 2, 1, 3 ])
    trgt_rot = tf.reshape(trgt_rot, (trgt_dim,)*2)

    # return U = U* R*
    return tf.matmul(minimal_unitary, trgt_rot)

# perform higher-order singular value decomposition
def hosvd(tensor, singular_value_ratio_cutoff = 1e-10):
    axis_num = len(tensor.shape)
    total_dim = np.prod(tensor.shape, dtype = int)
    def _axis_pull_perm_shape(axis):
        other_axes = [ jj for jj in range(axis_num) if jj != axis ]
        permutation = [ axis ] + other_axes
        pull_shape = ( tensor.shape[axis], total_dim // tensor.shape[axis] )
        return permutation, pull_shape

    factor_mats = []
    core_tensor = tensor
    for axis in range(axis_num):
        perm, shape = _axis_pull_perm_shape(axis)
        mat_factor = tf.reshape(tf.transpose(tensor, perm), shape)
        vals_D, mat_U, _ = tf.linalg.svd(mat_factor)

        vals_ratios = vals_D[1:] / vals_D[:-1]
        vals_to_keep = 1 + np.sum(abs(vals_ratios) > singular_value_ratio_cutoff)

        mat_V = mat_U[:,:vals_to_keep]
        mat_V_dag = tf.transpose(mat_V, conjugate = True)

        factor_mats.append(mat_V)
        core_tensor = tf.tensordot(mat_V_dag, core_tensor, axes = [ [1], [axis] ])

    factor_mats = tuple(factor_mats[::-1])
    core_tensor = tf.transpose(core_tensor, list(reversed(range(axis_num))))

    return core_tensor, factor_mats
