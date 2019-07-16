#!/usr/bin/env python3

import numpy as np
from itertools import product as set_product

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

def _tf_sparse_add(self, other):
    if other == 0: return self
    return tf.sparse.add(self, other)
tf.SparseTensor.__add__ = _tf_sparse_add

def _tf_sparse_radd(self, other):
    return self + other
tf.SparseTensor.__radd__ = _tf_sparse_radd

def _tf_sparse_neg(self):
    return -1 * self
tf.SparseTensor.__neg__ = _tf_sparse_neg

def _tf_sparse_sub(self, other):
    return self + ( -1 * other )
tf.SparseTensor.__sub__ = _tf_sparse_sub

def _tf_sparse_mul(self, num):
    if num == 1: return self
    return tf.SparseTensor(self.indices, num * self.values, self.dense_shape)
tf.SparseTensor.__mul__ = _tf_sparse_mul

def _tf_sparse_rmul(self, num):
    return self * num
tf.SparseTensor.__rmul__ = _tf_sparse_rmul

def _tf_sparse_truediv(self, other):
    return 1/other * self
tf.SparseTensor.__truediv__ = _tf_sparse_truediv

def tf_outer_product(tensor_a, tensor_b):
    if type(tensor_a) is not tf.SparseTensor:
        return tf.tensordot(tensor_a, tensor_b, axes = 0)
    else:
        index_iterator = set_product(tensor_a.indices.numpy(), tensor_b.indices.numpy())
        indices = [ np.concatenate(index_pair) for index_pair in index_iterator ]

        values_a = tensor_a.values.numpy()
        values_b = tensor_b.values.numpy()
        values_shape = ( len(tensor_a.values), len(tensor_b.values) )
        values = np.empty(values_shape, dtype = values_a.dtype)
        values = np.outer(values_a, values_b, values).flatten()

        dense_shape = tf.concat([ tensor_a.dense_shape, tensor_b.dense_shape ], 0)
        return tf.SparseTensor(indices, values, dense_shape)

def tf_transpose(tensor, permutation):
    if type(tensor) is not tf.SparseTensor:
        return tf.transpose(tensor, permutation)
    else:
        return tf.sparse.transpose(tensor, permutation)
