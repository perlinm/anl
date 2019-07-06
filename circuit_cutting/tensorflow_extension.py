#!/usr/bin/env python3

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

def tf_sparse_add(self, other):
    if other == 0: return self
    return tf.sparse.add(self, other)
tf.SparseTensor.__add__ = tf_sparse_add

def tf_sparse_radd(self, other):
    return self + other
tf.SparseTensor.__radd__ = tf_sparse_radd

def tf_sparse_mul(self, num):
    if num == 1: return self
    return tf.SparseTensor(self.indices, num * self.values, self.dense_shape)
tf.SparseTensor.__mul__ = tf_sparse_mul

def tf_sparse_rmul(self, num):
    return self * num
tf.SparseTensor.__rmul__ = tf_sparse_rmul

def tf_sparse_neg(self):
    return -1 * self
tf.SparseTensor.__neg__ = tf_sparse_neg

def tf_sparse_sub(self, other):
    return self + ( -1 * other )
tf.SparseTensor.__sub__ = tf_sparse_sub

def tf_outer_product(tensor_a, tensor_b):
    if type(tensor_a) is not tf.SparseTensor:
        return tf.tensordot(tensor_a, tensor_b, axes = 0)
    else:
        indices = [ tf.concat([ idx_a, idx_b ], 0)
                    for idx_a in tensor_a.indices
                    for idx_b in tensor_b.indices ]
        values =  [ val_a * val_b
                    for val_a in tensor_a.values
                    for val_b in tensor_b.values ]
        dense_shape = tf.concat([ tensor_a.dense_shape, tensor_b.dense_shape ], 0)
        return tf.SparseTensor(indices, values, dense_shape)
