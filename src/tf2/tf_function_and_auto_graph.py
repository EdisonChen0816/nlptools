# encoding=utf-8
from tensorflow import keras
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)


# tf.function and autograph
def scaled_elu(z, scale=1.0, alpha=1.0):
    # z >= 0 ? scale * z : scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))


print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., -2.5])))


scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))

print(scaled_elu_tf.python_function is scaled_elu)


# 1 + 1/2 + 1/2^2 + ... + 1/2^n
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total


print(converge_to_2(20))


def display_tf_code(func):
    code = tf.autograph.to_code(func)
    return code


print(display_tf_code(scaled_elu))


var = tf.Variable(0.)
@tf.function
def add_21():
    return var.assign_add(21)


print(add_21())


@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)


try:
    print(cube(tf.constant([1., 2., 3.])))
except ValueError as ex:
    print(ex)
print(cube(tf.constant([1, 2, 3])))


# @tf.function py func -> graph
# get_concrete_function -> add input signature -> SavedModel

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32))
print(cube_func_int32)
print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1, 2, 3])))
print(cube_func_int32.graph)
print(cube_func_int32.graph.get_operations())
pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)
print(list(pow_op.inputs))
print(list(pow_op.outputs))
print(cube_func_int32.graph.get_operation_by_name('x'))
print(cube_func_int32.graph.get_tensor_by_name('x:0'))
print(cube_func_int32.graph.as_graph_def())