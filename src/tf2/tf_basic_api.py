# encoindg=utf-8
import numpy as np
import tensorflow as tf


print(tf.__name__, tf.__version__)


t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t)
print(t[:, 1:])
print(t[..., 1])


# ops
print(t + 10)
print(tf.square(t))
print(t @ tf.transpose(t))


# numpy conversion
print(t.numpy())
print(np.square((t)))
np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(np_t))


# Scalars
t = tf.constant(2.718)
print(t.numpy())
print(t.shape)


# string
t = tf.constant('cafe')
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit='UTF8_CHAR'))
print(tf.strings.unicode_decode(t, 'UTF8'))


# string array
t = tf.constant(['cafe', 'coffee', '咖啡'])
print(tf.strings.length(t, unit='UTF8_CHAR'))
r = tf.strings.unicode_decode(t, 'UTF8')
print(r)


# ragged tensor
r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
# index op
print(r)
print(r[1])
print(r[1:2])


# opa on ragged tensor
r2 = tf.ragged.constant([[51, 52], [], [71]])
print(tf.concat([r, r2], axis=0))
r3 = tf.ragged.constant([[13, 14], [15], [], [42, 43]])
print(tf.concat([r, r3], axis=1))
print(r.to_tensor())


# sparse tensor
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], values=[1., 2., 3.], dense_shape=[3, 4])
print(s)
print(tf.sparse.to_dense(s))


# ops on sparse tensor
s2 = s * 2.0
print(s2)
try:
    s3 = s + 1
except TypeError as ex:
    print(ex)
s4 = tf.constant([[10., 20.],
                  [30., 40.],
                  [50., 60.],
                  [70., 80.]])
print(tf.sparse.sparse_dense_matmul(s, s4))

s5 = tf.SparseTensor(indices=[[0, 2], [0, 1], [2, 3]], values=[1., 2., 3.], dense_shape=[3, 4])
print(s5)
# print(tf.sparse.to_dense(s5))
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))


# Variabies
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
print(v.value())
print(v.numpy())


# assing value
v.assign(2 * v)
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)