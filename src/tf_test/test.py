# encoding=utf-8
import tensorflow as tf
import numpy as np

# message = tf.constant('hello world')
# with tf.Session() as sess:
#     print(sess.run(message).decode())

# t1 = tf.constant([1, 2, 3, 4])
# t2 = tf.constant([5, 6, 7, 8])
# with tf.Session() as sess:
#     print(sess.run(t1 + t2))


# bais = tf.Variable(tf.zeros([100, 100]))
# with tf.Session() as sess:
#     sess.run(bais.initializer)
#     print(sess.run(bais))

# x_data = np.random.rand(100).astype(np.float32)
# y_data = 0.3 * x_data + 0.1
#
# weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# bais = tf.Variable(tf.zeros([1]))
#
# y_prediction = weight * x_data + bais
#
# loss = tf.reduce_mean(tf.square(y_data-y_prediction))
# optm = tf.train.GradientDescentOptimizer(0.5)
#
# train = optm.minimize(loss)
# init = tf.global_variables_initializer()
# tf.summary.scalar('loss', loss)
# smy = tf.summary.merge_all()
#
# with tf.Session() as sess:
#     sess.run(init)
#     writer = tf.summary.FileWriter("C:/logs", sess.graph)
#     for step in range(200):
#         sess.run(train)
#         writer.add_summary(summary=sess.run(smy), global_step=step)


t1 = tf.Variable([[1, 2, 3], [4, 5, 6]])
t2 = tf.Variable([[7, 8, 9]])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(t1 + t2))