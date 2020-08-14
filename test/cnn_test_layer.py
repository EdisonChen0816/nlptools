# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
tf.layer
'''

mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
drop_rate = tf.placeholder(dtype=tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])


def conv_net(x_dict, n_classes, dropout):
    conv1 = tf.layers.conv2d(x_dict, 32, 5, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    fc1 = tf.layers.flatten(pool2, name="fc1")
    fc2 = tf.layers.dense(fc1, 1024)
    fc2 = tf.layers.dropout(fc2, rate=dropout)
    out = tf.layers.dense(fc2, n_classes)
    return out


logits = conv_net(x_image, 10, drop_rate)
cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(cross_loss)
step = tf.train.AdamOptimizer(0.001).minimize(loss)

# accuracy
acc_mat = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
acc = tf.reduce_sum(tf.cast(acc_mat, tf.float32))
prediction = tf.argmax(logits, axis=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        print(batch_xs.shape, batch_ys.shape)
        _, curr_loss = sess.run([step, loss], feed_dict={x: batch_xs, y: batch_ys, drop_rate: 0.5})
        if (i + 1) % 10 == 0:
            conv_y, curr_acc = sess.run([logits, acc], feed_dict={x: mnist.test.images, y: mnist.test.labels, drop_rate: 0.0})
            print("current loss: %f, current test Accuracy : %f" % (curr_loss, curr_acc))