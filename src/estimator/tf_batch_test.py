# encoding=utf-8
import tensorflow as tf
import numpy as np


tf.app.flags.DEFINE_string('str_input', '123', '')

FLAGS = tf.app.flags.FLAGS


def main(_):
    print(FLAGS.str_input)


def load_data():
    data = np.array([x for x in range(1, 101)]).reshape(10, 10)
    label = np.array([x for x in range(1, 11)]).reshape(10, 1)
    batch_size = 3
    capacity = 100 + 3 * batch_size
    input_queue = tf.train.slice_input_producer([data, label])
    label = input_queue[1]
    image = input_queue[0]
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=32, capacity=capacity, min_after_dequeue=50)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(20):
            cur_example_batch, cur_label_batch = sess.run(
                [image_batch, label_batch])
            print(cur_example_batch, cur_label_batch)
            print('!!!!!!!!!!!!')
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run(
        main=load_data
    )