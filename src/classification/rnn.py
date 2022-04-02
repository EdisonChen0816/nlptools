# encoding=utf-8
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import f1_score, recall_score, precision_score


class RNN:

    def __init__(self, num_units, keep_prob, rate=0.001, epoch=5, batch_size=32):
        self.num_units = num_units
        self.keep_prob = keep_prob
        self.rate = rate
        self.epoch = epoch
        self.batch_size = batch_size

    def get_input_feature(self, mode):
        if mode == 'train':
            path = './train.txt'
        elif mode == 'eval':
            path = './test.txt'
        features = []
        with open(path) as f:
            for line in f:
                data, label = line.replace('\n', '').split('\t')
                label = label.strip()
                data = data.strip().split(',')
                feature = []
                feature.append([0.0])
                for i in range(0, len(data), 2):
                    if (int(data[i]) % 2 == 0 and int(data[i+1]) % 2 == 0) or (int(data[i]) % 2 == 1 and int(data[i+1]) % 2 == 1):
                        feature.append([0.0])
                    else:
                        feature.append([1.0])
                feature.append([0.0])
                features.append([feature, [int(label)]])
        return np.asarray(features[:36000])

    def batch_yield(self, data, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(data)
        seqs, labels = [], []
        for sent, tag in data:
            if len(seqs) == batch_size:
                yield np.asarray(seqs), np.asarray(labels)
                seqs, labels = [], []
            seqs.append(sent)
            labels.append(tag)
        if len(seqs) != 0:
            yield np.asarray(seqs), np.asarray(labels)

    def rnn_net(self, inputs):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.num_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        output, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            dtype=tf.float32
        )
        h_state = output[:, -1, :]
        W = tf.get_variable("W", [self.num_units, 2],
                            initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
        bias = tf.get_variable("bias", [2],
                               initializer=tf.zeros_initializer(), dtype=tf.float32)
        y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
        return y_pre

    def train(self):
        features = self.get_input_feature('train')
        num_batches = (len(features) + self.batch_size - 1) // self.batch_size
        x = tf.placeholder(shape=[None, 8, 1], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y')
        x_input = tf.reshape(x, shape=[-1, 8, 1])
        with tf.variable_scope('train'):
            logits = self.rnn_net(x_input)
        cross_entropy = -tf.reduce_mean(y * tf.log(logits))
        optim = tf.train.AdamOptimizer(self.rate).minimize(cross_entropy)
        # optim = tf.train.GradientDescentOptimizer(self.rate).minimize(cross_entropy)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for step, (seqs, labels) in enumerate(self.batch_yield(features, self.batch_size)):
                    _, curr_loss = sess.run([optim, cross_entropy], feed_dict={x: seqs, y: labels})
                    if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                        print("epoch: %d, batch: %d, current loss: %f" % (i, step + 1, curr_loss))
            self.eval(sess, logits, x, y)

    def eval(self, sess, logits, x, y):
        features = self.get_input_feature('eval')
        y_pred = []
        y_test = []
        for _, (X, Y) in enumerate(self.batch_yield(features, self.batch_size)):
            pred = sess.run(logits, feed_dict={x: X, y: Y})
            y_pred += pred.argmax(axis=1).tolist()
            y_test += Y.reshape(1, -1).tolist()[0]
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print('recall_score:' + str(recall), 'precision_score:' + str(precision), 'f1_score:' + str(f1))


if __name__ == '__main__':
    rnn_cfg = {
        'num_units': 256,
        'keep_prob': 1.0,
        'rate': 0.001,
        'epoch': 5,
        'batch_size': 32
    }
    rnn = RNN(**rnn_cfg)
    rnn.train()
