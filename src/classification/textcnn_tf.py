# encoding=utf-8
import tensorflow as tf
import jieba
import numpy as np
import random


'''
word2vec + textcnn
'''


class TextCNN:

    def __init__(self, data_path, max_len, w2v, filters, kernel_size, pool_size, strides, loss, rate, epoch, batch_size, model_path, summary_path, tf_config):
        self.data_path = data_path
        self.max_len = max_len
        self.w2v = w2v
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.loss = loss
        self.rate = rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_path = model_path
        self.summary_path = summary_path
        self.tf_config = tf_config

    def get_input_feature(self):
        data = []
        label_count = set()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                embedding = []
                text, tag = line.replace('\n', '').split('\t')
                label_count.add(tag)
                for word in jieba.lcut(text):
                    if word in self.w2v:
                        embedding.append(self.w2v[word])
                if len(embedding) < self.max_len:
                    for i in range(self.max_len - len(embedding)):
                        embedding.append([0] * 300)
                else:
                    embedding = embedding[: self.max_len]

                if 'Meeting' == tag:
                    data.append([embedding, [0]])
                else:
                    data.append([embedding, [1]])
        self.num_class = len(label_count)
        return np.asarray(data)

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

    def conv_net(self, x, dropout):
        conv1 = tf.layers.conv2d(x, self.filters, kernel_size=(self.kernel_size, 300), strides=self.strides, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(self.pool_size, 1), strides=self.strides)
        fc1 = tf.layers.flatten(pool1, name="fc1")
        fc2 = tf.layers.dense(fc1, 128)
        fc3 = tf.layers.dropout(fc2, rate=dropout)
        out = tf.layers.dense(fc3, self.num_class)
        return out

    def train(self):
        data = self.get_input_feature()
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        x = tf.placeholder(shape=[None, self.max_len, 300], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y')
        x_input = tf.reshape(x, shape=[-1, self.max_len, 300, 1])
        drop_rate = tf.placeholder(dtype=tf.float32, name='drop_rate')
        logits = self.conv_net(x_input, drop_rate)
        tf.add_to_collection("logits", logits)
        cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(cross_loss)
        if 'adam' == self.loss.lower():
            optim = tf.train.AdamOptimizer(self.rate).minimize(loss)
        elif 'sgd' == self.loss.lower():
            optim = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        else:
            optim = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for step, (seqs, labels) in enumerate(self.batch_yield(data, self.batch_size)):
                    _, curr_loss = sess.run([optim, loss], feed_dict={x: seqs, y: labels, drop_rate: 0})
                    if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                        print("epoch:%d, batch: %d, current loss: %f" % (i, step+1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)

    def _predict_text_process(self, text):
        embedding = []
        for word in jieba.lcut(text):
            if word in self.w2v:
                embedding.append(self.w2v[word])
        if len(embedding) < self.max_len:
            for i in range(self.max_len - len(embedding)):
                embedding.append([0] * 300)
        else:
            embedding = embedding[: self.max_len]
        return np.asarray([embedding]), np.asarray([[-1]])

    def predict(self, texts):
        predict_result = []
        with tf.Session(config=tf_config) as sess:
            saver = tf.train.import_meta_graph(self.model_path + '/tf2model.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name('x:0')
            y = graph.get_tensor_by_name('y:0')
            drop_rate = graph.get_tensor_by_name('drop_rate:0')
            logits = tf.get_collection('logits')
            for text in texts:
                seqs, label = self._predict_text_process(text)
                pred = sess.run([logits], feed_dict={x: seqs, y: label, drop_rate: 0})
                predict_result.append(np.argmax(pred[0][0], 1).tolist())
        return predict_result


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    data_path = '../../data/textcnn/data'
    max_len = 20
    w2v = KeyedVectors.load('../../tf2model/w2v/w2v.tf2model')
    filters = 16
    kernel_size = 3
    pool_size = 3
    strides = 1
    loss = 'sgd'
    rate = 0.001
    epoch = 200
    batch = 32
    model_path = '../../model/textcnn'
    summary_path = '../../model/textcnn/summary'
    textcnn = TextCNN(data_path, max_len, w2v, filters, kernel_size, pool_size,
                      strides, loss, rate, epoch, batch, model_path, summary_path, tf_config)
    textcnn.train()
    print(textcnn.predict(['请年假扣不扣钱', '会上都有谁', '下午三点开会']))