# encoding=utf-8
import tensorflow as tf
import numpy as np
import random
import jieba

'''
w2v + lstm
'''


class BiLstm:

    def __init__(self, data_path, max_len, w2v, num_units, loss, rate, epoch, batch_size, model_path, summary_path, tf_config):
        self.data_path = data_path
        self.max_len = max_len
        self.w2v = w2v
        self.num_units = num_units
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

    def bi_lstm_net(self, inputs, keep_prob):
        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_units)
        lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_fw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_units)
        lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_bw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        output, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_fw,
            cell_bw=lstm_cell_bw,
            inputs=inputs,
            dtype=tf.float32
        )
        h_state = output[0][:, -1, :]
        W = tf.get_variable("W", [self.num_units, self.num_class],
                            initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
        bias = tf.get_variable("bias", [self.num_class],
                               initializer=tf.zeros_initializer(), dtype=tf.float32)
        y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
        return y_pre

    def train(self):
        data = self.get_input_feature()
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        x = tf.placeholder(shape=[None, self.max_len, 300], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y')
        x_input = tf.reshape(x, shape=[-1, self.max_len, 300])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        logits = self.bi_lstm_net(x_input, keep_prob)
        tf.add_to_collection("logits", logits)
        cross_entropy = -tf.reduce_mean(y * tf.log(logits))
        if 'adam' == self.loss.lower():
            optim = tf.train.AdamOptimizer(self.rate).minimize(cross_entropy)
        elif 'sgd' == self.loss.lower():
            optim = tf.train.GradientDescentOptimizer(self.rate).minimize(cross_entropy)
        else:
            optim = tf.train.GradientDescentOptimizer(self.rate).minimize(cross_entropy)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for step, (seqs, labels) in enumerate(self.batch_yield(data, self.batch_size)):
                    _, curr_loss = sess.run([optim, cross_entropy], feed_dict={x: seqs, y: labels, keep_prob: 0.9})
                    if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                        print("epoch:%d, batch: %d, current loss: %f" % (i, step + 1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)

    def predict_text_process(self, text):
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
            saver = tf.train.import_meta_graph(self.model_path + '/model.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name('x:0')
            y = graph.get_tensor_by_name('y:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            logits = tf.get_collection('logits')
            for text in texts:
                seqs, label = self.predict_text_process(text)
                pred = sess.run([logits], feed_dict={x: seqs, y: label, keep_prob: 1.0})
                predict_result.append(np.argmax(pred[0][0], 1).tolist())
        return predict_result


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    bilstm_cfg = {
        'data_path': '../../data/textcnn/data',
        'max_len': 5,
        'w2v': KeyedVectors.load('../../model/w2v/w2v.model'),
        'num_units': 64,
        'loss': 'sgd',
        'rate': 0.001,
        'epoch': 50,
        'batch_size': 8192,
        'model_path': '../../model/bilstm',
        'summary_path': '../../model/bilstm/summary',
        'tf_config': tf_config
    }
    bilstm = BiLstm(**bilstm_cfg)
    bilstm.train()
    print(bilstm.predict(['请年假扣不扣钱', '会上都有谁', '下午三点开会']))