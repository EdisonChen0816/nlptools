# encoding=utf-8
import tensorflow as tf
import random
import numpy as np
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bilstm_crf')


class BiLstmCrf:

    def __init__(self, train_path, eval_path, max_len, batch_size, epoch, loss, rate, num_units,
                 tf_config, model_path, summary_path, embedding_dim=300, tag2label=None):
        self.train_path = train_path
        self.eval_path = eval_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss = loss
        self.rate = rate
        self.num_units = num_units
        self.tf_config = tf_config
        self.model_path = model_path
        self.summary_path = summary_path
        self.embedding_dim = embedding_dim
        if tag2label is None:
            tag2label = {
                "O": 0,
                "B": 1,
                "I": 2
            }
        self.tag2label = tag2label
        self.word2id, self.id2word = self.get_mapping()

    def get_mapping(self):
        word2id = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        id2word = {
            0: '<PAD>',
            1: '<UNK>'
        }
        count = 2
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\n' == line:
                    continue
                word, _ = line.replace('\n', '').split('\t')
                if word not in word2id:
                    word2id[word] = count
                    id2word[count] = word
                    count += 1
        return word2id, id2word

    def get_input_feature(self, data_path):
        data = []
        seq = []
        label = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\n' == line:
                    if len(label) != len(seq):
                        raise('label and seq, length is not match')
                    seq_len = len(seq)
                    if seq_len > self.max_len:
                        seq = seq[: self.max_len]
                        label = label[: self.max_len]
                    else:
                        seq += [self.word2id['<PAD>']] * (self.max_len - seq_len)
                        label += [self.tag2label['O']] * (self.max_len - seq_len)
                    data.append([seq, seq_len, label])
                    seq = []
                    label = []
                else:
                    word, tag = line.replace('\n', '').split('\t')
                    seq.append(self.word2id[word])
                    label.append(self.tag2label[tag])
        return np.asarray(data)

    def batch_yield(self, data, shuffle=False):
        if shuffle:
            random.shuffle(data)
        seqs, seq_lens, labels = [], [], []
        for (seq, seq_len, label) in data:
            if len(seqs) == self.batch_size:
                yield np.asarray(seqs), np.asarray(seq_lens), np.asarray(labels)
                seqs, seq_lens, labels = [], [], []
            seqs.append(seq)
            seq_lens.append(seq_len)
            labels.append(label)
        if len(seqs) != 0:
            yield np.asarray(seqs), np.asarray(seq_lens), np.asarray(labels)

    def model(self, seqs, seq_lens, labels):
        with tf.variable_scope('embedding_layer'):
            embedding_matrix = tf.get_variable("embedding_matrix", [len(self.word2id), self.embedding_dim], dtype=tf.float32)
            embedded = tf.nn.embedding_lookup(embedding_matrix, seqs)
        with tf.variable_scope('encoder'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_units)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_units)
            ((rnn_fw_outputs, rnn_bw_outputs),
             (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=embedded,
                sequence_length=seq_lens,
                dtype=tf.float32
            )
            rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs)
        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(rnn_outputs, len(self.tag2label))
            log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, labels, seq_lens)
            preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, seq_lens)
        return preds_seq, log_likelihood

    def fit(self):
        data = self.get_input_feature(self.train_path)
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        seqs = tf.placeholder(tf.int32, [None, None], name="seqs")
        seq_lens = tf.placeholder(tf.int32, [None], name="seq_lens")
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
        preds_seq, log_likelihood = self.model(seqs, seq_lens, labels)
        with tf.variable_scope('loss'):
            loss = -log_likelihood / tf.cast(seq_lens, tf.float32)
        loss = tf.reduce_mean(loss)
        if 'sgd' == self.loss.lower():
            train_op = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        elif 'adam' == self.loss.lower():
            train_op = tf.train.AdamOptimizer(self.rate).minimize(loss)
        else:
            train_op = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for step, (seqs_batch, seq_lens_batch, labels_batch) in enumerate(self.batch_yield(data)):
                    _, curr_loss = sess.run([train_op, loss], feed_dict={seqs: seqs_batch, seq_lens: seq_lens_batch, labels: labels_batch})
                    if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                        logger.info("epoch:%d, batch: %d, current loss: %f" % (i, step+1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)

    def evaluate(self):
        eval_data = self.get_input_feature(self.eval_path)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    blc_cfg = {
        'train_path': '../../data/bilstm_crf/train_bio.txt',
        'eval_path': '../../data/bilstm_crf/eval_bio.txt',
        'max_len': 50,
        'batch_size': 64,
        'epoch': 50,
        'loss': 'adam',
        'rate': 0.001,
        'num_units': 128,
        'tf_config': tf_config,
        'model_path': '../../model/new_bilstm_crf/model',
        'summary_path': '../../model/new_bilstm_crf/summary'
    }
    blc = BiLstmCrf(**blc_cfg)
    blc.fit()