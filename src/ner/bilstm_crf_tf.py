# ecnoding=utf-8
import sys
import time
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell
import logging
import pandas as pd
import numpy as np
import os
import random


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bilstm_crf')


class BiLSTM_CRF:

    def __init__(self, batch_size, epoch, hidden_dim, CRF, update_embedding, dropout, optimizer, lr,
                 clip, shuffle, model_path, data_path, tag2label, summary_path, config, is_train=False, embedding_dim=300):
        self.batch_size = batch_size
        self.epoch_num = epoch
        self.hidden_dim = hidden_dim
        self.embeddings, self.word2id = self._get_embeddings(data_path, embedding_dim)
        self.CRF = CRF
        self.update_embedding = update_embedding
        self.dropout_keep_prob = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.shuffle = shuffle
        self.model_path = model_path
        self.summary_path = summary_path
        self.config = config
        if not is_train:
            self.ckpt_file, self.saver = self.load_model(self.model_path)

    def load_model(self, model_path):
        ckpt_file = tf.train.latest_checkpoint(model_path)
        self.build_graph()
        saver = tf.train.Saver()
        return ckpt_file, saver

    def _batch_yield(self, data, batch_size, tag2label, shuffle=False):
        """
        :param data:
        :param batch_size:
        :param tag2label:
        :param shuffle:
        :return:
        """
        if shuffle:
            random.shuffle(data)
        seqs, labels = [], []
        for (sent_, tag_) in data:
            sent_ = self._sentence2id(sent_)
            label_ = [tag2label[tag] for tag in tag_]
            if len(seqs) == batch_size:
                yield seqs, labels
                seqs, labels = [], []
            seqs.append(sent_)
            labels.append(label_)
        if len(seqs) != 0:
            yield seqs, labels

    def _get_embeddings(self, data_path, embedding_dim):
        word2id = {}
        df = pd.read_csv(data_path, sep='\t', names=['text', 'tag'])
        wi = 1
        for word in df["text"].values.tolist():
            if word not in word2id:
                word2id[word] = wi
                wi += 1
        word2id['<UNK>'] = wi
        word2id['<PAD>'] = 0
        embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), embedding_dim))
        return np.float32(embedding_mat), word2id

    def _sentence2id(self, sent):
        """
        :param sent:
        :param word2id:
        :return:
        """
        sentence_id = []
        for word in sent:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in self.word2id:
                word = '<UNK>'
            sentence_id.append(self.word2id[word])
        return sentence_id

    def _pad_sequences(self, sequences, pad_mark=0):
        """
        :param sequences:
        :param pad_mark:
        :return:
        """
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W", shape=[2 * self.hidden_dim, self.num_tags], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name="b", shape=[self.num_tags], initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """
        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train):
        """
        :param train:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, epoch)
            saver.save(sess, self.model_path)

    def predict_one(self, sess, sent):
        """
        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs, labels in self._batch_yield(sent, self.batch_size, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, epoch):
        """
        :param sess:
        :param train:
        :param dev:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = self._batch_yield(train, self.batch_size, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1, loss_train, step_num))
            self.file_writer.add_summary(summary, step_num)
        # saver.save(sess, self.model_path)
        # logger.info('===========validation / test===========')
        # label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        # self.evaluate(label_list_dev, dev)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = self._pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = self._pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list
        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, data):
        """
        :param label_list:
        :param data:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)

    def get_entity(self, text, tags):
        entities = []
        i = 0
        while i < len(tags):
            if 'B' == tags[i]:
                entity = text[i]
                i += 1
                while i < len(tags) and ('M' == tags[i] or 'E' == tags[i]):
                    if 'E' == tags[i]:
                        entity += text[i]
                        break
                    entity += text[i]
                    i += 1
                entities.append(entity)
            i += 1
        return entities

    def predict(self, text):
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.ckpt_file)
            l = []
            if isinstance(text, list):
                l = text
            else:
                i = 0
                while i < len(text):
                    if '<' == text[i]:
                        s = text[i]
                        i += 1
                        while i < len(text) and '>' != text[i]:
                            s += text[i]
                            i += 1
                        s += text[i]
                        l.append(s)
                    else:
                        l.append(text[i])
                    i += 1
            text = [(l, ['O'] * len(text))]
            tag = self.predict_one(sess, text)
            entities = self.get_entity(l, tag)
            return entities


# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

tag2label = {
    "O": 0,
    "B": 1,
    "M": 2,
    "E": 3
}


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


def train(train_path, model_path, summary_path):
    '''
    :param train_path: 训练数据路径
    :param test_path:  测试数据路径
    :param model_path:  模型保存路径
    :param summary_path: summary保存路径
    :return:
    '''
    model = BiLSTM_CRF(batch_size=64, epoch=40, hidden_dim=300, CRF=True, update_embedding=True,
                       dropout=0.5, optimizer='Adam', lr=0.001, clip=5.0, shuffle=True, model_path=model_path,
                       data_path=train_path, tag2label=tag2label, summary_path=summary_path, config=config, is_train=True)
    model.build_graph()
    train_data = read_corpus(train_path)
    model.train(train=train_data)


if __name__ == '__main__':
    train_data = '../../../data/bilstm_crf/train_data'
    train_model_path = '../../../model/bilstm_crf/bilstm_crf_model'
    predict_model_path = '../../model/bilstm_crf'
    summary_path = '../../model/bilstm_crf/summary'
    # train(train_data, train_model_path, summary_path)

    model = BiLSTM_CRF(batch_size=64, epoch=40, hidden_dim=300, CRF=True, update_embedding=True,
                       dropout=0.5, optimizer='Adam', lr=0.001, clip=5.0, shuffle=True, model_path=predict_model_path,
                       data_path=train_data, tag2label=tag2label, summary_path=summary_path, config=config)

    print(model.predict('中国很大'))