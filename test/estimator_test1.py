# encoding=utf-8
import tensorflow as tf
import os
import numpy as np
import jieba
from gensim.models import KeyedVectors
import random


def model_fn(features, labels, mode, params):
    lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=64)
    lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_fw, input_keep_prob=1.0, output_keep_prob=0.9)
    lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=64)
    lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_bw, input_keep_prob=1.0, output_keep_prob=0.9)
    output, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_cell_fw,
        cell_bw=lstm_cell_bw,
        inputs=features,
        dtype=tf.float64
    )
    h_state = output[0][:, -1, :]
    W = tf.get_variable("W", [64, 2],
                        initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float64)
    bias = tf.get_variable("bias", [2],
                           initializer=tf.zeros_initializer(), dtype=tf.float64)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
    cross_entropy_loss = -tf.reduce_mean(labels * tf.log(y_pre))
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
            cross_entropy_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy_loss,
            train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        accu = tf.metrics.accuracy(labels=labels,
                                   predictions=tf.argmax(y_pre, axis=-1, output_type=tf.int32))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=np.argmax(y_pre[0][0], 1).tolist(),
            loss=cross_entropy_loss,
            eval_metric_ops={"eval_accu": accu, "eval_loss": cross_entropy_loss})
    if mode == tf.estimator.ModeKeys.PREDICT:
        pass


w2v = KeyedVectors.load('../model/w2v/w2v.model')


def batch_yield(data, batch_size, shuffle=False):
    if shuffle:
        random.shuffle(data)
    seqs, labels = [], []
    for sent, tag in data:
        if len(seqs) == batch_size:
            yield np.asarray(seqs), np.asarray(labels)
            seqs = []
            labels = []
        seqs.append(sent)
        labels.append(tag)
    if len(seqs) != 0:
        yield np.asarray(seqs), np.asarray(labels)


def get_data():
    data = []
    with open('../data/textcnn/data', 'r', encoding='utf-8') as f:
        for line in f:
            embedding = []
            text, tag = line.replace('\n', '').split('\t')
            for word in jieba.lcut(text):
                if word in w2v:
                    embedding.append(w2v[word])
            if len(embedding) < 20:
                for i in range(20 - len(embedding)):
                    embedding.append([0] * 300)
            else:
                embedding = embedding[: 20]
            if 'Meeting' == tag:
                data.append([embedding, [0]])
            else:
                data.append([embedding, [1]])
    return data


def train_input_fn(x, y):
    def input_fn():
        return x, y
    return input_fn


session_config = tf.ConfigProto(log_device_placement=True)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
run_config = tf.estimator.RunConfig(
                model_dir='../model/test',
                save_checkpoints_steps=100,
                session_config=session_config)
model = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

for epoch in range(10):
    print(epoch)
    for x, y in batch_yield(get_data(), 32, True):
        fn = train_input_fn(x, y)
        model.train(input_fn=fn, max_steps=1000)