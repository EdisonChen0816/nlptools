# encoding=utf-8
import tensorflow as tf
from gensim.models import KeyedVectors
import jieba
import numpy as np
import os


w2v = KeyedVectors.load('../../model/w2v/w2v.model')


def input_fn():
    def get_input_feature():
        datas = []
        labels = []
        with open('../../data/textcnn/data', 'r', encoding='utf-8') as f:
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
                datas.append(embedding)
                if 'Meeting' == tag:
                    labels.append([0])
                else:
                    labels.append([1])
        return np.asarray(datas), np.asarray(labels)
    data, label = get_input_feature()
    data_batch, label_batch = tf.train.shuffle_batch([data[0], label[0]], batch_size=32, num_threads=32, capacity=(100+3*32), min_after_dequeue=50)
    return tf.reshape(data_batch, [32, 20, 300, 1]), label_batch


def model_fn(features, labels, mode, params):
    conv1 = tf.layers.conv2d(features, 16, kernel_size=(3, 300), strides=1, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 1), strides=1)
    fc1 = tf.layers.flatten(pool1, name="fc1")
    fc2 = tf.layers.dense(fc1, 128)
    fc3 = tf.layers.dropout(fc2, rate=0.2)
    logits = tf.layers.dense(fc3, params['n_classes'])
    pred = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predicitons = {
            'class_ids': pred[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predicitons)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=pred, name='acc_op')
    metrics = {'accuracy': accuracy}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = False
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
run_config = tf.estimator.RunConfig().replace(session_config=tf_config)
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='./tt',
    params={
        'hidden_units': [100, 100],
        'n_classes': 2
    },
    config=run_config
)
estimator.train(input_fn=input_fn, steps=5)