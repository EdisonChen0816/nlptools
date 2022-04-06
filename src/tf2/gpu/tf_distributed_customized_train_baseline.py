# encoding=utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# 打印版本信息
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(len(gpus))
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus))

fanshion_minst = keras.datasets.fashion_mnist

(x_train_all, y_train_all), (x_test, y_test) = fanshion_minst.load_data()
x_valid, x_train = x_train_all[: 5000], x_train_all[5000:]
y_valid, y_train = y_train_all[: 5000], y_train_all[5000:]

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

## 归一化
#  x = (x - mean) / std
scaler = StandardScaler()
# x_train: [None, 28, 28] -> [None, 784]
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)


def make_dataset(images, labels, epochs, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size).prefetch(50)
    return dataset


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    batch_size_per_replica = 256
    batch_size = batch_size_per_replica * len(logical_gpus)
    train_dataset = make_dataset(x_train_scaled, y_train, 1, batch_size)
    valid_dataset = make_dataset(x_valid_scaled, y_valid, 1, batch_size)
    train_dataset_distribute = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset_distribute = strategy.experimental_distribute_dataset(valid_dataset)


with strategy.scope():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# customized training loop.
# 1. define losses functions
# 2. define function train_step
# 3. define function test_step
# 4. for-loop training loop
with strategy.scope():
    # batch_size, batch_size / #{gpu}
    # eg: 64, gpu: 16
    loss_func = keras.losses.SparseCategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)


    def compute_loss(labels, predictions):
        per_replica_loss = loss_func(labels, predictions)
        return tf.nn.compute_average_loss(per_replica_loss, global_batch_size=batch_size)


    test_loss = keras.metrics.Mean(name='test_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    optimizer = keras.optimizers.SGD(lr=0.01)


    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss


    @tf.function
    def distributed_train_step(inputs):
        per_replica_average_loss = strategy.experimental_run_v2(train_step, args=(inputs, ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_average_loss, axis=None)


    def test_step(inputs):
        images, labels = inputs
        predictions = model(images)
        t_loss = loss_func(labels, predictions)
        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)


    @tf.function
    def distributed_test_step(inputs):
        strategy.experimental_run_v2(test_step, args=(inputs, ))

    epochs = 10
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for x in train_dataset:
            start_time = time.time()
            total_loss += distributed_train_step(x)
            run_time = time.time() - start_time
            num_batches += 1
            print('\rtotal_loss: %3.3f, num_batches: %3.3f, average_loss: %3.3f, time: %3.3f'
                  % (total_loss, num_batches, total_loss / num_batches, run_time), end='')
        train_loss = total_loss / num_batches
        for x in valid_dataset:
            distributed_test_step(x)
        print('\rEpoch: %d, Loss: %3.3f, Acc: %3.3f, Val_Loss: %3.3f, Val_Acc: %3.3f'
              % (epoch+1, train_loss, train_accuracy.result(), test_loss.result(), test_accuracy.result()))
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()