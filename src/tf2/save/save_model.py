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


print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fanshion_minst = tf.keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fanshion_minst.load_data()
x_valid, x_train = x_train_all[: 5000], x_train_all[5000:]
y_valid, y_train = y_train_all[: 5000], y_train_all[5000:]
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# history = model.fit(x_train, y_train, epochs=3, validation_data=(x_valid, y_valid))
#
# tf.saved_model.save(model, './keras_saved_graph')

# saved_model_cli show --dir ./keras_saved_graph --all
# saved_model_cli show --dir ./keras_saved_graph --all --tag_set serve -- signature_def serving_default
# saved_model_cli run --dir ./keras_saved_graph --all --tag_set serve -- signature_def serving_default
# --input_exprs 'flatten_input=np.ones((2, 28, 28))'

loaded_saved_model = tf.saved_model.load('./keras_saved_graph')
print(list(loaded_saved_model.signatures.keys()))
inference = loaded_saved_model.signatures['serving_default']
print(inference)
print(inference.structured_outputs)
results = inference(tf.constant(x_test[0: 1]))
print(results['dense_2'])