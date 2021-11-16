# encoding=utf-8
from tensorflow import keras
import tensorflow as tf

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train, y_valid, y_train = x_train_all[:5000], x_train_all[5000:], y_train_all[:5000], y_train_all[5000:]
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


# 函数式API 功能API
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
# 复合函数：f(x) = h(g(x))

concat = keras.layers.concatenate([input, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input], outputs=[output])
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=100, callbacks=callbacks)