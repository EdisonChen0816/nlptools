# encoding=utf-8
from tensorflow import keras
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print(tf.__version__)

housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)


x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

# 多输入
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
x_train_wide = x_train[:, :5]
x_train_deep = x_train[:, 2:]
x_valid_wide = x_valid[:, :5]
x_valid_deep = x_valid[:, 2:]
x_test_wide = x_test[:, :5]
x_test_deep = x_test[:, 2:]
history = model.fit([x_train_wide, x_train_deep], y_train, validation_data=([x_valid_wide, x_valid_deep], y_valid), epochs=100, callbacks=callbacks)
model.evaluate([x_test_wide, x_test_deep], y_test)