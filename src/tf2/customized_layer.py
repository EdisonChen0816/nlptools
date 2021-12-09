# encoding=utf-8
from tensorflow import keras
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(100, input_shape=[None, 5])
layer(tf.zeros([10, 5]))

# x * w + b
print(layer.variables)
print(layer.trainable_variables)
help(layer)


# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
#     keras.layers.Dense(1)
# ])
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='sgd')
# history = model.fit(x_train_scaled, y_train, validation_data=[x_valid_scaled, y_valid], epochs=100)
