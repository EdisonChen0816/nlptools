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


customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))
print(customized_softplus([-10., -5., 0., 10.]))


# customized dense layer
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''构建所需要的参数'''
        # x * w + b  input_shape: [None, a] w:[a, b] output_shape: [None, b]
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.units),
                                      initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.units, ),
                                    initializer='zeros', trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        '''完成正向计算'''
        return self.activation(x @ self.kernel + self.bias)


model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu', input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # 等价于 keras.layers.Dense(1, activation='softplus')
    # 或者 keras.layers.Dense(1),keras.layers.Activation('softplus')
])

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
#     keras.layers.Dense(1)
# ])
model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd')
history = model.fit(x_train_scaled, y_train, validation_data=[x_valid_scaled, y_valid], epochs=100)
