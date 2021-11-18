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


# 子类API
class WideDeepModel(keras.models.Model):

    def __init__(self):
        super(WideDeepModel, self).__init__()
        '''定义模型的层次'''
        self.hidden1_layer = keras.layers.Dense(30, activation='relu')
        self.hideen2_layer = keras.layers.Dense(30, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, input):
        '''
        完成模型的正向计算
        :param input:
        :return:
        '''
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hideen2_layer(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output


# model = WideDeepModel()
model = keras.models.Sequential([
    WideDeepModel()
])
model.build(input_shape=(None, 8))

model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=100, callbacks=callbacks)
model.evaluate(x_test, y_test)