# encoding=utf-8
from tensorflow import keras
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

'''
超参数搜索
'''

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

# learning_rate: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
# W = W + grad * leaning_rate
learing_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
hisrories = []
for lr in learing_rates:
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
        keras.layers.Dense(1),
    ])
    optimizer = keras.optimizers.SGD(lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=30, callbacks=callbacks)
    hisrories.append(history)

for lr, history in zip(learing_rates, hisrories):
    print(lr, history)