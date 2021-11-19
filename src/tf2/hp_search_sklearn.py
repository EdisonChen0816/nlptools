# encoding=utf-8
from tensorflow import keras
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import  reciprocal
from sklearn.model_selection import RandomizedSearchCV

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

# RandomizedSearchCV
# 1，转化为sklearn的model
# 2，定义参数集合
# 3，搜索参数


def build_model(hidden_layer=1, layer_size=30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu', input_shape=x_train.shape[1:]))
    for _ in range(hidden_layer - 1):
        model.add(keras.layers.Dense(layer_size, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = sklearn_model.fit(x_train, y_train, epochs=10, validation_data=[x_valid, y_valid], callbacks=callbacks)

print(history)

param_distribution = {
    'hidden_layer': [4, 5, 6],
    'layer_size': np.arange(1, 100),
    'learning_rate': reciprocal(1e-4, 1e-2) # f(x) = 1/(x*log(b/a))  a<=x<=b
}

random_search_cv = RandomizedSearchCV(sklearn_model, param_distribution, n_iter=3, n_jobs=1)
random_search_cv.fit(x_train, y_train, epochs=10, validation_data=[x_valid, y_valid], callbacks=callbacks)

print(random_search_cv.best_params_)
print(random_search_cv.best_score_)
print(random_search_cv.best_estimator_)

model = random_search_cv.best_estimator_.model
model.evaluate(x_test. y_test)