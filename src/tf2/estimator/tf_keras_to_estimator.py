# encoding=utf-8
import os
import pprint
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)

train_file = './'
eval_file = './'

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

print(train_df.head())
print(eval_df.head())

y_train = train_df.pop('survived') # 去掉类别标签
y_evel = eval_df.pop('survived')

print(train_df.head())
print(eval_df.head())

train_df.describe()

print(train_df.shape, eval_df.shape)
train_df.age.hist(bins=20)
train_df.sex.value_counts().plot(kind='barh')
train_df['class'].value_counts().plot(kind='barv')
pd.concat([train_df, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh')



#######################################

fanshion_minst = keras.datasets.fashion_mnist

(x_train_all, y_train_all), (x_test, y_test) = fanshion_minst.load_data()
x_valid, x_train = x_train_all[: 5000], x_train_all[5000:]
y_valid, y_train = y_train_all[: 5000], y_train_all[5000:]

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


model = keras.models.Sequential([
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])

estimator = keras.estimator.model_to_estimator(model)


def input_fn():
    features = []
    labels = []
    return (features, labels)


# input_fn 必须是函数，
# 返回 (1) (features, labels) (2) datasets -> (features, labels)
estimator.train(input_fn=input_fn)
estimator.evaluate(input_fn=input_fn)