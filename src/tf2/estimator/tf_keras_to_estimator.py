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
