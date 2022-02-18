# encoding=utf-8
import os
import pprint

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)


# tfrecord 文件格式
# -> tf.train.Example
#   -> tf.train.Features -> {"key": tf.train.Feature}
#       -> tf.train.Feature -> tf.train.ByteList/FloatList/Int64List

favorite_books = [name.encode('utf-8') for name in ['machine learning', 'cc150']]
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bytelist)

hours_floatlist = tf.train.FloatList(value=[15.5, 9.5, 7.0, 8.0])
print(hours_floatlist)

age = tf.train.Int64List(value=[42])
print(age)

features = tf.train.Features(
    feature={
        'favorite_books': tf.train.Feature(bytes_list=favorite_books_bytelist),
        'hours': tf.train.Feature(float_list=hours_floatlist),
        'age': tf.train.Feature(int64_list=age)
    }
)
print(features)


example = tf.train.Example(features=features)
print(example)
serialized_example = example.SerializeToString()
print(serialized_example)


output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = 'test.tfrecords'
filename_fullpath = os.path.join(output_dir, filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)


dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)