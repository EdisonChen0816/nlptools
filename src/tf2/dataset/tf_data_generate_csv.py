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

output_dir = 'generate_csv'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def save_to_csv(output_dir, data, name_prefix, header=None, n_parts=10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []
    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, 'wt', encoding='utf-8') as f:
            if header is not None:
                f.write(header + '\n')
            for row_index in row_indices:
                f.write(','.join([repr(col) for col in data[row_index]]))
                f.write('\n')
    return filenames


train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]
header_cols = housing.feature_names + ['MidianHouserValue']
header_str = ','.join(header_cols)

train_filenames = save_to_csv(output_dir, train_data, 'train', header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, 'valid', header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data, 'test', header_str, n_parts=10)

# 1，filename -> dataset
# 2，read file -> dataset —> datasets -> merge
# 3，parse csv
filename_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filename_dataset:
    print(filename)

n_readers = 5
dataset = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),
    cycle_length=n_readers
)

for line in dataset.take(15):
    print(line.numpy())


# tf.io.decode_csv(str, record_defaults)
sample_str = '1,2,3,4,5'
# record_defaults = [tf.constant(0, dtype=tf.int32)] * 5
record_defaults = [
    tf.constant(0, dtype=tf.int32),
    0,
    np.nan,
    'hello',
    tf.constant([])
]
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fields)

try:
    parsed_fields = tf.io.decode_csv(',,,,', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

try:
    parsed_fields = tf.io.decode_csv('1,2,3,4,5,6,7', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0: -1])
    y = tf.stack(parsed_fields[-1: ])
    return x, y


parse_csv_line(b'-0.9974222662636643,1.2333642636907922,-0.7577192870888144,-0.011109251557751528,-0.23003784053222506,0.05487422342718872,-0.757726890467217,0.7065494722340417,1.739',
               n_fields=9)


def csv_reader_dataset(filenames, n_reader=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=1000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filenames).skip(1),
        cycle_length=n_reader
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


train_set = csv_reader_dataset(train_filenames, batch_size=3)
for x_batch, y_batch in train_set.take(2):
    print('x:')
    pprint.pprint(x_batch)
    print('y:')
    pprint.pprint(y_batch)

batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(train_filenames, batch_size=batch_size)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[8]),
    keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(train_set,
                    validation_data=valid_set,
                    steps_per_epoch=11160//batch_size,
                    validation_steps=3870//batch_size,
                    epochs=100,
                    callbacks=callbacks)
model.evaluate(test_set, steps=5160//batch_size)