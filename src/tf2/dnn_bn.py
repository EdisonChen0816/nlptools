# encoding=utf-8
import tensorflow as tf
from tensorflow import keras


print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation='relu'))
    # 批归一化
    # model.add(keras.layers.BatchNormalization())
    # BN在激活函数之前
    # model.add(keras.layers.Dense(100))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Activation('relu'))
# AlphaDropout：1，均值和方差不变。2，归一化性质也不变
model.add(keras.layers.AlphaDropout(rate=0.5))
# model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
print(history.history)