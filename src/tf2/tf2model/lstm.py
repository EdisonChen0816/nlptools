# encoding=utf-8
from tensorflow import keras

imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size, index_from=index_from)
print(train_data[0], train_labels[0])
print(train_data.shape, train_labels.shape)
print(len(train_data[0]), len(train_data[1]))
print(test_data.shape, test_labels.shape)

word_index = imdb.get_word_index()
print(len(word_index))
# print(word_index)
word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3

reverse_word_index = dict([(value, key) for key, value in word_index.items()])


def decode_review(text_ids):
    return ' '.join(reverse_word_index.get(word_id, '<UNK>') for word_id in text_ids)


decode_review(train_data[0])

max_length = 500
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=max_length)
print(train_data[0])
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=max_length)

embedding_dim = 16
batch_size = 128


bi_rnn_model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(
        keras.layers.LSTM(units=64, return_sequences=True)
    ),
    keras.layers.Bidirectional(
        keras.layers.LSTM(units=64, return_sequences=False)
    ),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
bi_rnn_model.summary()
bi_rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_bi_rnn = bi_rnn_model.fit(train_data, train_labels, epochs=5, batch_size=batch_size, validation_split=0.2)
bi_rnn_model.evaluate(test_data, test_labels, batch_size=batch_size)