# encoding=utf-8
import os
from tensorflow import keras
import numpy as np
import tensorflow as tf

input_filepath = './shakespeare.txt'
text = open(input_filepath, 'r').read()

print(len(text))
print(text[0: 100])

# 1.generate vocab
# 2.build mapping char -> id
# 3.data -> id_data
# 4.abcd -> bcd<eos>
vocab = sorted(set(text))
print(len(vocab))
print(vocab)

char2idx = {char: idx for idx, char in enumerate(vocab)}
print(char2idx)
idx2char = np.array(vocab)
print(idx2char)
text_as_int = np.array([char2idx[c] for c in text])
print(text_as_int[0: 10])
print(text[0: 10])


def split_input_target(id_text):
    '''
    abcde -> abcd, bcde
    :param id_text:
    :return:
    '''
    return id_text[0: -1], id_text[1:]


char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
seq_length = 100
seq_dataset = char_dataset.batch(seq_length + 1, drop_remainder=True)
for ch_id in char_dataset.take(2):
    print(ch_id, idx2char[ch_id.numpy()])

for seq_id in seq_dataset.take(2):
    print(seq_id)
    print(repr(''.join(idx2char[seq_id.numpy()])))

seq_dataset = seq_dataset.map(split_input_target)
for item_input, item_output in seq_dataset.take(2):
    print(item_input.numpy())
    print(item_output.numpy())

batch_sise = 64
buffer_size = 10000
seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_sise, drop_remainder=True)


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        # keras.layers.SimpleRNN(units=rnn_units, return_sequences=True),
        keras.layers.LSTM(units=rnn_units, stateful=True, recurrent_initializer='glorot_uniform', return_sequences=True),
        keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_sise)
model.summary()

for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)

# random sampling
# greedy, random
sample_indices = tf.random.categorical(logits=example_batch_predictions[0], num_samples=1)
print(sample_indices)
sample_indices = tf.squeeze(sample_indices, axis=-1)
print(sample_indices)
print('Input:', repr(''.join(idx2char[input_example_batch[0]])))
print('Output:', repr(''.join(idx2char[target_example_batch[0]])))
print('Prediction:', repr(''.join(idx2char[sample_indices])))


def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)
example_loss = loss(target_example_batch, example_batch_predictions)
print(example_loss.shape)
print(example_loss.numpy().mean())

output_dir = './text_generation_checkpoints'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)
epochs = 3
history = model.fit(seq_dataset, epochs=epochs, callbacks=[checkpoint_callback])

print(tf.train.latest_checkpoint(output_dir))

model2 = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model2.load_weights(tf.train.latest_checkpoint(output_dir))
model2.build(tf.TensorShape([1, None]))
# start ch sequenece A,
# A -> tf2model -> b
# A.append(b) -> B
# B -> tf2model -> c
# B.append(c) -> C
# C(Abc) -> tf2model -> ...
model2.summary()


def generate_text(model, start_string, num_generate=1000):
    input_evel = [char2idx[ch] for ch in start_string]
    input_evel = tf.expand_dims(input_evel, 0)
    text_generated = []
    model.reset_states()
    # temperature > 1, random，更平滑
    # temperature < 1, greddy，更陡峭
    temperature = 0.5
    for _ in range(num_generate):
        # 1. tf2model inference -> predictions
        # 2. sample
        # 3. update input_eval
        # predictions : [batch_size, input_eval_len, vocab_size]
        predictions = model(input_evel)
        # predictions: logits -> softmax -> prob
        # softmax: e^xi
        # eg: 4, 2 e^4 / (e^4 + e^2) = 0.88, e^2 / (e^4 + e^2) = 0.12
        # eg: 2, 1 e^2 / (e^2 + e) = 0.73, e^1 / (e^2 + e) = 0.27
        predictions = predictions / temperature
        # predictions : [input_evel_len, vocab_size]
        predictions = tf.squeeze(predictions, 0)
        # a b c -> b c da
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        text_generated.append(idx2char[predicted_id])
        # s, x -> rnn -> s', y
        input_evel = tf.expand_dims([predicted_id], 0)
    return start_string + ''.join(text_generated)


new_text = generate_text(model2, 'All:')