'''
The Recurrent Neural Network is trained using TensorFlow + Keras
To use GPU, change keras layers from GRU to CuDDGRU
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import datetime
import functools

tf.enable_eager_execution()

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    '''
    Wrapper function to build keras TF RNN model.
    
    Parameters
    ----------
    vocab_size : int
        Character-level training. vocab_size refers to the number
        of unique characters found in training text.
    embedding_dim : int
        Output dimension of dense vector for embedding layer
    rnn_units : int
        Dimensionality of output space
    batch_size : int
        Batch size, # of samples processed before model is 
        updated
    
    Returns
    -------
    obj
        Returns a tensorflow model object
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, 
            embedding_dim,
            batch_input_shape=[batch_size, None]
        ),
        tf.keras.layers.GRU(
            units=rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True,
            recurrent_activation='sigmoid',
        ),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, 
        logits, 
        from_logits=True
    )

print(f'TensorFlow Version: {tf.VERSION}')
print(f'GPU Available: {tf.test.is_gpu_available()}')

#FILE_PATH = "../data/bt_headlines.txt"
FILE_PATH = "./data/st_headlines.txt"
data = open(FILE_PATH, 'rb').read().decode(encoding='utf-8')
words = sorted(set(data))
char2idx = {u:i for i, u in enumerate(words)}
idx2char = np.array(words)
data_int = np.array([char2idx[c] for c in data])
seq_length = 100
BUFFER_SIZE = 10000
BATCH_SIZE = 64

examples_per_epoch = len(data)//seq_length
steps_per_epoch = examples_per_epoch//BATCH_SIZE
char_dataset = tf.data.Dataset.from_tensor_slices(data_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = build_model(
    vocab_size = len(words),
    embedding_dim=256,
    rnn_units=1024,
    batch_size=64
)
model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss
)
print(model.summary())

# Set Training Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "st_ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Continue training from latest checkpoint
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
train = model.fit(
    dataset.repeat(), 
    epochs=3, 
    steps_per_epoch=steps_per_epoch, 
    callbacks=[checkpoint_callback]
)
print(tf.train.latest_checkpoint(checkpoint_dir))

# Save latest model
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model.save("./models/st_model_{}.h5".format(timestamp))