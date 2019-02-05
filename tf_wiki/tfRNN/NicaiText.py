import os
import sys
# sys.path.append('tf_wiki/common')
import tensorflow as tf
import tensorflow.keras
import numpy as np

from misc.dataset.DataLoader import DataLoaderNiez
from tf_wiki.tfRNN.RNN import RNN

tf.enable_eager_execution()
summary_writer = tf.contrib.summary.create_file_writer('./tensorboard')

num_batches = 10000
batch_size = 100
learning_rate = 0.001
seq_length = 10
model = RNN(26)

data_loader = DataLoaderNiez()
model = RNN(len(data_loader.chars))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(X)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    X = X_
    print("diversity %f:" % diversity)
    for t in range(400):
        y_pred = model.predict(X, diversity)
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
