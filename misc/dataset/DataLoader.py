import tensorflow as tf
import numpy as np


class DataLoaderMnist():
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images  # np.array [55000, 784]
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)  # np.array [55000] of int32
        self.eval_data = mnist.test.images  # np.array [10000, 784]
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)  # np.array [10000] of int32

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]


class DataLoaderNiez():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
                                       origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index + seq_length])
            next_char.append(self.text[index + seq_length])
        return np.array(seq), np.array(next_char)  # [num_batch, seq_length], [num_batch]
