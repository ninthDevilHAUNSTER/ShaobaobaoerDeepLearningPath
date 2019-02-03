import numpy as np
import tensorflow.keras
import tensorflow as tf
from misc.dataset.DataLoader import DataLoader
tf.enable_eager_execution()
class MLP(tf.keras.Model):
    def __init__(self):
        '''
        设置一个附带输出的全连接层和一个输出层
        '''
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        logits = self(x)
        return tf.argmax(logits, axis=-1)
