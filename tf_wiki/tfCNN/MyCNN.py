import tensorflow as tf
import tensorflow.keras


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=2
        )
        self.flatten = tf.keras.layers.Reshape(
            target_shape=(7 * 7 * 32,)
        )
        self.dense1 = tf.keras.layers.Dense(
            units=1024, activation=tf.nn.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10
        )

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        x = self.conv1(inputs) # N  28 * 28 * 32
        x = self.pool1(x) # N 14 14 32
        x = self.conv2(x) # N 14 14 64
        x = self.pool2(x) # N 7 7 64
        x = self.flatten(x) # 7 * 7 * 64
        x = self.dense1(x) # N 1024
        x = self.dense2(x) # N 10
        return x

    def predict(self, inputs):
        logits = self(inputs)
        return tf.argmax(logits, axis=-1)
