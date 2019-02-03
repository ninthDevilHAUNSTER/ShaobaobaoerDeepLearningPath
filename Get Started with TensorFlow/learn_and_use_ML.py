from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from misc.dataset.mnist import load_mnist
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

def help_function(train_images):
    a = []
    for i in range(train_images.__len__()):
        a.append(train_images[i][0])
    return np.array(a,dtype='int32')



def train():
    # keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=False, flatten=False)
    train_images =help_function(train_images)
    test_images = help_function(test_images)


    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    show_img(train_images, class_names, train_labels)

    test_images = test_images / 255.0
    train_images = train_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=20)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


def plot_image(i, predictions_array, true_label, img,class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def show_img(train_images, class_names, train_labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # print(train_images[i])
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


if __name__ == '__main__':
    train()
