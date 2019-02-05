import tensorflow as tf
import tensorflow.keras
import numpy as np
from misc.dataset.DataLoader import DataLoaderMnist
from tf_wiki.tfMLP.MLP import MLP
from tf_wiki.tfCNN.MyCNN import CNN

tf.enable_eager_execution()

num_batches = 5000
batch_size = 120
learning_rate = 0.001
model = CNN()
data_loader = DataLoaderMnist()
# 设置参数更新的方法是adam
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        # conver 2 tensor 28 * 28 => 1 * 28**2
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        # 计算损失
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

num_eval_samples = np.shape(data_loader.eval_labels)[0]
y_pred = model.predict(data_loader.eval_data).numpy()
print("test accuracy: %f" % (sum(y_pred == data_loader.eval_labels) / num_eval_samples))
