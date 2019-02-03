import numpy as np
from ConvolutionalNeuralNetwork.SimpleCNN import SimpleCNN
from learning_technology.TrainerBox import TrainerBox
from misc.dataset.mnist import load_mnist
from matplotlib import pyplot as plt

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

max_epochs = 20

network = SimpleCNN(input_dim=(1, 28, 28),
                    conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                    hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = TrainerBox(network, x_train, t_train, x_test, t_test,
                     epochs=max_epochs, mini_batch_size=120,
                     optimizer='Adam', optimizer_param={'lr': 0.001},
                     evaluate_sample_num_per_epoch=1000)
trainer.train()

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
