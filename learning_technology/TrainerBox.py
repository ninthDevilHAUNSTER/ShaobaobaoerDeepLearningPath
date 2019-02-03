import sys, os
import numpy as np
from misc.mathlab.shaobaobaoer_optimizer_lab import *


class TrainerBox(object):
    '''
    神经网络训练类
    '''

    def __init__(self,
                 network,
                 x_train, t_train, x_test, t_test,
                 epochs=20,
                 mini_batch_size=100,
                 optimizer='SGD', optimizer_param=None,
                 evaluate_sample_num_per_epoch=None,
                 verbose=True):
        '''

        :param network: 神经网络有序字典
        :param x_train: 训练数据
        :param t_train: 训练标签
        :param x_test: 测试数据
        :param t_test: 测试标签
        :param epochs: 迭代次数
        :param mini_batch_size: 小批batch
        :param optimizer: 梯度更新算法
        :param optimizer_param: 梯度更新算法参数
        :param evaluate_sample_num_per_epoch: 单批数据的个数
        :param verbose: 是否输出更多数据?
        '''
        if optimizer_param is None:
            optimizer_param = {'lr': .01}
        self.network = network
        self.x_train, self.t_train, self.x_test, self.t_test = x_train, t_train, x_test, t_test
        self.mini_batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose

        # optimizer :
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'adagrad': AdaGrad, 'rmsprpo': RMSprop,'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        # acc and loss list for drawing
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        '''
        单步训练
        :return:
        '''
        # mini batch function
        batch_mask = np.random.choice(self.train_size, self.mini_batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        # 更新梯度参数
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        # 神经网络跑一次
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        # 检测正确率
        if self.verbose:
            print("train loss :: " + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(
                    "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(
                        test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
