# -*- coding: UTF-8 -*-
"""
@file_name: simple_neural_network.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import fundamentals_python_mathematics_01.activity_functions as af
from util_font import *


class SimpleNeuralNetwork:
    """
    简单的单层神经网络，即无隐层，可实现线性可分的二分类数据，仅实现批量梯度下降法
    不采用特殊的优化方法：动量法、adagrad、adam....，仅为广义增量规则
    """

    def __init__(self, alpha=1e-2, eps=1e-10, av_fun="sigmoid", epochs=1000, SEED=0):
        self.alpha = alpha  # 学习率，即更新一次权重的尺度
        self.eps = eps  # 停机精度，满足精度要求即可停止优化
        # 返回值两个(激活函数activity_fun[0]，激活函数一阶导activity_fun[1])
        self.activity_fun = af.activity_functions(av_fun)  # 激活函数，默认sigmoid
        self.epochs = epochs  # 最大训练次数
        self.SEED = SEED  # 初始化权重系数的随机种子
        self.nn_weight = None  # 单层神经网络权重
        self.loss_values = []  # 每次训练的损失值

    @staticmethod
    def cal_cross_entropy(y, y_prob):
        """
        计算交叉熵损失，静态函数，无特征属性变量
        :param y: 样本真值，一维数组，shape=(n,)
        :param y_prob: 模型预测类别概率，一维数组，shape=(n,)
        :return:
        """
        return -(np.dot(y, np.log(y_prob)) + np.dot(1 - y, np.log(1 - y_prob)))

    def backward(self, y, y_hat):
        """
        反向传播算法，计算广义增量规则各变量的值，所有运算均为矢量化计算
        :param y: 样本真值，一维数组
        :param y_hat: 当前训练的网络输出值，一维数组
        :return:
        """
        error = y - y_hat  # 误差，一维数组相减，矢量化计算
        # 广义增量规则：fai(y_hat) * (1 - fai(y_hat)) * error，皆为矢量化计算
        delta = self.activity_fun[1](y_hat) * error  # 矢量化计算，向量间的乘法
        return delta

    def fit_net(self, X_train, y_train):
        """
        核心算法：单层神经网络模型训练，无隐藏层，只有一个输出节点
        :param X_train: 训练集，格式ndarray，shape = (n, m)
        :param y_train: 目标集，正确类别，格式ndarray，shape = (n, )
        :return:
        """
        if type(X_train) is not np.ndarray or type(y_train) is not np.ndarray:
            X_train, y_train = np.asarray(X_train, np.float), np.asarray(y_train)
        n_samples, n_feature = X_train.shape  # 样本量与特征数, n个样本，m个特征变量
        np.random.seed(self.SEED)  # 设置随机种子，以便可重现实验结果
        self.nn_weight = np.random.randn(n_feature) / 100  # 初始化网络权重，一维数组shape = (m,)
        # 在最大训练次数内，逐次迭代更新网络权重，即神经网络的训练过程，满足精度为止
        for epoch in range(self.epochs):
            # 批量梯度下降法，正向传播计算, 此处为矢量化计算：fai(net)
            y_prob = self.activity_fun[0](np.dot(self.nn_weight, X_train.T))  # 预测概率，非类别
            #  交叉熵损失函数，先计算当前预测概率，然后计算交叉熵损失，皆是矢量化计算
            self.loss_values.append(self.cal_cross_entropy(y_train, y_prob))  # 存储当前训练的误差损失
            # 停机规则：两次训练误差损失差小于给定的精度，即停止训练，最少训练10次
            if epoch > 10 and np.abs(self.loss_values[-1] - self.loss_values[-2]) < self.eps:
                break
            delta = self.backward(y_train, y_prob)  # 广义增量规则，delta为一维数组，shape = (n,)
            dw = self.alpha * np.dot(delta, X_train) / n_samples  # 权重更新增量，矢量化计算
            self.nn_weight = self.nn_weight + dw  # 更新权重，矢量化计算

    def predict_prob(self, X_test):
        """
        采用最终训练得到的网络权重，预测样本属于某个类别的概率
        :param X_test: 测试样本，二维数组，shape = (k, m)，k为样本量
        :return:
        """
        # 计算测试样本的预测概率y_prob，矢量化计算，含义等同于训练过程训练样本的预测值
        y_prob = self.activity_fun[0](np.dot(X_test, self.nn_weight))  # 或np.dot(self.nn_weight, X_test.T)
        y_hat_prob = np.zeros((X_test.shape[0], 2))  # 由于是两个类别，故维度(k, 2)
        y_hat_prob[:, 0] = 1 - y_prob  # 第1列为预测为0类别的概率
        y_hat_prob[:, 1] = y_prob  # 第2列为预测为1类别的概率
        return y_hat_prob

    def predict(self, X_test):
        """
        预测测试样本所属的类别
        :param X_test: 测试样本，二维数组
        :return:
        """
        y_hat_prob = self.predict_prob(X_test)  # 调用函数预测样本属于某个类别的概率
        # 按轴1获取最大值索引，即每一行的最大值索引，一行表示一个样本的预测为两个类别的概率
        return np.argmax(y_hat_prob, axis=1)

    def plt_loss_curve(self, is_show=True, title_txt=""):
        """
        绘制神经网络训练过程的损失下降曲线
        :return:
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(self.loss_values, "-.", lw=2,
                 label="$loss = %.10e$" % np.abs(self.loss_values[-1] - self.loss_values[-2]))
        plt.xlabel("迭代次数", fontdict={"fontsize": 18})
        plt.ylabel("交叉熵损失", fontdict={"fontsize": 18})
        plt.title("单层神经网络损失曲线（$%s$）" % title_txt, fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
