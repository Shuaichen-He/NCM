# -*- coding: UTF-8 -*-
"""
@file_name: mat_eig_utils.py
@time: 2022-11-13
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class MatrixEigenvalueUtils:
    """
    矩阵特征值计算工具类，主要用于显示和可视化
    """

    def __init__(self, iter_eigenvalue, iter_eig_vector):
        self.iter_eigenvalue = np.asarray(iter_eigenvalue)  # 特征值迭代过程值
        self.iter_eig_vector = np.asarray(iter_eig_vector)  # 特征向量迭代过程值
        self.n_iter = self.iter_eig_vector.shape[0]  # 迭代次数

    def show_iteration(self):
        """
        显示求解过程的特征值和特征向量
        :return:
        """
        print("矩阵的特征值和特征向量迭代求解过程如下：")
        print("-" * 70)
        for eig, vector in zip(self.iter_eigenvalue, self.iter_eig_vector):
            print("%3d %15.10f %3s" % (eig[0], eig[1], "|"), end="")
            for v in vector:
                print("%15.10f" % (v / np.max(np.abs(vector))), end="")
            print()
        print("-" * 70)
        print("迭代求解，矩阵的特征值：%.20e" % self.iter_eigenvalue[-1, 1])
        print("特征向量：[", end="")
        for i, v in enumerate(self.iter_eig_vector[-1]):
            if i < len(self.iter_eig_vector[-1]) - 1:
                # print("%.15f" % (v / np.max(np.abs(self.iter_eig_vector[-1]))), end="  ") # 归一化
                print("%.15f" % (v), end="  ")  # 未进行归一化
            else:
                # print("%.15f" % (v / np.max(np.abs(self.iter_eig_vector[-1]))))  # 归一化
                print("%.15f" % (v))  # 未进行归一化

    def plt_matrix_eig(self, title=""):
        """
        对特征值、特征向量迭代过程进行可视化
        :return:
        """
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        # plt.plot(self.iter_eigenvalue[:, 0] + 1, self.iter_eigenvalue[:, 1], "o--",
        #          label="$\lambda_1^{(k)}: k=%d$" % self.n_iter)
        plt.plot(self.iter_eigenvalue[:, 0] + 1, self.iter_eigenvalue[:, 1],
                 label="$\lambda_1: k=%d$" % self.n_iter)
        plt.plot(self.iter_eigenvalue[-1, 0] + 1, self.iter_eigenvalue[-1, 1], "D",
                 label="$\lambda_1=%.15f$" % self.iter_eigenvalue[-1, 1])
        plt.legend(frameon=False, fontsize=18)
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$\lambda_1^{(k)}$", fontdict={"fontsize": 18})
        plt.title("%s：特征值收敛曲线" % title, fontdict={"fontsize": 18})
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.subplot(122)
        p_type = ["*", "+", "x", "o", "v", "^", "<", ">", "p", "s", "h", "d"]  # 点的类型
        for i in range(len(self.iter_eig_vector[-1])):
            # plt.plot(np.arange(1, self.n_iter + 1), self.iter_eig_vector[:, i], "-" + p_type[i],
            #          label="$x_%d=%.8f$" % (i + 1, self.iter_eig_vector[-1, i]))
            plt.plot(np.arange(1, self.n_iter + 1), self.iter_eig_vector[:, i], "-",
                     label="$x_{%d}=%.8f$" % (i + 1, self.iter_eig_vector[-1, i]))
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$X_1(\lambda_1)$", fontdict={"fontsize": 18})
        plt.title("%s：特征向量收敛曲线" % title, fontdict={"fontsize": 18})
        plt.grid(ls=":")
        plt.legend(frameon=False, fontsize=18, loc="center right")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.show()
