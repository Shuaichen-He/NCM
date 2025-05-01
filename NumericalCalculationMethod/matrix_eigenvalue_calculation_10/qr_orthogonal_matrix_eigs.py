# -*- coding: UTF-8 -*-
"""
@file_name: qr_orthogonal_matrix_eigs.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from decimal import *
from util_font import *

getcontext().prec = 128


class QROrthogonalMatrixEigenvalues:
    """
    QR正交化方法求解矩阵全部特征值，其对应的特征向量采用最速下降法求解
    """

    def __init__(self, A, eps=1e-8, max_iter=1000, is_show=False):
        self.A = np.asarray(A, dtype=np.float64)
        self.n = self.A.shape[0]
        if np.linalg.matrix_rank(A) != self.n:
            print("矩阵A非满秩，不能用qr正交化分解.")
            exit(0)
        self.eps = eps  # 迭代精度要求
        self.max_iter = max_iter  # 最大迭代次数
        self.is_show = is_show  # 是否打印结果
        self.eigenvalues = np.zeros(self.n)  # 存储矩阵全部特征值
        self.iter_eigenvalues = []  # 存储迭代求解全部特征值的值
        self.iter_precision = []  # 存储相邻两次特征值差的2范数

    def fit_eig(self):
        """
        QR方法求解矩阵全部特征值
        :return:
        """
        # 第一轮迭代
        Q, R = self._schmidt_orthogonal(self.A)  # 施密特正交分解
        A_k = np.dot(R, Q)  # A^(k)
        self.iter_eigenvalues.append(np.diag(A_k))  # 记录过程
        tol, iter_ = np.infty, 1  # 初始化精度和迭代变量
        while np.abs(tol) > self.eps and iter_ < self.max_iter:
            Q, R = self._schmidt_orthogonal(A_k)  # 施密特正交分解
            A_k = np.dot(R, Q)  # A^(k)
            self.iter_eigenvalues.append(np.diag(A_k))  # 记录过程
            tol = np.linalg.norm(self.iter_eigenvalues[-1] - self.iter_eigenvalues[-2])
            self.iter_precision.append(tol)
            iter_ += 1
        self.eigenvalues = sorted(self.iter_eigenvalues[-1], reverse=True)  # 最终特征值
        return self.eigenvalues

    def _schmidt_orthogonal(self, A_k):
        """
        施密特正交分解法
        :return:
        """
        Q = np.copy(A_k)  # 正交矩阵Q
        Q[:, 0] = Q[:, 0] / np.linalg.norm(Q[:, 0])  # A的第一列正规化
        for i in range(1, self.n):
            for j in range(i):
                # 使A的第i列与前面所有的列正交
                Q[:, i] = Q[:, i] - np.dot(Q[:, i], Q[:, j]) * Q[:, j]
            Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])  # 正规化
        R = np.dot(Q.T, A_k)
        return Q, R

    def show_iteration(self):
        """
        显示求解过程的特征值和特征向量
        :return:
        """
        iter_num = len(self.iter_eigenvalues)
        if iter_num > 1:
            print("矩阵的特征值迭代求解过程及相邻两次迭代特征值差的2范数如下：")
            print("-" * 70)
            print("%3d" % 1, end="")
            for e in self.iter_eigenvalues[0]:
                print("%20.15f" % e, end="")
            print()
            for iter_, (eig, prec) in enumerate(zip(self.iter_eigenvalues[1:], self.iter_precision)):
                print("%3d" % (iter_ + 2), end="")
                for e in eig:
                    print("%20.15f" % e, end="")
                print("%20.10e" % prec)
            print("-" * 70)

    def plt_eigenvalues(self):
        """
        绘制qr分解迭代求解过程的特征值收敛曲线
        :return:
        """
        eigenvalues = np.asarray(self.iter_eigenvalues)
        iter_ = np.arange(1, eigenvalues.shape[0] + 1)  # 迭代次数
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        # ls_ = ["-", "--", "-.", ":", "-"]
        for i in range(self.n):
            plt.plot(iter_, eigenvalues[:, i], lw=2,
                     label="$\lambda_{%d}=%.8f$" % (i + 1, eigenvalues[-1, i]))
            # plt.plot(iter_, eigenvalues[:, i], ls_[i], lw=2,
            #          label="$\lambda_{%d}=%.8f$" % (i + 1, eigenvalues[-1, i]))
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$\lambda_k$", fontdict={"fontsize": 18})
        plt.title("$Schmidt \ QR$法求解$\lambda^{*}_{k}$的收敛性", fontdict={"fontsize": 18})
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.legend(frameon=False, fontsize=18, loc="upper right", ncol=2)
        plt.subplot(122)
        plt.semilogy(iter_[1:], self.iter_precision, "*-",
                     label="$\epsilon, \ k = %d$" % len(iter_))
        plt.semilogy(iter_[-1], self.iter_precision[-1], "D", label="$\epsilon = %.10e$" % self.iter_precision[-1])
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$\epsilon = \Vert \lambda^{k+1} - \lambda^k \Vert$", fontdict={"fontsize": 18})
        plt.title("$Schmidt \ QR$法求解$\lambda^{*}$的精度收敛曲线", fontdict={"fontsize": 18})
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.legend(frameon=False, fontsize=18)
        plt.show()
