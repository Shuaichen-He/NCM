# -*- coding: UTF-8 -*-
"""
@file_name: heisenberg_qr_matrix_eigs.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.up_heisenberg_matrix import UPHeisenbergMatrix
from decimal import *
from util_font import *

getcontext().prec = 128


class HeisenbergQRMatrixEig:
    """
    上海森伯格矩阵，默认Givens 正交化方法求解矩阵全部特征值
    """

    def __init__(self, A, eps=1e-8, max_iter=1000, transform="Givens"):
        heisenberg = UPHeisenbergMatrix(A)
        self.A = heisenberg.cal_heisenberg_mat()  # 上海森伯格矩阵计算
        # print("上海森伯格矩阵为：\n", self.A)
        self.n = self.A.shape[0]
        # if np.linalg.det(A) < 1e-10:
        #     print("矩阵A非满秩，不能用qr正交化分解.")
        #     exit(0)
        self.eps, self.max_iter = eps, max_iter  # 迭代精度要求和最大迭代次数
        self.transform = transform  # QR正交分解方法，默认Givens
        self.eigenvalues = np.zeros(self.n)  # 存储矩阵全部特征值
        self.iter_eigenvalues = []  # 存储迭代求解全部特征值的值
        self.iter_precision = []  # 存储相邻两次特征值差的2范数

    def fit_eig(self):
        """
        QR方法求解矩阵全部特征值
        :return:
        """
        orthogonal_fun = None  # 用于选择正交化的方法
        if self.transform.lower() == "givens":
            orthogonal_fun = eval("self._givens_rotation_")
        elif self.transform.lower() == "schmidt":
            orthogonal_fun = eval("self._schmidt_orthogonal_")
        elif self.transform.lower() == "householder":
            orthogonal_fun = eval("self._householder_transformation_")
        else:
            print("QR正交分解有误，支持Givens、Schmidt或Householder。")
            exit(0)

        Q, R = orthogonal_fun(self.A)  # QR正交分解法
        orthogonal_mat = np.dot(R, Q)  # Ak
        self.iter_eigenvalues.append(np.diag(orthogonal_mat))
        tol, iter_ = np.infty, 1  # 初始化精度和迭代变量
        while np.abs(tol) > self.eps and iter_ < self.max_iter:
            Q, R = orthogonal_fun(orthogonal_mat)  # QR正交分解法
            orthogonal_mat = np.dot(R, Q)  # Ak
            self.iter_eigenvalues.append(np.diag(orthogonal_mat))
            tol = np.linalg.norm(self.iter_eigenvalues[-1] - self.iter_eigenvalues[-2])
            self.iter_precision.append(tol)
            iter_ += 1
        self.eigenvalues = sorted(self.iter_eigenvalues[-1], reverse=True)  # 最终特征值
        return self.eigenvalues

    def _schmidt_orthogonal_(self, orth_mat):
        """
        施密特正交分解法
        :return:
        """
        Q = np.copy(orth_mat)
        Q[:, 0] = Q[:, 0] / np.linalg.norm(Q[:, 0])  # A的第一列正规化
        for i in range(1, self.n):
            for j in range(i):
                Q[:, i] = Q[:, i] - np.dot(Q[:, i], Q[:, j]) * Q[:, j]
            Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])
        R = np.dot(Q.T, orth_mat)
        return Q, R

    def _householder_transformation_(self, orth_mat):
        """
        豪斯霍尔德Householder变换方法求解QR
        :return:
        """
        # 1. 按照householder变换进行正交化求解QR
        # 1.1 初始化，第1列进行正交化
        I = np.eye(self.n)
        omega = orth_mat[:, 0] - np.linalg.norm(orth_mat[:, 0]) * I[:, 0]
        omega = omega.reshape(-1, 1)
        Q = I - 2 * np.dot(omega, omega.T) / np.dot(omega.T, omega)
        R = np.dot(Q, orth_mat)
        # 1.2 从第2列开始直到右下方阵为2*2
        for i in range(1, self.n - 1):
            # 每次循环取当前R矩阵的右下(n-i) * (n-i)方阵进行正交化
            sub_mat, I = R[i:, i:], np.eye(self.n - i)
            omega = (sub_mat[:, 0] - np.linalg.norm(sub_mat[:, 0]) *
                     I[:, 0]).reshape(-1, 1)  # 按照公式求解omega
            # 按公式计算右下方阵的正交化矩阵
            Q_i = I - 2 * np.dot(omega, omega.T) / np.dot(omega.T, omega)
            # 将Q_i作为右下方阵， 扩展为n*n矩阵，且其前i个对角线元素为1
            Q_i_expand = np.r_[np.zeros((i, self.n)),
                               np.c_[np.zeros((self.n - i, i)), Q_i]]
            for k in range(i):
                Q_i_expand[k, k] = 1
            R[i:, i:] = np.dot(Q_i, sub_mat)  # 替换原右下角矩阵元素
            Q = np.dot(Q, Q_i_expand)  # 每次右乘正交矩阵Q_i
        return Q, R

    def _givens_rotation_(self, orth_mat):
        """
        吉文斯(Givens)变换方法求解QR分解：通过将原矩阵 A 的主对角线下方的元素都通过Givens旋转置换成0，
        形成上三角矩阵 R，同时左乘的一系列Givens矩阵相乘得到一个正交阵Q。
        :return:
        """
        # 1. 按照Givens变换进行正交化求解QR
        Q, R = np.eye(self.n), np.copy(orth_mat)
        rows, cols = np.tril_indices(self.n, -1, self.n)  # 获得主对角线以下三角矩阵的元素索引
        for row, col in zip(rows, cols):
            if R[row, col]:  # 不为零，则变换
                norm_ = np.linalg.norm([R[col, col], R[row, col]])
                c = R[col, col] / norm_  # cos(theta)
                s = R[row, col] / norm_  # sin(theta)
                # 构造Givens旋转矩阵
                givens_mat = np.eye(self.n)
                givens_mat[[col, row], [col, row]] = c  # 对角为cos
                givens_mat[row, col], givens_mat[col, row] = -s, s  # 反对角为sin
                R = np.dot(givens_mat, R)  # 左乘
                Q = np.dot(Q, givens_mat.T)
        return Q, R

    def show_iteration(self):  # 参考类QROrthogonalDecompositionMatrixEig中方法
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

    def plt_eigenvalues(self):  # 参考类QROrthogonalDecompositionMatrixEig中方法
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
            plt.plot(iter_, eigenvalues[:, i], lw=1.5,
                     label="$\lambda_{%d}=%.8f$" % (i + 1, eigenvalues[-1, i]))
            # plt.plot(iter_, eigenvalues[:, i], ls_[i], lw=1.5, label="$\lambda_{%d}=%.8f$" % (i + 1, eigenvalues[-1, i]))
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$\lambda_k$", fontdict={"fontsize": 18})
        plt.title("上海森伯格矩阵$QR$法求解$\lambda^{*}_{k}$的收敛性", fontdict={"fontsize": 18})
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.legend(frameon=False, fontsize=18, loc="upper right")
        plt.subplot(122)
        plt.semilogy(iter_[1:], self.iter_precision, "*-",
                 label="$\epsilon, \ k = %d$" % len(iter_))
        plt.semilogy(iter_[-1], self.iter_precision[-1], "D", label="$\epsilon = %.10e$" % self.iter_precision[-1])
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$\epsilon = \Vert \lambda^{(k+1)} - \lambda^{(k)} \Vert$", fontdict={"fontsize": 18})
        plt.title("上海森伯格矩阵$QR$法求解$\lambda^{*}_{k}$的精度收敛曲线", fontdict={"fontsize": 18})
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.legend(frameon=False, fontsize=18)
        plt.show()

