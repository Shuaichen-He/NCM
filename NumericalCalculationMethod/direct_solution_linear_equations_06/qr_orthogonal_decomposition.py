# -*- coding: UTF-8 -*-
"""
@file_name: qr_orthogonal_decomposition.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class QROrthogonalDecomposition:
    """
    QR正交分解法求解方程组的解，Q为正交矩阵，R为上三角矩阵
    """

    def __init__(self, A, b, sol_method="schmidt"):
        self.A = np.asarray(A, dtype=np.float64)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("系数矩阵不是方阵，不能用高斯消元法求解！")
        else:
            self.n = self.A.shape[0]  # 矩阵维度
        self.b = np.asarray(b, dtype=np.float64)
        if len(self.b) != self.n:
            raise ValueError("右端向量维度与系数矩阵维度不匹配！")
        # QR分解法的类型，施密特正交分解法schmidt和Householder变换方法
        self.sol_method = sol_method
        self.x = None  # 线性方程组的解
        self.eps = None  # 验证精度
        self.Q, self.R = None, None  # A = QR

    def fit_solve(self):
        """
        QR正交分解法
        :return:
        """
        # 1. 通过不同方法构造正交矩阵Q和上三角矩阵R
        self.Q, self.R = np.copy(self.A), np.zeros((self.n, self.n))
        if self.sol_method in ["Schmidt", "schmidt"]:
            self._schmidt_orthogonal_()
        elif self.sol_method in ["Householder", "householder"]:
            self._householder_transformation_()
        elif self.sol_method in ["Givens", "givens"]:
            self._givens_rotation_()
        else:
            raise ValueError("仅支持Schmidt正交分解法和Householder变换分解法.")
        # 2. 求解线性方程组的解：Rx = Q^T * b
        self.x = self._solve_linear_equations_x_(self.R, self.Q)
        # 3. 验证解的精度度
        self.eps = np.dot(self.A, self.x) - self.b
        return self.x

    def _schmidt_orthogonal_(self):
        """
        施密特正交分解法
        :return:
        """
        self.Q[:, 0] = self.Q[:, 0] / np.linalg.norm(self.Q[:, 0])  # A的第一列正规化
        for i in range(1, self.n):
            for j in range(i):
                # 使A的第i列与前面所有的列正交
                self.Q[:, i] = self.Q[:, i] - np.dot(self.Q[:, i], self.Q[:, j]) * \
                               self.Q[:, j]
            self.Q[:, i] = self.Q[:, i] / np.linalg.norm(self.Q[:, i])
        self.R = np.dot(self.Q.T, self.A)


    def _householder_transformation_(self):
        """
        豪斯霍尔德Householder变换方法求解QR
        :return:
        """
        # 1. 按照householder变换进行正交化求解QR
        # 1.1 初始化，第1列进行正交化
        I = np.eye(self.n)  # 单位矩阵
        omega = self.A[:, [0]] - np.linalg.norm(self.A[:, 0]) * I[:, [0]]  # 保持维度，shape=(n, 1)
        self.Q = I - 2 * np.dot(omega, omega.T) / np.dot(omega.T, omega)
        self.R = np.dot(self.Q, self.A)
        # 1.2 从第2列开始直到右下方阵为2*2
        for i in range(1, self.n - 1):
            # 每次循环取当前R矩阵的右下(n-i) * (n-i)方阵进行正交化
            sub_mat, I = np.copy(self.R[i:, i:]), np.copy(np.eye(self.n - i))
            omega = sub_mat[:, [0]] - np.linalg.norm(sub_mat[:, 0]) * I[:, [0]]  # 按照公式求解omega
            # 按公式计算右下方阵的正交化矩阵
            Q_i = I - 2 * np.dot(omega, omega.T) / np.dot(omega.T, omega)
            # 将Q_i作为右下方阵， 扩展为n*n矩阵，且其前i个对角线元素为1
            Q_i_expand = np.r_[np.zeros((i, self.n)),
                               np.c_[np.zeros((self.n - i, i)), Q_i]]
            for k in range(i):
                Q_i_expand[k, k] = 1
            self.R[i:, i:] = np.dot(Q_i, sub_mat)  # 替换原右下角矩阵元素
            self.Q = np.dot(self.Q, Q_i_expand)  # 每次右乘正交矩阵Q_i

    def _givens_rotation_(self):
        """
        吉文斯(Givens)变换方法求解QR分解：通过将原矩阵 A 的主对角线下方的元素都通过Givens旋转置换成0，
        形成上三角矩阵 R，同时左乘的一系列Givens矩阵相乘得到一个正交阵Q。
        :return:
        """
        self.Q, self.R = np.eye(self.n), np.copy(self.A)
        rows, cols = np.tril_indices(self.n, -1, self.n)  # 获得主对角线以下三角矩阵的元素索引
        for row, col in zip(rows, cols):
            if self.R[row, col]:  # 不为零，则变换
                norm_ = np.linalg.norm([self.R[col, col], self.R[row, col]])
                c, s = self.R[col, col] / norm_, self.R[row, col] / norm_  # 分别对应cos(theta)，sin(theta)
                givens_mat = np.eye(self.n)  # 构造Givens旋转矩阵
                givens_mat[[col, row], [col, row]] = c  # 对角为cos
                givens_mat[row, col], givens_mat[col, row] = -s, s  # 反对角为sin
                self.R = np.dot(givens_mat, self.R)  # 不断左乘Givens旋转矩阵
                self.Q = np.dot(self.Q, givens_mat.T)  # 不断右乘转换矩阵的转置

    def _solve_linear_equations_x_(self, R, Q):
        """
        求解线性方程组的解
        :param R: 上三角矩阵
        :param Q: 正交矩阵
        :return:
        """
        b = np.dot(Q.T, self.b)
        x = np.zeros(self.n)
        x[-1] = b[-1] / self.R[-1, -1]
        for i in range(self.n - 2, -1, -1):
            x[i] = (b[i] - np.dot(R[i, i:], x[i:])) / R[i, i]
        return x
