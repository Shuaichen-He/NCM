# -*- coding: UTF-8 -*-
"""
@file_name: doolittle_decomposition_lu.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class DoolittleTriangularDecompositionLU:
    """
    杜利特尔分解：矩阵分解A=LU，求解Ly=b得y，求解Ux=y得x
    """

    def __init__(self, A, b, sol_method="doolittle"):
        self.A = np.asarray(A, dtype=np.float64)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("系数矩阵不是方阵，不能用高斯消元法求解！")
        else:
            self.n = self.A.shape[0]  # 矩阵维度
        self.b = np.asarray(b, dtype=np.float64)
        if len(self.b) != self.n:
            raise ValueError("右端向量维度与系数矩阵维度不匹配！")
        self.sol_method = sol_method  # LU分解法的类型，不选主元doolittle和选主元pivot
        self.x, self.y = None, None  # 线性方程组的解
        self.eps = None  # 验证精度
        self.L, self.U, self.P = None, None, None  # A = LU或PA=LU
        self.inverse_matrix = None  # A^(-1) = U^(-1) * L^(-1) * P逆矩阵

    def fit_solve(self):
        """
        杜利特尔分解算法与选主元的三角分解法
        :return:
        """
        # print("系数矩阵A的条件数为：", np.linalg.cond(self.A))
        self.L, self.U = np.eye(self.n), np.zeros((self.n, self.n))
        self.y, self.x = np.zeros(self.n), np.zeros(self.n)
        if self.sol_method == "doolittle":  # 不选主元
            self.x = self._solve_doolittle_()
        elif self.sol_method == "pivot":  # 选主元
            self.x = self._solve_pivot_doolittle_()
        else:
            raise ValueError("仅适合doolittle LU分解法和选主元LU分解法.")
        return self.x

    def _solve_doolittle_(self):
        """
        不选主元的三角分解法，即杜利特尔分解法。
        :return:
        """
        # 1. L和U得分解过程
        self.U[0, :] = self.A[0, :]  # U的第一行与系数矩阵A的第一行相同
        if self.U[0, 0] == 0:
            raise ValueError("不适宜用Doolittle-LU分解.")
        self.L[:, 0] = self.A[:, 0] / self.U[0, 0]  # 求L的第一列
        for r in range(1, self.n):  # 每次循环计算U第r行和L的第r列
            for i in range(r, self.n):  # 列在变化, 求第r行
                # U的计算公式, 由于Python索引为左闭右开, 故右索引r未减一操作
                self.U[r, i] = self.A[r, i] - np.dot(self.L[r, :r], self.U[:r, i])
            for i in range(r + 1, self.n):  # 行在变化, 求L第r列
                if self.U[r, r] == 0:
                    raise ValueError("不适宜用Doolittle-LU分解.")
                # L的计算公式
                self.L[i, r] = (self.A[i, r] -
                                np.dot(self.L[i, :r], self.U[:r, r])) / self.U[r, r]
        self.x = self._back_substitution_process_(self.b)  # 2. 回代求解
        self.eps = np.dot(self.A, self.x) - self.b  # 3. 验证解的精度
        return self.x

    def _back_substitution_process_(self, b):
        """
        LU分解回代过程
        :return:
        """
        self.y[0] = b[0]
        for i in range(1, self.n):
            self.y[i] = b[i] - np.dot(self.L[i, :i], self.y[:i])
        self.x[-1] = self.y[-1] / self.U[-1, -1]
        for i in range(self.n - 2, -1, -1):
            self.x[i] = (self.y[i] - np.dot(self.U[i, i + 1:], self.x[i + 1:])) / \
                        self.U[i, i]
        return self.x

    def _column_pivot_swap_(self, k, idx):
        """
        列主元选定后，交换系数矩阵的行、置换矩阵P的行以及L的行
        :param k: 当前行索引
        :param idx: 列主元所在的行索引，且k != idx
        :return:
        """
        commutator, c_p = np.copy(self.A[k, :]), np.copy(self.P[k, :])
        self.A[k, :], self.P[k, :] = np.copy(self.A[idx, :]), np.copy(self.P[idx, :])
        self.A[idx, :], self.P[idx, :] = np.copy(commutator), np.copy(c_p)
        L = np.copy(self.L[k, :k])  # 当前单位元1不参与变换
        self.L[k, :k] = np.copy(self.L[idx, :k])
        self.L[idx, :k] = np.copy(L)

    def _solve_pivot_doolittle_(self):
        """
        选主元三角分解法，PA=LU
        :return:
        """
        # 1. 第一行第一列元素选主元，初始化
        self.P = np.eye(self.n)  # 初值置换矩阵
        idx = np.argmax(np.abs(self.A[:, 0]))  # 第1列最大元的行索引
        if idx != 0:
            self._column_pivot_swap_(0, idx)  # 当前矩阵A的第一个元素为列最大值
        # 2. L和U得分解过程中U的第一行和L的第一列
        self.U[0, :] = self.A[0, :]  # U的第一行与系数矩阵A的第一行相同
        if self.U[0, 0] == 0:
            raise ValueError("不适宜用Doolittle-LU分解.")
        self.L[:, 0] = self.A[:, 0] / self.U[0, 0]  # 求L的第一列
        # 3. 每次循环计算U第r行和L的第r列, 并对第r行第r列右下方阵选主元
        for r in range(1, self.n):
            # 3.1 选列主元, 从第r行第r列右下方阵中, 第r列选主元
            s = []  # 标记第r列的U值, 以便列主元选取
            for i in range(r, self.n):
                s.append(self.A[i, r] - np.dot(self.L[r, :r], self.U[:r, i]))
            idx = np.argmax(np.abs(s))  # 当前第r列的最大绝对值U索引
            if idx + r != r:  # 非当前行, 交换
                self._column_pivot_swap_(r, idx + r)
            # 3.2 交换后, 求解第r行U和第r列L
            for i in range(r, self.n):  # 列在变化, 求第r行
                self.U[r, i] = self.A[r, i] - np.dot(self.L[r, :r], self.U[:r, i])  # U的计算公式
            for i in range(r + 1, self.n):  # 行在变化, 求L第r列
                if self.U[r, r] == 0:
                    raise ValueError("不适宜用Doolittle-LU分解.")
                # L的计算公式
                self.L[i, r] = (self.A[i, r] -
                                np.dot(self.L[i, :r], self.U[:r, r])) / self.U[r, r]
        # 4. 回代求解
        permutation_b = np.dot(self.P, self.b)  # 置换矩阵, 重排右端向量
        self.x = self._back_substitution_process_(permutation_b)
        self.eps = np.dot(self.A, self.x) - permutation_b  # 5. 验证解的精度
        # 6. 逆矩阵:  A^(-1) = U^(-1) * L^(-1) * P
        self.inverse_matrix = np.dot(np.dot(np.linalg.inv(self.U),
                                            np.linalg.inv(self.L)), self.P)
        return self.x
