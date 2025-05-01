# -*- coding: UTF-8 -*-
"""
@file:guass_elimination_algorithm.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class GaussianEliminationAlgorithm:
    """
    高斯消元法，包括sequential顺序消元，列主元，全主元和高斯—约当消去法
    其中高斯—约当可获得逆矩阵
    """

    def __init__(self, A, b, sol_method="column"):
        self.A = np.asarray(A, dtype=np.float64)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("系数矩阵不是方阵，不能用高斯消元法求解！")
        else:
            self.n = self.A.shape[0]  # 矩阵维度
        self.b = np.asarray(b, dtype=np.float64)
        if len(self.b) != self.n:
            raise ValueError("右端向量维度与系数矩阵维度不匹配！")
        self.augmented_matrix = np.c_[self.A, self.b]  # 增广矩阵
        self.sol_method = sol_method  # 高斯消元的方法类型，默认列主元
        self.x = None  # 线性方程组的解
        self.eps = None  # 验证精度
        self.jordan_inverse_matrix = None  # 高斯约当消元获得的逆矩阵

    def fit_solve(self):
        """
        求解过程
        :return:
        """
        if self.sol_method == "sequential":
            self._solve_sequential_()
        elif self.sol_method == "column":
            self._solve_column_pivot_element_()
        elif self.sol_method == "complete":
            self._solve_complete_pivot_element_()
        elif self.sol_method == "jordan":
            self._solve_jordan_()
        else:
            raise ValueError("仅支持顺序sequential、列主元column、全主元complete、约当jordan四种高斯消元法.")

    def _elimination_process(self, i, k):
        """
        高斯消元核心公式
        :param i: 当前行
        :param k: 当前列
        :return:
        """
        if self.augmented_matrix[k, k] == 0:
            raise ValueError("系数矩阵不满足高斯顺序消元法求解！")
        # 每行的乘子
        multiplier = self.augmented_matrix[i, k] / self.augmented_matrix[k, k]
        # 第i行元素消元更新，包括右端向量
        self.augmented_matrix[i, k:] -= multiplier * self.augmented_matrix[k, k:]

    def _back_substitution_process_(self):
        """
        高斯回代过程
        :return:
        """
        x = np.zeros(self.n)  # 线性方程组的解
        for k in range(self.n - 1, -1, -1):
            sum_ = np.dot(self.augmented_matrix[k, k + 1:self.n], x[k + 1:self.n])
            x[k] = (self.augmented_matrix[k, -1] - sum_) / self.augmented_matrix[k, k]
        return x

    def _solve_sequential_(self):
        """
        高斯顺序消元法求解
        :return:
        """
        for k in range(self.n - 1):  # 共需消元的行数为n-1行
            for i in range(k + 1, self.n):  # 从下一行开始消元
                self._elimination_process(i, k)  # 消元核心公式
        self.x = self._back_substitution_process_()  # 回代过程
        self.eps = np.dot(self.A, self.x) - self.b  # 验证解的精度

    def _solve_column_pivot_element_(self):
        """
        列主元高斯消去法求解
        :return:
        """
        for k in range(self.n - 1):  # 共需消元的行数为n-1行
            idx = np.argmax(np.abs(self.augmented_matrix[k:, k]))  # 当前列最大元的行索引
            # print("当前列主元：", self.augmented_matrix[idx + k, k])
            # 由于查找列主元是从当前k行开始到最后一行，故索引idx + k，列主元为当前行，则idx为0
            if idx + k != k:  # 不为当前行，则交换使之成为列最大主元
                commutator = np.copy(self.augmented_matrix[k, :])  # 拷贝，不可赋值交换
                self.augmented_matrix[k, :] = np.copy(self.augmented_matrix[idx + k, :])
                self.augmented_matrix[idx + k, :] = np.copy(commutator)
            for i in range(k + 1, self.n):  # 从下一行开始消元
                self._elimination_process(i, k)  # 高斯消元核心公式
        self.x = self._back_substitution_process_()  # 回代过程
        self.eps = np.dot(self.A, self.x) - self.b  # 验证解的精度

    def _solve_complete_pivot_element_(self):
        """
        全主元高斯消去法求解
        :return:
        """
        self.x = np.zeros(self.n)  # 线性方程组的解
        # 交换的列索引，以便解顺序排序，初始化为未交换之前的顺序
        column_index = np.linspace(0, self.n - 1, self.n, dtype=np.int64)
        for k in range(self.n - 1):  # 共需消元的行数为n-1行
            max_x = np.max(np.abs(self.augmented_matrix[k:, k:-1]))  # 当前小方阵中绝对值最大元素
            id_r, id_c = np.where(np.abs(self.augmented_matrix[k:, k:-1]) == max_x)  # 行列索引
            id_r, id_c = int(id_r[0]), int(id_c[0])  # 防止同行列出现多个相同值，若存在多个，则只选择第一个
            print("当前全主元：", self.augmented_matrix[id_r + k, id_c + k],
                  "行列索引：", [id_r + k, id_c + k])
            # 由于查找全主元是从当前k行k列开始的右下方阵，故索引idx + k
            if id_r + k != k:  # 不为当前行，则交换使之称为列最大主元
                commutator_r = np.copy(self.augmented_matrix[k, :])  # 拷贝，不可赋值交换
                self.augmented_matrix[k, :] = \
                    np.copy(self.augmented_matrix[id_r + k, :])
                self.augmented_matrix[id_r + k, :] = np.copy(commutator_r)
            if id_c + k != k:  # 不为当前列，则交换使之称为行最大主元
                pos = column_index[k]  # 当前原有列索引
                column_index[k] = id_c + k  # 新的需交换的列索引
                column_index[id_c + k] = pos  # 列交换
                commutator_c = np.copy(self.augmented_matrix[:, k])  # 拷贝，不可赋值交换
                self.augmented_matrix[:, k] = \
                    np.copy(self.augmented_matrix[:, id_c + k])
                self.augmented_matrix[:, id_c + k] = np.copy(commutator_c)
            for i in range(k + 1, self.n):  # 从下一行开始消元
                self._elimination_process(i, k)  # 高斯消元核心公式
        solve_x = self._back_substitution_process_()  # 回代过程
        print(solve_x)
        # 按照列交换的顺序逐个存储解
        for k in range(self.n):
            for j in range(self.n):
                if k == column_index[j]:
                    self.x[k] = solve_x[j]
                    break
        self.eps = np.dot(self.A, self.x) - self.b  # 验证解的精度

    def _solve_jordan_(self):
        """
        高斯约当消元法，并结合列主元求解，并求逆矩阵
        :return:
        """
        self.augmented_matrix = np.c_[self.augmented_matrix, np.eye(self.n)]  # 增广矩阵
        for k in range(self.n):  # 每行都要轮流处理
            idx = np.argmax(np.abs(self.augmented_matrix[k:, k]))  # 当前列最大元的行索引
            if idx + k != k:  # 不为当前行，则交换使之称为列最大主元
                commutator = np.copy(self.augmented_matrix[k, :])  # 拷贝，不可赋值交换
                self.augmented_matrix[k, :] = np.copy(self.augmented_matrix[idx + k, :])
                self.augmented_matrix[idx + k, :] = np.copy(commutator)
            if self.augmented_matrix[k, k] == 0:
                raise ValueError("系数矩阵不满足高斯—约当消元法.")
            # 当前行都处于对角线元素，对角线元素变为1，当前行其余元素都除于对角线元素
            self.augmented_matrix[k, :] /= self.augmented_matrix[k, k]
            # 消元过程，即当前k行的上下各行乘以k行对角线乘子（负数） + 各元素
            for i in range(self.n):
                if i != k:  # 当前行不需要消元
                    multiplier = -1.0 * self.augmented_matrix[i, k]  # 乘子
                    self.augmented_matrix[i, :] = self.augmented_matrix[i, :] + \
                                                  multiplier * self.augmented_matrix[k, :]
        self.x = self.augmented_matrix[:, self.n]  # 最后一列即为解
        self.jordan_inverse_matrix = self.augmented_matrix[:, self.n + 1:]  # 逆矩阵
        self.augmented_matrix = self.augmented_matrix[:, :self.n + 1]
        self.eps = np.dot(self.A, self.x) - self.b  # 验证解的精度
