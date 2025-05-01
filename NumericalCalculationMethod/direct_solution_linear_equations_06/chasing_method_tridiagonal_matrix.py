# -*- coding: UTF-8 -*-
"""
@file_name: chasing_method_tridiagonal_matrix.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class ChasingMethodTridiagonalMatrix:
    """
    追赶法求解三对角矩阵
    """
    def __init__(self, diag_a, diag_b, diag_c, d_vector, sol_method="gauss"):
        self.a = np.asarray(diag_a, dtype=np.float64)  # 次对角线元素，对角线以下
        self.b = np.asarray(diag_b, dtype=np.float64)  # 主对角线元素
        self.c = np.asarray(diag_c, dtype=np.float64)  # 次对角线元素，对角线以上
        self.n = len(self.b)
        if len(self.a) != self.n - 1 or len(self.c) != self.n - 1:
            raise ValueError("系数矩阵对角线元素维度不匹配.")
        self.d_vector = np.asarray(d_vector, dtype=np.float64)
        if len(self.d_vector) != self.n:
            raise ValueError("右端向量维度与系数矩阵维度不匹配.")
        # 追赶法的求解类型，有高斯消元法gauss，杜利特尔分解doolittle和克劳特crout分解
        self.sol_method = sol_method
        self.x, self.y = None, None  # 线性方程组的解
        self.eps = None  # 验证精度

    def fit_solve(self):
        """
        追赶法求解三对角矩阵
        :return:
        """
        self.y, self.x = np.zeros(self.n), np.zeros(self.n)
        if self.sol_method in ["gauss", "Gauss"]:
            self.x = self._gauss_solve_()
        elif self.sol_method in ["doolittle", "Doolittle"]:
            self._doolittle_solve()
        elif self.sol_method in ["crout", "Crout"]:
            self._crout_solve_()
        else:
            raise ValueError("仅支持Gauss消元、Doolittle分解和Crout分解三种追赶法.")
        return self.x

    def _gauss_solve_(self):
        """
        采用高斯消元法求解三对角矩阵
        :return:
        """
        b, d = np.copy(self.b), np.copy(self.d_vector)
        for k in range(self.n - 1):
            multiplier = - self.a[k] / b[k]  # 行乘子
            # 仅更新对角元素b，c不更新，因为其上一行同列元素为0
            b[k + 1] += self.c[k] * multiplier
            # 右端向量更新，其中d1不更新, 对角线a不更新，求解x用不到
            d[k + 1] += d[k] * multiplier
        # print("b元素：", b)
        # print("d元素：", d)
        self.x[-1] = d[-1] / b[-1]
        for i in range(self.n - 2, -1, -1):
            self.x[i] = (d[i] - self.c[i] * self.x[i + 1]) / b[i]
        # 3. 验证解的精度
        self.eps = self._check_solve_eps_(self.x)
        return self.x

    def _check_solve_eps_(self, x):
        """
        验证解的精度
        :return:
        """
        eps = np.zeros(self.n)
        eps[0] = self.b[0] * x[0] + self.c[0] * x[1] - self.d_vector[0]
        for i in range(1, self.n - 1):
            eps[i] = self.a[i - 1] * x[i - 1] + self.b[i] * x[i] + \
                     self.c[i] * x[i + 1] - self.d_vector[i]
        eps[-1] = self.a[-1] * x[-2] + self.b[-1] * x[-1] - self.d_vector[-1]
        return eps

    def _doolittle_solve(self):
        """
        采用杜利特尔分解法求解三对角矩阵，即A = LU，L为单位下三角矩阵，U为上三角矩阵
        :return:
        """
        self._check_diagonally_dominant_mat_()  # 判断是否为对角占优矩阵
        # 1. 求解L和U分解的元素
        l_, u = np.zeros(self.n - 1), np.zeros(self.n)
        u[0] = self.b[0]
        for i in range(1, self.n):
            l_[i - 1] = self.a[i - 1] / u[i - 1]
            u[i] = self.b[i] - self.c[i - 1] * l_[i - 1]
        print("L元素：", l_)
        print("U元素：", u)
        # 2. 回代过程，追赶法
        self.y[0] = self.d_vector[0]
        for k in range(1, self.n):
            self.y[k] = self.d_vector[k] - l_[k - 1] * self.y[k - 1]
        self.x[-1] = self.y[-1] / u[-1]
        for k in range(self.n - 2, -1, -1):
            self.x[k] = (self.y[k] - self.c[k] * self.x[k + 1]) / u[k]
        # 3. 验证解的精度
        self.eps = self._check_solve_eps_(self.x)
        return self.x

    def _check_diagonally_dominant_mat_(self):
        """
        判断对角占优矩阵
        :return:
        """
        a, b, c = np.abs(self.a), np.abs(self.b), np.abs(self.c)
        if b[0] > c[0] > 0 and b[-1] > a[-1] > 0:
            for i in range(1, self.n - 1):
                if b[i] < a[i - 1] + c[i - 1] or a[i - 1] * c[i - 1] == 0:
                    print("非三对角占优矩阵，用LU分解法的解可能存在较大误差.")
        else:
            print("非三对角占优矩阵，用LU分解法的解可能存在较大误差.")

    def _crout_solve_(self):
        """
        采用克劳特crout分解法求解三对角矩阵，即A = LU，L为下三角矩阵，U为单位上三角矩阵
        :return:
        """
        self._check_diagonally_dominant_mat_()  # 判断是否为对角占优矩阵
        # 求解L和U的元素
        l_, u = np.zeros(self.n), np.zeros(self.n - 1)
        l_[0], u[0] = self.b[0], self.c[0] / self.b[0]
        for i in range(1, self.n - 1):
            l_[i] = self.b[i] - self.a[i - 1] * u[i - 1]
            u[i] = self.c[i] / l_[i]
        l_[-1] = self.b[-1] - self.a[-1] * u[-1]
        print("L元素：", l_)
        print("U元素：", u)
        # 2. 回代求解
        self.y[0] = self.d_vector[0] / l_[0]
        for i in range(1, self.n):
            self.y[i] = (self.d_vector[i] - self.a[i - 1] * self.y[i - 1]) / l_[i]
        self.x[-1] = self.y[-1]
        for i in range(self.n - 2, -1, -1):
            self.x[i] = self.y[i] - u[i] * self.x[i + 1]
        # 3. 验证解的精度
        self.eps = self._check_solve_eps_(self.x)
        return self.x
