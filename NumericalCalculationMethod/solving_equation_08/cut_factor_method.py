# -*- coding: UTF-8 -*-
"""
@file_name: cut_factor_method.py
@time: 2022-11-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import cmath
import matplotlib.pyplot as plt


class CutFactorMethod:
    """
    劈因子法，求实系数代数方程的两个实根或一对共轭复根
    """

    def __init__(self, P, p0, q0, eps=1e-8, max_iter=500):
        self.p_c = np.asarray(P, np.float64)  # 实系数代数方程的系数，从高到底
        self.p_c = self.p_c / self.p_c[0]  # 转换最高次幂系数为1
        self.P = self.p_c[1:]  # 由于求解c、b、r和R不用最高次幂，故截去
        self.p0, self.q0 = p0, q0  # 二次因子的一次项系数和二次项系数
        self.eps, self.max_iter = eps, max_iter  # 精度要求和最大迭代次数
        self.omega_x = None  # 最后二次因子：x ** 2 + p * x + q
        self.root = None  # 两个实根或一对共轭复根
        self.precision = np.array([1, 1], dtype=np.float64)  # 根的精度

    def fit_cut_factor(self):
        """
        核心算法：劈因子法，逐步迭代逼近二次因子
        :return:
        """
        n = len(self.P)  # 获得实系数方程的系数数量
        if n <= 3:
            raise ValueError("仅限于高于3次幂的实系数代数多项式。")
        b, c = np.zeros(n), np.zeros(n - 1)  # 初始商式的系数
        p, q = self.p0, self.q0  # 初始二次因子的一次项系数p和常数项q
        iter_, tol = 0, np.infty  # 初始化迭代次数
        while tol > self.eps and iter_ < self.max_iter:
            b[0], b[1], c[0], c[1] = 0, 1, 0, 1  # 每次迭代的初始项
            for i in range(2, n):
                b[i] = self.P[i - 2] - p * b[i - 1] - q * b[i - 2]  # 商式Q(x)的系数
            r1, r2 = self.P[-2] - p * b[-1] - q * b[-2], self.P[-1] - q * b[-1]  # 余项R(x)
            for i in range(2, n - 1):
                c[i] = b[i] - p * c[i - 1] - q * c[i - 2]  # 商式M(x)的系数
            R11, R12 = b[-1] - p * c[-1] - q * c[-2], -q * c[-1]  # 余项R1(x)的系数
            R21, R22 = b[-2] - p * c[-2] - q * c[-3], b[-1] - q * c[-2]  # 余项R2(x)的系数
            tmp = R21 * R22 - R11 * R12  # 求解线性方程组，u和v变量的分母
            if np.abs(tmp) < 1e-25:
                break
            u, v = (r2 * R21 - r1 * R12) / tmp, (r1 * R22 - r2 * R11) / tmp  # 获得改变量
            p, q = p + u, q + v  # 更新二次因子系数
            iter_, tol = iter_ + 1, max(np.abs([u, v]))  # 改变量最大值作为精度
        self.omega_x = np.array([1, p, q])  # 构造二次因子
        self._solve_roots(p, q, n)  # 求实系数代数方程的两个实根或一对共轭复根，并验证精度

    def _solve_roots(self, p, q, n):
        """
        求实系数代数方程的两个实根或一对共轭复根，并验证精度
        """
        term = p ** 2 - 4 * q  # 二次方程根公式：delta = b^2 - 4 *a * c
        if term >= 0:  # 两个实数根
            self.root = 0.5 * np.array([-p + np.sqrt(term), -p - np.sqrt(term)])
            pow_values = np.zeros(n + 1, dtype=np.float64)  # 存储幂次项的值
        else:  # 一对共轭复根
            self.root = 0.5 * np.array([-p + cmath.sqrt(term), -p - cmath.sqrt(term)])
            self.precision = np.zeros(2, dtype="complex_")  # 复数类型，用于存储复数计算
            pow_values = np.zeros(n + 1, dtype="complex_")  # 存储幂次项的值
        try:
            for i in range(2):
                for k in range(n + 1):
                    pow_values[k] = self.root[i] ** (n - k)  # 幂次项的值
                self.precision[i] = np.dot(self.p_c, pow_values)  # 方程的精度
        except OverflowError:
            return
