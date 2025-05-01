# -*- coding: UTF-8 -*-
"""
@file:least_square_curve_fitting.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import sympy
from direct_solution_linear_equations_06.square_root_decomposition \
    import SquareRootDecompositionAlgorithm  # 第6章 线性方程组的平方根分解法求解
from util_font import *


class LeastSquarePolynomialCurveFitting:
    """
    多项式曲线拟合，线性最小二乘拟合同样适用，k阶次为1即可。
    """
    fit_poly = None  # 曲线拟合的多项式
    poly_coefficient = None  # 曲线拟合多项式的系数
    polynomial_orders = None  # 曲线拟合多项式各项阶次
    fit_error = None  # 拟合误差向量
    mse = np.infty  # 拟合均方误差

    def __init__(self, x, y, k, w=None):
        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.k = k  # 多项式曲线拟合的最高阶次
        if len(self.x) != len(self.y):
            raise ValueError("数据点坐标不匹配！")
        else:
            self.n = len(self.x)
        if w is None:
            self.w = np.ones(self.n)  # 初始权重为1
        else:
            self.w = np.asarray(w, dtype=np.float64)
            if len(self.w) != self.n:
                raise ValueError("权重系数维度与坐标点维度不匹配！")

    def fit_ls_curve(self):
        """
        最小二乘多项式曲线拟合核心算法
        :return:
        """
        c = np.zeros(2 * self.k + 1)  # 系数矩阵2n+1个不同的值
        b = np.zeros(self.k + 1)
        for i in range(2 * self.k + 1):
            c[i] = np.dot(self.w, np.power(self.x, i))
            if i < self.k + 1:
                b[i] = np.dot(self.w, self.y * np.power(self.x, i))
        C = np.zeros((self.k + 1, self.k + 1))  # 构造对称正定系数矩阵
        C[0, :] = c[:self.k + 1]
        for k in range(1, self.k + 1):
            C[k, :] = c[k: self.k + k + 1]
        # 采用改进的平方根分解法求解，也可采用Householder或Givens正交分解法
        srd = SquareRootDecompositionAlgorithm(C, b)
        srd.fit_solve()
        self.poly_coefficient = srd.x
        # self.poly_coefficient = np.linalg.solve(C, b)  # 求解拟合系数

        # 如下代码构造拟合多项式
        t = sympy.Symbol("t")
        self.fit_poly = self.poly_coefficient[0] * 1
        for p in range(1, self.k + 1):
            px = np.power(t, p)  # p(x)多项式各项
            self.fit_poly += self.poly_coefficient[p] * px
        polynomial = sympy.Poly(self.fit_poly, t)
        self.polynomial_orders = polynomial.monoms()  # 阶次，从低到高
        self.cal_fit_error()  # 计算误差

    def predict_x0(self, x0):
        """
        计算给定数值的拟合多项式值
        :param x0: 给定的数值序列
        :return:
        """
        t = self.fit_poly.free_symbols.pop()
        fit_poly = sympy.lambdify(t, self.fit_poly)
        return fit_poly(x0)

    def cal_fit_error(self):
        """
        计算拟合的误差和均方误差
        :return:
        """
        self.fit_error = self.y - self.predict_x0(self.x)  # 真值 - 预测值
        self.mse = np.mean(self.fit_error ** 2)
        return self.mse

    def plt_curve_fit(self, is_show=True):
        """
        可视化最小二乘多项式曲线拟合
        :return:
        """
        xi = np.linspace(min(self.x), max(self.x), 100, endpoint=True)
        yi = self.predict_x0(xi)
        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(xi, yi, "k-", lw=1.5, label="$p(x):\ k=%d$" % self.k)
        plt.plot(self.x, self.y, "ro", label="$(x_k, y_k)$")
        plt.grid(ls=":")
        plt.legend(frameon=False, fontsize=18)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$y$", fontdict={"fontsize": 18})
        plt.title("最小二乘多项式曲线拟合$(MSE = %.2e)$" % self.mse, fontdict={"fontsize": 18})
        if is_show:
            plt.show()
