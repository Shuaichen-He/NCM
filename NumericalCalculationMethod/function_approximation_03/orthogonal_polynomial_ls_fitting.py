# -*- coding: UTF-8 -*-
"""
@file:orthogonal_polynomial_ls_fitting.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from util_font import *


class OrthogonalPolynomialLSFitting:
    """
    正交多项式最小二乘拟合：p(x) = a0*P0(x) + a1*P1(x) + ... + an*Pn(x)
    """
    fit_polynomial = None  # 正交多项式最小二乘拟合的多项式
    poly_coefficient, polynomial_orders = None, None  # 多项式系数和各项阶次
    fit_error, mse = None, np.infty  # 拟合误差向量, 拟合均方误差mse

    def __init__(self, x, y, k, w=None):  # 参考最小二乘多项式拟合
        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.k = k  # 正交多项式最小二乘拟合的最高阶次
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

    def fit_orthogonal_poly(self):
        """
        正交多项式最小二乘曲线拟合核心算法
        :return:
        """
        t = sympy.Symbol("t")  # 正交多项式的符号变量
        self.poly_coefficient = np.zeros(self.k + 1)  # 正交多项式系数a
        px = sympy.Matrix.zeros(self.k + 1, 1)  # 带权正交多项式
        sw = np.sum(self.w)  # 实验数据的权重和
        wy = self.w * self.y  # 权重w_i与y_i的对应元素相乘
        wx = self.w * self.x  # 权重w_i与x_i的对应元素相乘
        # 1. 构造正交多项式的前两项
        alpha = wx.sum() / sw  # alpha_1
        px[0], px[1] = 0 * t + 1, t - alpha  # 正交多项式前两项
        self.poly_coefficient[0] = wy.sum() / sw  # a0
        p_x0 = np.ones(self.n)  # 正交多项式的函数值P(x_i)
        p_x1 = sympy.lambdify(t, px[1], "numpy")(self.x)  # P(x_{i+1})
        self.poly_coefficient[1] = (wy * p_x1).sum() / \
                                   (self.w * p_x1 ** 2).sum()  # a1
        wp = (self.w * p_x1 ** 2).sum()  # alpha和beta子项
        # 2. 从第三项开始，逐步递推构造
        for k in range(2, self.k + 1):
            alpha = (wx * p_x1 ** 2).sum() / wp  # 参数公式
            beta = wp / (self.w * p_x0 ** 2).sum()  # 参数公式
            # 基函数的递推公式
            px[k] = sympy.simplify((t - alpha) * px[k - 1] - beta * px[k - 2])
            p_x0 = np.copy(p_x1)  # P(xi)值的更替
            p_x1 = sympy.lambdify(t, px[k], "numpy")(self.x)  # 新值
            wp = (self.w * p_x1 ** 2).sum()  # alpha和beta子项
            self.poly_coefficient[k] = (wy * p_x1).sum() / wp  # 系数
        # 3. 正交多项式的构造
        self.fit_polynomial = self.poly_coefficient[0] * px[0]
        for k in range(1, self.k + 1):
            self.fit_polynomial += self.poly_coefficient[k] * px[k]
        polynomial = sympy.Poly(self.fit_polynomial, t)
        self.polynomial_orders = polynomial.monoms()[::-1]  # 阶次，从低到高
        # 4. 计算误差
        self.cal_fit_error()

    def predict_x0(self, x0):  # 参考最小二乘多项式拟合
        """
        求解逼近多项式给定点的值
        :return:
        """
        t = self.fit_polynomial.free_symbols.pop()
        fit_poly = sympy.lambdify(t, self.fit_polynomial, "numpy")
        return fit_poly(x0)

    def cal_fit_error(self):  # 参考最小二乘多项式拟合
        """
        计算拟合的误差和均方根误差
        :return:
        """
        self.fit_error = self.y - self.predict_x0(self.x)
        self.mse = np.mean(self.fit_error ** 2)
        return self.mse

    def plt_curve_fit(self, is_show=True):  # 参考最小二乘多项式拟合
        """
        可视化
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
        plt.title("正交多项式曲线拟合$(MSE = %.2e)$" % self.mse, fontdict={"fontsize": 18})
        if is_show:
            plt.show()
