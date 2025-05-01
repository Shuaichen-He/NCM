# -*- coding: UTF-8 -*-
"""
@file_name: trigonometric_polynomial_appr.py
@IDE:PyCharm   Python:3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from util_font import *


class TrigonometricPolynomialApproximation:
    """
    当被逼近函数为周期函数时，用代数多项式逼近效率不高，而且误差较大，这时用三角多项式来逼近是较好的选择。
    三角多项式逼近即傅里叶逼近，任一周期函数都可以展开为傅里叶级数，通过选取有限的展开级数，就可以达到所
    需精度的逼近效果。
    """

    def __init__(self, y, interval, fun=None):
        # 被逼近函数fun，被逼近的离散数值y，逼近区间self.a, self.b = interval[0], interval[1]
        self.fun = fun  # 被逼近函数，可以为None，如果提供，可度量逼近性能
        self.y = np.asarray(y, dtype=np.float64)  # 被逼近的离散数值
        self.a, self.b = interval[0], interval[1]  # 逼近区间
        self.Ak, self.Bk = None, None  # 展开后的余弦项、正弦项系数
        self.approximation_poly = None  # 逼近的三角多项式

    def fit_approximation(self):
        """
        核心算法：三角多项式插值逼近
        :return:
        """
        t = sympy.Symbol("t")
        n = len(self.y)  # 离散数据点个数
        m = n // 2  # ak系数的个数
        self.Ak = np.zeros(m + 1)
        self.approximation_poly = 0.0
        idx = np.linspace(0, n, n, endpoint=False, dtype=np.int64)
        if np.mod(n, 2) == 0:  # 偶数个数
            self.Bk = np.zeros(m - 1)
            for k in range(m + 1):
                self.Ak[k] = np.dot(self.y, np.cos(np.pi * idx * k / m))
                if k == 0 or k == m:  # 第一个值a0和最后一个值am特殊处理
                    self.Ak[k] = self.Ak[k] / (2 * m) * (-1) ** k
                else:
                    self.Ak[k] = self.Ak[k] / m * (-1) ** k
                self.approximation_poly += self.Ak[k] * sympy.cos(k * t)
            for k in range(1, m):
                self.Bk[k - 1] = np.dot(self.y, np.sin(np.pi * idx * k / m)) / \
                                 m * (-1) ** k
                self.approximation_poly += self.Bk[k - 1] * sympy.sin(k * t)
        else:  # 奇数个数
            self.Bk = np.zeros(m)
            for k in range(m + 1):
                self.Ak[k] = np.dot(self.y, np.cos(np.pi * idx * k * 2 / (2 * m + 1)))
                if k == 0:
                    self.Ak[k] = self.Ak[k] * 1 / (2 * m + 1) * (-1) ** k
                else:
                    self.Ak[k] = self.Ak[k] * 2 / (2 * m + 1) * (-1) ** k
                self.approximation_poly += self.Ak[k] * sympy.cos(k * t)
            for k in range(1, m + 1):
                sv = np.sin(np.pi * idx * k * 2 / (2 * m + 1))  # Bk子项
                self.Bk[k - 1] = np.dot(self.y, sv) * 2 / (2 * m + 1) * (-1) ** k
                self.approximation_poly += self.Bk[k - 1] * sympy.sin(k * t)

    def predict_x0(self, x0):
        """
        求解三角插值逼近在给定点的值
        :return:
        """
        t = self.approximation_poly.free_symbols.pop()
        approximation_poly = sympy.lambdify(t, self.approximation_poly)  # 转换成lambda函数
        # 区间变换，[a, b] -->[-pi, pi]
        x0 = np.asarray(x0, dtype=np.float64)
        xi = (x0 - (self.a + self.b) / 2) * 2 / (self.b - self.a) * np.pi
        y0 = approximation_poly(xi)
        return y0

    def plt_approximate(self, x0=None, y0=None, is_show=True, is_fh_marker=False):
        """
        可视化函数和三角插值逼近函数
        :return:
        """
        if is_show:
            plt.figure(figsize=(7, 5))

        if self.fun is not None:
            xi = np.linspace(self.a, self.b, 200)
            y_true = self.fun(xi)
            yi_hat = self.predict_x0(xi)  # 求解模拟点
            plt.plot(xi, yi_hat, "r-", lw=2, label="$p(x)$三角逼近")
            if is_fh_marker:
                xi = self.a + np.random.rand(50) * (self.b - self.a)
                xi = np.array(sorted(xi))  # list-->ndarray，升序排列
                y_true_ = self.fun(xi)
                plt.plot(xi, y_true_, "k*", label="$f(x_k), \ x_k \sim U(a, b)$")
            else:
                plt.plot(xi, y_true, "k--", lw=2, label="$f(x)$")
            mae = np.mean(np.abs(yi_hat - y_true))
            print("平均绝对值误差：%.10e" % mae)
            print("最大绝对值误差：%.10e" % np.max(np.abs(y_true - yi_hat)))
            plt.title("三角多项式逼近$(MAE=%.2e, n_{nodes}=%d)$" % (mae, len(self.y)), fontdict={"fontsize": 18})
            plt.ylabel(r"$f(x) \quad / \quad p(x)$", fontdict={"fontsize": 18})
            plt.legend(frameon=False, fontsize=18)
        elif x0 is not None and y0 is not None:
            xi = np.linspace(self.a, self.b, 200)
            yi = self.predict_x0(xi)  # 求傅里叶逼近的值
            plt.title("最小二乘三角多项式逼近函数曲线 ", fontdict={"fontsize": 18})
            plt.ylabel(r"$p(x)$", fontdict={"fontsize": 18})
            plt.plot(xi, yi, "k-", lw=1.5, label="$p(x)$三角逼近")
            plt.plot(x0, y0, "ro", label="$(x_k, y_k)$")
            plt.legend(frameon=False, fontsize=17, loc="upper left")
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
