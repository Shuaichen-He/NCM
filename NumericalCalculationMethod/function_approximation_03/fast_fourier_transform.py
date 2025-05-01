# -*- coding: UTF-8 -*-
"""
@file_name: fast_fourier_transform.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from function_approximation_03.trigonometric_polynomial_appr import TrigonometricPolynomialApproximation
from util_font import *


class FastFourierTransformApproximation(TrigonometricPolynomialApproximation):
    """
    快速傅里叶变换逼近算法，继承三角多项式逼近类TrigonometricPolynomialApproximation
    """

    def __init__(self, y, interval, fun=None):
        TrigonometricPolynomialApproximation.__init__(self, y, interval, fun)
        self.n, self.m = len(y), int(len(y) / 2)  # 离散数据的个数n
        if np.ceil(np.log2(self.n)) != np.log2(self.n):
            raise ValueError("离散数据点应满足2 ** p")
        self.p = int(np.log2(self.n))  # N = 2 ** p

    def fit_fourier(self):
        """
        核心算法：快速傅里叶变换逼近
        :return:
        """
        t = sympy.Symbol("t")  # 符号变量
        omega = np.exp(1j * 2 * np.pi / self.n)  # w_N
        W = omega ** np.arange(0, self.m)  # 当N = 2 ** p时，只有N/2个不同的值
        A1, A2 = np.asarray(self.y, dtype=complex), np.zeros(self.n, dtype=complex)  # 初始，表示A_(q-1)和Aq
        for q in range(1, self.p + 1):  # 逐次计算
            i_2, i_3 = 2 ** (self.p - 1), 2 ** (q - 1)  # 索引下标
            if np.mod(q, 2) == 1:  # 奇数
                for k in range(2 ** (self.p - q)):
                    for j in range(2 ** (q - 1)):
                        i_0, i_1 = k * 2 ** q + j, k * 2 ** (q - 1) + j  # 索引下标
                        A2[i_0] = A1[i_1] + A1[i_1 + i_2]
                        A2[i_0 + i_3] = (A1[i_1] - A1[i_1 + i_2]) * W[k * i_3]
            else:  # 偶数
                for k in range(2 ** (self.p - q)):
                    for j in range(2 ** (q - 1)):
                        i_0, i_1 = k * 2 ** q + j, k * 2 ** (q - 1) + j  # 索引下标
                        A1[i_0] = A2[i_1] + A2[i_1 + i_2]
                        A1[i_0 + i_3] = (A2[i_1] - A2[i_1 + i_2]) * W[k * i_3]
        C = np.copy(A1) if np.mod(self.p, 2) == 0 else np.copy(A2)  # 复数系数
        t_e = np.exp(-1j * np.pi * np.arange(0, self.m + 1))
        self.Ak = np.real(C[:self.m + 1] * t_e) / self.m  # 余弦系数
        self.Bk = np.imag(C[:self.m + 1] * t_e) / self.m  # 正弦系数
        self.approximation_poly = self.Ak[0] / 2  # 构造三角逼近多项式，第一项
        for k in range(1, self.m + 1):  # 构造其他项
            self.approximation_poly += self.Ak[k] * sympy.cos(k * t) + \
                                       self.Bk[k] * sympy.sin(k * t)

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
            plt.title("$FFT$逼近函数$(MAE=%.2e, n_{nodes}=%d)$" % (mae, len(self.y)), fontdict={"fontsize": 18})
            plt.ylabel(r"$f(x) \quad / \quad p(x)$", fontdict={"fontsize": 18})
            plt.legend(frameon=False, fontsize=18)
        elif x0 is not None and y0 is not None:
            xi = np.linspace(self.a, self.b, 200)
            yi = self.predict_x0(xi)  # 求傅里叶逼近的值
            plt.title("快速傅里叶变换$FFT$最小二乘三角逼近函数曲线 ", fontdict={"fontsize": 18})
            plt.ylabel(r"$p(x)$", fontdict={"fontsize": 18})
            plt.plot(xi, yi, "k-", lw=1.5, label="$p(x)$三角逼近")
            plt.plot(x0, y0, "ro", label="$(x_k, y_k)$")
            plt.legend(frameon=False, fontsize=17, loc="upper left")
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
