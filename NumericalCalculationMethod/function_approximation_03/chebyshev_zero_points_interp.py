# -*- coding: UTF-8 -*-
"""
@file_name:chebyshev_zero_points_interp.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from interpolation_02.lagrange_interpolation import LagrangeInterpolation  # 拉格朗日插值
from util_font import *


class ChebyshevZeroPointsInterpolation:
    """
    切比雪夫多项式零点插值算法
    """
    terms_zeros = None  # 切比雪夫多项式零点
    approximation_poly = None  # 逼近的多项式
    poly_coefficient, polynomial_orders = None, None  # 逼近多项式的系数，各项阶次
    max_abs_error, mae = None, None  # 逼近多项式的最大绝对值误差，绝对误差均值mae

    def __init__(self, approximate_fx, orders=6, x_span=np.array([-1, 1])):
        """
        必要参数的初始化
        """
        self.approximate_fx = approximate_fx
        self.orders = int(orders)  # 插值最高阶次
        self.a, self.b = x_span[0], x_span[1]

    def fit_approximation(self):
        """
        切比雪夫多项式零点插值核心算法：先求零点并变换空间，再进行拉格朗日插值，生成p(x)
        :return:
        """
        k = np.arange(0, self.orders + 1)  # 切比雪夫零点索引下标
        zero = np.cos((2 * k + 1) / 2 / (self.orders + 1) * np.pi)
        self.terms_zeros = (self.b - self.a) / 2 * zero + (self.b + self.a) / 2  # 存储零点，区间变换
        fun_values = self.approximate_fx(self.terms_zeros)  # 零点的函数值
        lag = LagrangeInterpolation(self.terms_zeros, fun_values)  # 拉格朗日插值
        lag.fit_interp()  # 生成拉格朗日插值多项式，符号多项式
        self.approximation_poly = lag.polynomial  # 插值后的逼近多项式
        self.poly_coefficient = lag.poly_coefficient  # 多项式系数
        self.polynomial_orders = lag.coefficient_order  # 多项式的阶次
        self.error_analysis()  # 误差分析

    def predict_x0(self, x0):
        """
        求解逼近多项式p(x)在给定点x0的值
        :return:
        """
        t = self.approximation_poly.free_symbols.pop()  # 提取自由变量
        appr_poly = sympy.lambdify(t, self.approximation_poly)  # 转换为lambda函数
        return np.array(appr_poly(x0))

    def error_analysis(self):
        """
        误差分析：10轮模拟，每轮100个服从U(a,b)的随机点，选取最大绝对误差和计算绝对误差均值
        :return:
        """
        mae = np.zeros(10)  # 存储10次随机值，真值与逼近多项式的均方根误差
        max_error = np.zeros(10)  # 存储10次随机模拟，每次最大的绝对值误差
        for i in range(10):
            xi = self.a + np.random.rand(1000) * (self.b - self.a)
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true = self.approximate_fx(xi)  # 真值
            yi_hat = self.predict_x0(xi)  # 预测值
            mae[i] = np.mean(np.abs(yi_hat - y_true))  # 100个随机点的绝对误差均值
            max_error[i] = max(np.abs(yi_hat - y_true))  # 100个随机点选最大的绝对值误差
            idx = np.argmax(np.abs(yi_hat - y_true))
            # print(i, ":", xi[idx], np.abs(yi_hat - y_true)[idx])
        self.max_abs_error = max(max_error)  # 10次模拟选最大的
        self.mae = np.mean(mae)  # 10次模拟均值

    def plt_approximation(self, is_show=True, is_fh_marker=False):
        """
        可视化逼近多项式。针对被逼近函数，如果is_fh_marker为True，则随机化50个点，并标记
        :param is_show: 用于绘制子图，如果绘制子图，则值为False
        :param is_fh_marker: 真实函数是曲线类型还是marker类型
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        xi = np.linspace(self.a, self.b, 200)
        yi_hat = self.predict_x0(xi)  # 求解模拟点
        plt.plot(xi, yi_hat, "r-", lw=2, label="$p(x) \ (k=%d)$" % self.orders)
        if is_fh_marker:
            xi = self.a + np.random.rand(50) * (self.b - self.a)
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true = self.approximate_fx(xi)
            plt.plot(xi, y_true, "k*", label="$f(x_k), \ x_k \sim U(a, b)$")
        else:
            y_true = self.approximate_fx(xi)
            plt.plot(xi, y_true, "k--", lw=2, label="$f(x)$")
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f(x) \ / \ p(x)$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        # mse_10的含义为10次随机绝对误差均值
        plt.title("切比雪夫零点插值逼近$(MAE_{10}=%.2e)$" % self.mae, fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=18, loc="best")  # loc="upper left"
        plt.grid(ls=":")
        if is_show:
            plt.show()
