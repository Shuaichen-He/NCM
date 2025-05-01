# -*- coding: UTF-8 -*-
"""
@file_name: orthogonal_poly_utils.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from util_font import *


class OrthogonalPolynomialUtils:
    """
    正交多项式函数逼近实体工具类
    """
    T_coefficient = None  # 多项式各项和对应系数
    approximation_poly = None  # 逼近的多项式
    poly_coefficient = None  # 逼近多项式的系数
    polynomial_orders = None  # 逼近多项式各项阶次
    max_abs_error = np.infty  # 10次模拟选最大的
    mae = np.infty  # 10次模拟均值

    def __init__(self, fun, k=6, x_span=np.array([-1, 1])):
        self.a, self.b = x_span[0], x_span[1]  # 区间左右端点
        self.fun_transform, self.approximate_fx = self.interval_transform(fun)  # 区间转化函数
        self.k = k  # 逼近已知函数所需项数

    def interval_transform(self, fun):
        """
        函数的区间转化
        :return: 转化区间后的函数
        """
        t = fun.free_symbols.pop()  # 获取自由符号变量
        fun_transform = fun.subs(t, (self.b + self.a) / 2 + (self.b - self.a) / 2 * t)
        fun_expr = sympy.lambdify(t, fun)  # 转换为lambda函数
        return fun_transform, fun_expr

    def error_analysis(self):
        """
        误差分析：10轮模拟，每轮100个随机点，分析绝对误差均值和最大绝对误差
        :return:
        """
        mae_10 = np.zeros(10)  # 存储10次随机值，真值与逼近多项式的均方根误差
        max_error = np.zeros(10)  # 存储10次随机模拟，每次最大的绝对值误差
        for i in range(10):
            xi = self.a + np.random.rand(1000) * (self.b - self.a)
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true = self.approximate_fx(xi)
            yi_hat = self.predict_x0(xi)
            mae_10[i] = np.mean(np.abs(yi_hat - y_true))  # 100个随机点的绝对误差均值
            max_error[i] = max(np.abs(yi_hat - y_true))  # 100个随机点选最大的绝对值误差
            idx = np.argmax(np.abs(yi_hat - y_true))
            # print(i, ":", xi[idx], np.abs(yi_hat - y_true)[idx])
        self.max_abs_error = max(max_error)  # 10次模拟选最大的
        self.mae = float(np.mean(mae_10))  # 10次模拟均值

    def predict_x0(self, x0):
        """
        求解逼近多项式p(x)在给定点x0的值
        :return:
        """
        t = self.approximation_poly.free_symbols.pop()
        y0 = np.zeros(len(x0))
        for i in range(len(x0)):
            xi = (x0[i] - (self.a + self.b) / 2) * 2 / (self.b - self.a)  # 区间转换
            y0[i] = self.approximation_poly.evalf(subs={t: xi})  # 求解逼近值
        return y0

    def plt_approximation(self, sub_title, is_show=True, is_fh_marker=False):
        """
        可视化逼近多项式。针对被逼近函数，如果is_fh_marker为True，则随机化50个点，并标记
        :param sub_title: 子标题
        :param is_show: 用于绘制子图，如果绘制子图，则值为False
        :param is_fh_marker: 真实函数是曲线类型还是marker类型
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        xi = np.linspace(self.a, self.b, 200)
        yi_hat = self.predict_x0(xi)  # 求解模拟点
        plt.plot(xi, yi_hat, "r-", lw=2, label="$p(x) \ (k=%d)$" % self.k)
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
        plt.title("%s逼近$(MAE_{10}=%.2e)$" % (sub_title, self.mae), fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=18, loc="best")
        plt.grid(ls=":")
        if is_show:
            plt.show()

