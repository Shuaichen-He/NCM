# -*- coding: UTF-8 -*-
"""
@file_name: best_approximation_entity_utils.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from util_font import *


class BestApproximationUtils:
    """
    最佳多项式逼近工具类
    """
    approximation_poly = None  # 逼近的多项式
    poly_coefficient = None  # 逼近多项式的系数
    polynomial_orders = None  # 逼近多项式各项阶次
    max_abs_error = np.infty  # 10次模拟选最大的
    mae = np.infty  # 10次模拟均值

    def __init__(self, fun, k, interval=np.array([-1, 1])):
        self.fun, self.approximate_fx = self.lambda_function(fun)  # 所逼近的函数
        self.k = k  # 逼近已知函数所需项数
        self.a, self.b = interval[0], interval[1]  # 区间左右端点

    @staticmethod
    def lambda_function(fun):
        """
        转换为lambda函数
        :param fun: 符号函数
        :return:
        """
        t = fun.free_symbols.pop()
        return fun, sympy.lambdify(t, fun)

    def error_analysis(self):  # 参考类OrthogonalPolynomialUtils设计
        """
        误差分析：10次模拟，每次100个随机点，分析绝对误差均值和最大绝对误差
        :return:
        """
        mae = np.zeros(10)  # 存储10次随机值，真值与逼近多项式的均方根误差
        max_error = np.zeros(10)  # 存储10次随机模拟，每次最大的绝对值误差
        for i in range(10):
            xi = self.a + np.random.rand(100) * (self.b - self.a)
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true = self.approximate_fx(xi)  # 原函数精确值
            yi = self.predict_x0(xi)  # 逼近多项式值
            mae[i] = np.mean(np.abs(yi - y_true))  # 100个随机点的绝对误差均值
            max_error[i] = max(np.abs(yi - y_true))  # 100个随机点选最大的绝对值误差
        self.max_abs_error = max(max_error)  # 10次模拟选最大的
        self.mae = float(np.mean(mae))  # 10次模拟均值

    def predict_x0(self, x0):
        """
        求解逼近多项式给定点的值
        :return:
        """
        t = self.approximation_poly.free_symbols.pop()
        appr_poly = sympy.lambdify(t, self.approximation_poly)
        return appr_poly(x0)

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
            xi = self.a + np.random.rand(100) * (self.b - self.a)
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true = self.approximate_fx(xi)
            plt.plot(xi, y_true, "k*", label="$f(x_k), \ x_k \sim U(a, b)$")
        else:
            y_true = self.approximate_fx(xi)
            plt.plot(xi, y_true, "k--", lw=2, label="$f(x)$")
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f(x) \quad / \quad p(x)$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        # mse_10的含义为10次随机绝对误差均值
        plt.title("%s逼近$(MAE_{10}=%.2e)$" % (sub_title, self.mae), fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=18)
        plt.grid(ls=":")
        if is_show:
            plt.show()

