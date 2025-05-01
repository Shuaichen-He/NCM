# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_2order_cubic_spline.py
@time:2021/08/26
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from numerical_differentiation_05.cubic_bspline_2_order_differentiation import CubicBSpline2OrderDifferentiation

# def fun(x):
#     return np.sin(x) * np.exp(-x)


# def fun(x):  # 微分函数
#     return np.log(x) * np.sin(x)

fun = lambda x: np.sin(x) / np.sqrt(x)  # 微分函数


# def derivative_fun(x):  # 一阶导函数
#     return np.sin(x) / x + np.log(x) * np.cos(x)

def second_derivative_fun():
    x = sympy.Symbol("x")
    fun = sympy.sin(x) / sympy.sqrt(x)  # 符号定义
    diff_fun = fun.diff(x, 2)  # 2阶导数
    fun_expr = sympy.lambdify(x, diff_fun, "numpy")  # lambda函数，进行数值运算
    return fun_expr


# x0 = np.array([1.23, 1.75, 1.89, 2.14, 2.56])
derivative_fun = second_derivative_fun()
x0 = np.array([2, 2.3, 3.6, 4.9, 5.2, 5.5, 6.8, 7.1, 8.4, 9.7, 10.5])  # 待求解微分点
h_num = [0.01, 0.05, 0.1, 0.15]  # 用于测试不同的微分步长
# print(derivative_fun(x0))

plt.figure(figsize=(14, 5))
plt.subplot(121)
cbsod = CubicBSpline2OrderDifferentiation(fun, n=11, h=0.1)
y0 = cbsod.predict_diff_x0(x0)
print("微分值：", y0, "\n误差：", y0 - derivative_fun(x0))
print("-" * 80)
cbsod.plt_2_order_different([2, 11], second_derivative_fun(), x0, y0, is_show=False, is_fh_marker=True)
plt.subplot(122)
xi = np.linspace(2, 11, 200)
n_num = [8, 9, 10, 11]  # 用于测试不同扩展节点数
print(n_num)
ls_ = ["--", "-.", ":", "-"]
for i, n in enumerate(n_num):  # 用于测试不同扩展节点数
    cbsod = CubicBSpline2OrderDifferentiation(fun, n=n, h=0.1)
    y0 = cbsod.predict_diff_x0(x0)
    yi = cbsod.predict_diff_x0(xi)
    error = np.abs(yi - derivative_fun(xi))
    mae = np.mean(error)
    print("最大绝对值误差：%.10e" % np.max(error))
    print("平均绝对值误差：%.10e" % mae)
    plt.semilogy(xi, error, ls_[i], lw=2, label="$n=%d$" % n)
    print("微分值：", y0, "\n误差：", derivative_fun(x0) - y0)
    print("=" * 80)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert f^{\prime\prime}(x)- \hat f^{\prime\prime}(x) \vert$", fontdict={"fontsize": 18})
plt.title("三次均匀B样条二阶数值微分误差曲线$(h=0.1)$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()


# 不同微分步长
plt.figure(figsize=(14, 5))
plt.subplot(121)
cbsod = CubicBSpline2OrderDifferentiation(fun, n=9, h=0.1)
y0 = cbsod.predict_diff_x0(x0)
cbsod.plt_2_order_different([2, 11], second_derivative_fun(), x0, y0, is_show=False, is_fh_marker=True)
plt.subplot(122)
xi = np.linspace(2, 11, 200)
h_ = [0.01, 0.05, 0.10, 0.15]  # 用于测试不同微分步长
ls_ = ["--", "-.", ":", "-"]
for i, h in enumerate(h_):  # 用于测试不同扩展节点数
    cbsod = CubicBSpline2OrderDifferentiation(fun, n=9, h=h)
    y0 = cbsod.predict_diff_x0(x0)
    yi = cbsod.predict_diff_x0(xi)
    error = np.abs(yi - derivative_fun(xi))
    mae = np.mean(error)
    print("最大绝对值误差：%.10e" % np.max(error))
    print("平均绝对值误差：%.10e" % mae)
    plt.semilogy(xi, error, ls_[i], lw=2, label="$h=%.2f$" % h)
    print("微分值：", y0, "\n误差：", derivative_fun(x0) - y0)
    print("=" * 80)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert f^{\prime\prime}(x)- \hat f^{\prime\prime}(x) \vert$", fontdict={"fontsize": 18})
plt.title("三次均匀B样条二阶数值微分误差曲线($n=9$)", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()

