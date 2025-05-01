# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_2d_cubic_spline.py
@time:2021/08/26
"""
import numpy as np
import sympy
from numerical_differentiation_05.cubic_bspline_2_order_differentiation import CubicBSpline2OrderDifferentiation

fun = lambda x: x ** 2 + x ** (1 / 3) + np.sin(x * np.cos(x) ** 2)


def second_derivative_fun():
    x = sympy.Symbol("x")
    fun = x ** 2 + x ** (1 / 3) + sympy.sin(x * sympy.cos(x) ** 2)
    diff_fun = fun.diff(x, 2)  # 2阶导数
    fun_expr = sympy.lambdify(x, diff_fun, "numpy")  # lambda函数，进行数值运算
    return fun_expr


x0 = np.array([4, 5, 6, 7])
derivative_fun = second_derivative_fun()
csd = CubicBSpline2OrderDifferentiation(fun, n=20, h=0.01)
csd.predict_diff_x0(x0)
csd.plt_differentiation([1, 5], second_derivative_fun())
print(csd.diff_value)
# print(-np.sin(x0))
print("准确值：", derivative_fun(x0))
print("误差：", derivative_fun(x0) - csd.diff_value)
