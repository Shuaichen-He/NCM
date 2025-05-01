# -*- coding: UTF-8 -*-
"""
@file:test_mulstablepoints.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from nonlinear_equations_09.nlinequs_fixed_point import NLinearFxFixedPoint
import sympy
from util_font import *

def fx1(x):
    """
    非线性方程组（1）的定义
    :param x:
    :return:
    """
    y = [(x[0] ** 2 + x[1] ** 2 + 8) / 10, (x[0] * x[1] ** 2 + x[0] + 8) / 10]
    return np.asarray(y, dtype=np.float64)


def fx2(x):
    """
    非线性方程组（2）的定义
    :param x:
    :return:
    """
    y = [0.5 * np.sin(x[0]) + 0.1 * np.cos(x[0] * x[1]), 0.5 * np.cos(x[0]) - 0.1 * np.cos(x[1])]
    return np.asarray(y, dtype=np.float64)


# 可视化方法
plt.figure(figsize=(7, 5))
x, y = sympy.symbols("x, y")
nlin_equs = [x ** 2 - 10 * x + y ** 2 + 8, x * y ** 2 + x - 10 * y + 8]  # 定义的非线性方程组
p0 = sympy.plot_implicit(nlin_equs[0], show=False, line_color="r", points=500)  # 绘制第一个方程
p1 = sympy.plot_implicit(nlin_equs[1], show=False, line_color="b", points=500)  # 绘制第二个方程
p0.extend(p1)  # 在方程1的基础上添加方程2的图像
p0.show()
plt.figure(figsize=(7, 5))
x, y = sympy.symbols("x, y")
nlin_equs = [0.5 * sympy.sin(x) + 0.1 * sympy.cos(x * y) - x, 0.5 * sympy.cos(x) - 0.1 * sympy.cos(y) - y]  # 定义的非线性方程组
p0 = sympy.plot_implicit(nlin_equs[0], show=False, line_color="r", points=500)  # 绘制第一个方程
p1 = sympy.plot_implicit(nlin_equs[1], show=False, line_color="b", points=500)  # 绘制第二个方程
p0.extend(p1)  # 在方程1的基础上添加方程2的图像
p0.show()

x0 = np.array([0, 0])
nlfp = NLinearFxFixedPoint(fx1, x0, max_iter=1000, eps=1e-16, is_plt=True)
nlfp.fit_roots()
print(nlfp.iter_roots_precision)
