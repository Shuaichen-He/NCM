# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
from Experiment.util_font import *
import numpy as np
from nonlinear_equations_09.nlinequs_fixed_point import NLinearFxFixedPoint

# 可视化方法
plt.figure(figsize=(7, 5))
x, y = sympy.symbols("x, y")
nlin_equs = [x ** 3 + y ** 3 - 6 * x + 3, x ** 3 - y ** 3 - 6 * y + 2]  # 定义的非线性方程组
p0 = sympy.plot_implicit(nlin_equs[0], show=False, line_color="r", points=500)  # 绘制第一个方程
p1 = sympy.plot_implicit(nlin_equs[1], show=False, line_color="b", points=500)  # 绘制第二个方程
p0.extend(p1)  # 在方程1的基础上添加方程2的图像
p0.show()


def nlin_equs(x):
    """
    非线性方程组的定义
    :param x: 向量, x[0]表示x, x[1]表示y
    :return:
    """
    y = [x[0] ** 3 / 6 + x[1] ** 3 / 6 + 0.5, x[0] ** 3 / 6 - x[1] ** 3 / 6 + 1 / 3]
    return np.asarray(y, dtype=np.float)


x_init = np.array([[-2.4238, -1.4893], [0.5, 0.5], [1.8, 1.2]])
nlfp = NLinearFxFixedPoint(nlin_equs, x_init[1], max_iter=1000, eps=1e-16, is_plt=True)
nlfp.fit_roots()
x = nlfp.iter_roots_precision[-1]
print("x1: %.10f, x2: %.10f, 精度: %.10e" % (x[1][0], x[1][1], x[2]))

