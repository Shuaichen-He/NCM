# -*- coding: UTF-8 -*-
"""
@file_name: test_cubic_appr.py
@time: 2022-09-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_optimization_13.cubic_approximation import CubicApproximationOptimization

x = sympy.symbols("x")
# fun = lambda x: x ** 2 * np.exp(-x) - np.sin(x)  # 例1
fun = -11 * sympy.sin(x) - 7 * sympy.cos(5 * x)  # 例2，极大值
# fun = 11 * sympy.sin(x) + 7 * sympy.cos(5 * x)  # 例2，极小值
# x_span, eps = [0, 2], 1e-15  # 极小值
# x_span, eps = [2, 6], 1e-15  # 极大值
# 例2参数设置
# x_span, eps = [-1, 0], 1e-15  # 极小值
x_span, eps = [1, 1.5], 1e-15 # 极大值
cao = CubicApproximationOptimization(fun, x_span, eps, is_minimum=False)  # 注意is_minimum的设置
e_x = cao.fit_optimize()
print("(%.15f, %.15f)" % (e_x[0], e_x[1]))
# for i in range(len(fso.reduce_zone)):
#     print("%d: %.15f, %.15f" % (i + 1, fso.reduce_zone[i][0], fso.reduce_zone[i][1]))
# fso.plt_optimization(plt_zone=[-1, 10])
cao.plt_optimization(plt_zone=[-3, 3])