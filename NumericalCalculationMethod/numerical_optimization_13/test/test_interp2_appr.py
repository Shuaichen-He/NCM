# -*- coding: UTF-8 -*-
"""
@file_name: test_interp2_appr.py
@time: 2022-09-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_optimization_13.interp2_approximation import Interp2ApproximationOptimization


# fun = lambda x: x ** 2 * np.exp(-x) - np.sin(x)  # 例1
# fun = lambda x: -1 * (11 * np.sin(x) + 7 * np.cos(5 * x))  # 例2，极大值
# fun = lambda x: -1 * x ** 2
fun = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 例2，极小值
# x_span, eps = [0, 2], 1e-15  # 极小值
# x_span, eps = [2, 6], 1e-15  # 极大值
# 例2参数设置
x_span, eps = [-1, 0], 1e-15  # 极小值
# x_span, eps = [1, 1.5], 1e-15 # 极大值
iao = Interp2ApproximationOptimization(fun, x_span, eps, is_minimum=True)  # 注意is_minimum的设置
e_x = iao.fit_optimize()
print("(%.15f, %.15f)" % (e_x[0], e_x[1]))
# for i in range(len(iao.reduce_zone)):
#     print("%d: %.15f, %.15f" % (i + 1, iao.reduce_zone[i][0], iao.reduce_zone[i][1]))
# iao.plt_optimization(plt_zone=[-1, 10])
iao.plt_optimization(plt_zone=[-3, 3])