# -*- coding: UTF-8 -*-
"""
@file_name: test_fibonacci_search.py
@time: 2022-09-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_optimization_13.fibonacci_search import FibonacciSearchOptimization


# fun = lambda x: x ** 2 * np.exp(-x) - np.sin(x)  # 例1
fun = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 例2
# x_span, eps = [0, 2], 1e-15  # 极小值
# x_span, eps = [2, 6], 1e-15  # 极大值
# 例2参数设置
# x_span, eps = [-1, 0], 1e-15  # 极小值
x_span, eps = [1, 1.5], 1e-15  # 极大值
fso = FibonacciSearchOptimization(fun, x_span, eps, is_minimum=False)  # 注意is_minimum的设置
e_x = fso.fit_optimize()
print("(%.15f, %.15f)" % (e_x[0], e_x[1]))
for i in range(len(fso.reduce_zone)):
    print("%d: %.15f, %.15f" % (i + 1, fso.reduce_zone[i][0], fso.reduce_zone[i][1]))
# fso.plt_optimization(plt_zone=[-1, 10])
fso.plt_optimization(plt_zone=[-3, 3])