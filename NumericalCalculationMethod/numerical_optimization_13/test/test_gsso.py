# -*- coding: UTF-8 -*-
"""
@file_name: test_gsso.py
@time: 2022-09-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_optimization_13.golden_section_search import GoldenSectionSearchOptimization


fun = lambda x: x ** 2 * np.exp(-x) - np.sin(x)
# fun = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 例2
print(fun(4.591466387693221))
x_span, eps = [0, 2], 1e-15  # 极小值
# x_span, eps = [2, 6], 1e-15  # 极大值
# x_span, eps = [1, 1.5], 1e-15  # 极大值
# x_span, eps = [-1, 0], 1e-15  # 极小值
gsso = GoldenSectionSearchOptimization(fun, x_span, eps, is_minimum=True)  # 注意is_minimum的设置
e_x = gsso.fit_optimize()
print("(%.15f, %.15f)" % (e_x[0], e_x[1]))
for i in range(len(gsso.reduce_zone)):
    print("%d: %.15f, %.15f" % (i + 1, gsso.reduce_zone[i][0], gsso.reduce_zone[i][1]))
gsso.plt_optimization(plt_zone=[-1, 10])