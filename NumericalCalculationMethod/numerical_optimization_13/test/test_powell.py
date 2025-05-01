# -*- coding: UTF-8 -*-
"""
@file_name: test_powell.py
@time: 2022-09-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_optimization_13.powell_method import PowellOptimization

x = sympy.symbols("x_1:3")
fun = sympy.cos(x[0]) - sympy.sin(x[1])
x0, eps = [0, 0], 1e-16
po = PowellOptimization(fun, x0, eps, is_minimum=True)  # 此处修改is_minimum为False求极大值
e_x = po.fit_optimize()
po.plt_optimization([5, 20], [5, 20])
print("%.15f, %.15f" % (e_x[0], e_x[1]))
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))

fun = 60 - 10 * x[0] - 4 * x[1] + x[0] ** 2 + x[1] ** 2 - x[0] * x[1]
x0, eps = [1, 1], 1e-16
po = PowellOptimization(fun, x0, eps, is_minimum=True)  # 此处修改is_minimum为False求极大值
e_x = po.fit_optimize()
po.plt_optimization([1, 15], [1, 15])
print("%.15f, %.15f" % (e_x[0], e_x[1]))
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))

# fun = -1 * (x[0] ** 2 - 2 * x[0]) * sympy.exp(-x[0] ** 2 - x[1] ** 2 - x[0] * x[1])  # 极小值
# x0, eps = [0, 1], 1e-15
# po = PowellOptimization(fun, x0, eps, is_minimum=False)  # 此处修改is_minimum为False求极大值
# e_x = po.fit_optimize()
# po.plt_optimization([-3, 3], [-3, 3])
# print(e_x)
# print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))

x = sympy.symbols("x_1:4")
fun = 10 * (x[0] + x[1] - 5) ** 4 + (x[0] - x[1] + x[2]) ** 2 + (x[1] + x[2]) ** 6
x0, eps = [0, 0, 0], 1e-16
po = PowellOptimization(fun, x0, eps, is_minimum=True)  # 此处修改is_minimum为False求极大值
e_x = po.fit_optimize()
print("%.15f, %.15f, %.15f" % (e_x[0], e_x[1], e_x[2]))
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1], x[2]: e_x[2]}))