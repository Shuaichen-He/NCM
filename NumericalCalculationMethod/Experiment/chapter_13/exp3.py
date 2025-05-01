# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from Experiment.util_font import *
from numerical_optimization_13.gradient_descent import GradientDescentOptimization
from numerical_optimization_13.newton_method import NewtonOptimization

# 梯度方法
x = sympy.symbols("x_1:3")
fun = x[0] ** 3 + x[1] ** 3 - 3 * x[0] - 3 * x[1] + 5
x0, eps = [1.5, 1.5], 1e-15
gdo = GradientDescentOptimization(fun, x0, eps, is_minimum=True)  # 极小值
e_x = gdo.fit_optimize()
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))
print(gdo.local_extremum)
gdo.plt_optimization([-2, 2], [-2, 2])
x = sympy.symbols("x_1:3")
fun = -1 * (x[0] ** 3 + x[1] ** 3 - 3 * x[0] - 3 * x[1] + 5)
x0, eps = [-0.5, -1.5], 1e-15
gdo = GradientDescentOptimization(fun, x0, eps, is_minimum=False)  # 极大值
e_x = gdo.fit_optimize()
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))
print(gdo.local_extremum)
gdo.plt_optimization([-2, 2], [-2, 2])
print("=" * 80)

# 牛顿法
x = sympy.symbols("x_1:3")
fun = x[0] ** 3 + x[1] ** 3 - 3 * x[0] - 3 * x[1] + 5
x0, eps, opt_type = [1.5, 1.5], 1e-16, "improve"
no= NewtonOptimization(fun, x0, eps, opt_type, is_minimum=True)  # 极小值
e_x = no.fit_optimize()
no.plt_optimization([-2, 2], [-2, 2])
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))
print(no.local_extremum)
x = sympy.symbols("x_1:3")
fun = -1 * (x[0] ** 3 + x[1] ** 3 - 3 * x[0] - 3 * x[1] + 5)
x0, eps, opt_type = [-0.5, -1.5], 1e-16, "improve"
no= NewtonOptimization(fun, x0, eps, opt_type, is_minimum=False)  # 极大值
e_x = no.fit_optimize()
no.plt_optimization([-2, 2], [-2, 2])
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))
print(no.local_extremum)