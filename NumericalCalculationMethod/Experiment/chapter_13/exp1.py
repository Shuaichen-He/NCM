# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_optimization_13.golden_section_search import GoldenSectionSearchOptimization
from numerical_optimization_13.fibonacci_search import FibonacciSearchOptimization
from numerical_optimization_13.interp2_approximation import Interp2ApproximationOptimization
from numerical_optimization_13.cubic_approximation import CubicApproximationOptimization
from Experiment.util_font import *

fun = lambda x: np.exp(-x ** 2 + 2 * x) * np.sin(x ** 2)

# 函数可视化
plt.figure(figsize=(7, 5))
xi = np.linspace(-1, 3, 200)
plt.plot(xi, fun(xi), "-")
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("单变量函数在区间的$[-1, 3]$图像", fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=18)
plt.show()

# 黄金分割法
x_span, eps = [1.5, 2.5], 1e-16  # 极小值
gsso = GoldenSectionSearchOptimization(fun, x_span, eps, is_minimum=True)  # 注意is_minimum的设置
e_x = gsso.fit_optimize()
print("黄金分割法极小值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
gsso.plt_optimization([-1, 3])
x_span, eps = [0.5, 1.5], 1e-16  # 极大值
gsso = GoldenSectionSearchOptimization(fun, x_span, eps, is_minimum=False)  # 注意is_minimum的设置
e_x = gsso.fit_optimize()
gsso.plt_optimization([-1, 3])
print("黄金分割法极大值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
print("=" * 60)

# 斐波那契搜索
x_span, eps = [1.5, 2.5], 1e-16  # 极小值
fso = FibonacciSearchOptimization(fun, x_span, eps, is_minimum=True)  # 注意is_minimum的设置
e_x = fso.fit_optimize()
print("斐波那契搜索极小值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
fso.plt_optimization([-1, 3])
x_span, eps = [0.5, 1.5], 1e-16  # 极大值
fso = FibonacciSearchOptimization(fun, x_span, eps, is_minimum=False)  # 注意is_minimum的设置
e_x = fso.fit_optimize()
print("斐波那契搜索极大值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
fso.plt_optimization([-1, 3])
print("=" * 60)

# 二次插值逼近
x_span, eps = [1.8, 2.2], 1e-16  # 极小值
iao = Interp2ApproximationOptimization(fun, x_span, eps, is_minimum=True)  # 注意is_minimum的设置
e_x = iao.fit_optimize()
print("二次插值逼近极小值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
iao.plt_optimization([-1, 3])
f_max = lambda x: -1 * np.exp(-x ** 2 + 2 * x) * np.sin(x ** 2)
x_span, eps = [1.0, 1.2], 1e-16  # 极大值
iao = Interp2ApproximationOptimization(f_max, x_span, eps, is_minimum=False)  # 注意is_minimum的设置
e_x = iao.fit_optimize()
print("二次插值逼近极大值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
iao.plt_optimization([-1, 3])
print("=" * 60)

# 三次逼近方法
x = sympy.symbols("x")
fun = sympy.exp(-x ** 2 + 2 * x) * sympy.sin(x ** 2)
x_span, eps = [1.8, 2.2], 1e-16  # 极小值
cao = CubicApproximationOptimization(fun, x_span, eps, is_minimum=True)  # 注意is_minimum的设置
e_x = cao.fit_optimize()
print("三次逼近极小值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
cao.plt_optimization([-1, 3])
f_max = -1 * sympy.exp(-x ** 2 + 2 * x) * sympy.sin(x ** 2)
x_span, eps = [1.0, 1.2], 1e-16  # 极大值
cao = CubicApproximationOptimization(f_max, x_span, eps, is_minimum=False)  # 注意is_minimum的设置
e_x = cao.fit_optimize()
print("三次逼近极大值：(%.15f, %.15f)" % (e_x[0], e_x[1]))
cao.plt_optimization([-1, 3])