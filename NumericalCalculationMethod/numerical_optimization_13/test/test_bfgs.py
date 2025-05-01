# -*- coding: UTF-8 -*-
"""
@file_name: test_bfgs.py
@time: 2022-09-14
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_optimization_13.bfgs_quasi_newton import BFGSQuasiNewtonOptimization

# fun = lambda x: (x[0] ** 2 - 2 * x[0]) * sympy.exp(-x[0] ** 2 - x[1] ** 2 - x[0] * x[1])  # 极大值在函数前整体添加负号
#
#
# def grad_g(X):  # 梯度向量
#     x, y = X[0], X[1]
#     grad_val = [(-2 * x - y) * (x ** 2 - 2 * x) * np.exp(-x ** 2 - x * y - y ** 2) +
#                 (2 * x - 2) * np.exp(-x ** 2 - x * y - y ** 2),
#                 (-x - 2 * y) * (x ** 2 - 2 * x) * np.exp(-x ** 2 - x * y - y ** 2)]
#     return np.asarray(grad_val)

x = sympy.symbols("x_1:3")
fun = -1 * (x[0] ** 2 - 2 * x[0]) * sympy.exp(-x[0] ** 2 - x[1] ** 2 - x[0] * x[1])  # 目标函数
x0, eps = [-1, 0], 1e-16  # 极大值
# x0, eps = [0, 0], 1e-16  # 极小值
# fun = -1 * (x[0] - x[1]) / (x[0] ** 2 + x[1] ** 2 + 2)
# x0, eps = [0.3, 0.2], 1e-16


bfgs = BFGSQuasiNewtonOptimization(fun, x0, eps=eps, is_minimum=False)  # 修改is_minimum变换极小极大值
sol = bfgs.fit_optimize()
print(bfgs.local_extremum)
print("最优解：%.15f, %.15f， 最优值：%.15f" % (sol[0], sol[1], sol[2]))
bfgs.plt_optimization([-3, 3], [-3, 3])