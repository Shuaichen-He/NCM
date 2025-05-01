# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
from numerical_optimization_13.dfp_quasi_newton import DFPQuasiNewtonOptimization
from numerical_optimization_13.bfgs_quasi_newton import BFGSQuasiNewtonOptimization

# DFP算法
x = sympy.symbols("x_1:3")
fun = x[0] ** 2 * sympy.sin(x[0] + x[1] ** 2) + x[1] ** 2 * sympy.exp(x[0]) + 6 * sympy.cos(x[0] ** 2 + x[1])
x0, eps = [-2.5, -2.5], 1e-16  # 极小值
dfp = DFPQuasiNewtonOptimization(fun, x0, eps=eps, is_minimum=True)  # 极小值
sol = dfp.fit_optimize()
print(dfp.local_extremum)
print("最优解：%.15f, %.15f， 最优值：%.15f" % (sol[0], sol[1], sol[2]))
dfp.plt_optimization([-3, -1], [-3, -1])
x = sympy.symbols("x_1:3")
fun = -1 * (x[0] ** 2 * sympy.sin(x[0] + x[1] ** 2) + x[1] ** 2 * sympy.exp(x[0]) + 6 * sympy.cos(x[0] ** 2 + x[1]))
x0, eps = [-1.5, -2], 1e-16  # 极大值
dfp = DFPQuasiNewtonOptimization(fun, x0, eps=eps, is_minimum=False)  # 极大值
sol = dfp.fit_optimize()
print(dfp.local_extremum)
print("最优解：%.15f, %.15f， 最优值：%.15f" % (sol[0], sol[1], sol[2]))
dfp.plt_optimization([-3, -1], [-3, -1])
print("=" * 80)

# BFGS
x = sympy.symbols("x_1:3")
fun = x[0] ** 2 * sympy.sin(x[0] + x[1] ** 2) + x[1] ** 2 * sympy.exp(x[0]) + 6 * sympy.cos(x[0] ** 2 + x[1])
x0, eps = [-2.5, -2.5], 1e-16  # 极小值
bfgs = BFGSQuasiNewtonOptimization(fun, x0, eps=eps, is_minimum=True)  # 极小值
sol = bfgs.fit_optimize()
print(bfgs.local_extremum)
print("最优解：%.15f, %.15f， 最优值：%.15f" % (sol[0], sol[1], sol[2]))
bfgs.plt_optimization([-3, -1], [-3, -1])
x = sympy.symbols("x_1:3")
fun = -1 * (x[0] ** 2 * sympy.sin(x[0] + x[1] ** 2) + x[1] ** 2 * sympy.exp(x[0]) + 6 * sympy.cos(x[0] ** 2 + x[1]))
x0, eps = [-1.5, -2], 1e-16  # 极大值
bfgs = BFGSQuasiNewtonOptimization(fun, x0, eps=eps, is_minimum=False)  # 极大值
sol = bfgs.fit_optimize()
print(bfgs.local_extremum)
print("最优解：%.15f, %.15f， 最优值：%.15f" % (sol[0], sol[1], sol[2]))
bfgs.plt_optimization([-3, -1], [-3, -1])