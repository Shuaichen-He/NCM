# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from Experiment.util_font import *
from nonlinear_equations_09.gauss_newton import GaussNewtonIteration
from nonlinear_equations_09.damped_least_square_method import DampedLeastSquare_LM
from nonlinear_equations_09.global_LM_method import GlobalLevenbergMarquardt


plt.figure(figsize=(7, 5))
x, y = sympy.symbols("x, y")
nlin_equs = [4 * x - y + 0.1 * sympy.exp(x) - 1, -x + 4 * y + 0.125 * x ** 2]
p0 = sympy.plot_implicit(nlin_equs[0], show=False, line_color="r")
p1 = sympy.plot_implicit(nlin_equs[1], show=False, line_color="c")
p0.extend(p1)
p0.show()


def nlin_funs(x):
    nlinequs = [4 * x[0] - x[1] + 0.1 * np.exp(x[0]) - 1,
                -x[0] + 4 * x[1] + 0.125 * x[0] ** 2]
    return np.asarray(nlinequs, dtype=np.float)

x0 = np.array([0, 0])  # 初值
h = [0.01, 0.01]
# 高斯—牛顿法
gni = GaussNewtonIteration(nlin_funs, x0, h, max_iter=1000, eps=1e-16, is_plt=True)
gni.fit_roots()
print(gni.iter_roots_precision[-1])

# 阻尼最小二乘
dls = DampedLeastSquare_LM(nlin_funs, x0, h, u=0.01, v=5, max_iter=1000, eps=1e-16, is_plt=True)
dls.fit_nlinequs_roots()
print(dls.iter_roots_precision[-1])

# 全局化LM法
glm = GlobalLevenbergMarquardt(nlin_funs, x0, h, delta=1, max_iter=1000, eps=1e-16, is_plt=True)
glm.fit_nlinequs_roots()
print(glm.iter_roots_precision[-1])