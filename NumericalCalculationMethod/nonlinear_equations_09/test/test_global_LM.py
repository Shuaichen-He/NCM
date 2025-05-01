# -*- coding: UTF-8 -*-
"""
@file:test_gloabl_LM.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from nonlinear_equations_09.global_LM_method import GlobalLevenbergMarquardt


def nlin_funs1(x):
    nlinequs = [np.sin(x[0]) + x[1] ** 2 + np.log(x[2]) - 7,
                3 * x[0] + 2 ** x[1] - x[2] ** 3 + 1,
                x[0] + x[1] + x[2] - 5]
    return np.asarray(nlinequs, dtype=np.float64)


def nlin_funs2(x):
    y = [3 * x[0] - np.cos(x[1] * x[2]) - 0.5,
         x[0] ** 2 - 81 * (x[1] + 0.1) ** 2 + np.sin(x[2]) + 1.06,
         np.exp(-x[0] * x[1]) + 20 * x[2] + 10 / 3 * np.pi - 1]
    return np.asarray(y, dtype=np.float64)


def nlin_funs3(x):
    # 非线性方程组的定义
    n = 100
    y = np.zeros((n, 1))
    y[:-1] = x[:-1] + np.sum(x) - (n + 1)
    y[-1] = np.prod(x) - 1
    return np.asarray(y, dtype=np.float64)


# x0 = np.array([1, 1, 1])
# h = [0.01, 0.01, 0.01]
# dls = GlobalLevenbergMarquardt(nlin_funs1, x0, h, delta=1, max_iter=1000, eps=1e-16, is_plt=True)
# dls.fit_nlinequs_roots()
x0 = np.array([0, 0, 0])
h = [0.1, 0.01, 0.01]
glm = GlobalLevenbergMarquardt(nlin_funs2, x0, h, delta=1, max_iter=1000, eps=1e-15, is_plt=True)
glm.fit_nlinequs_roots()

# x0 = 0.5 * np.ones(100)  # 改变初值
# x0[-1] = 0.01
# h = 0.1 * np.ones(100)
# glm = GlobalLevenbergMarquardt(nlin_funs3, x0, h, delta=1, max_iter=1000, eps=1e-16, is_plt=True)
# glm.fit_nlinequs_roots()
