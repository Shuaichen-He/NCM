# -*- coding: UTF-8 -*-
"""
@file:test_gauss_newton.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from nonlinear_equations_09.gauss_newton import GaussNewtonIteration


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


x0 = np.array([1, 1, 1])
h = [0.1, 0.1, 0.1]
gni = GaussNewtonIteration(nlin_funs1, x0, h, max_iter=1000, eps=1e-15, is_plt=True)
gni.fit_roots()

x0 = np.array([0, 0, 0])
h = [0.1, 0.01, 0.01]
gni = GaussNewtonIteration(nlin_funs2, x0, h, max_iter=1000, eps=1e-15, is_plt=True)
gni.fit_roots()