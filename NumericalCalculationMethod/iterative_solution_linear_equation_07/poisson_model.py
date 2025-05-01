# -*- coding: UTF-8 -*-
"""
@file_name: poisson_model.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class PoissonModel:
    """
    泊松方程模型，按照二维坐标系命名，u(0, y)为左边界，u(x, 0)为下边界，依次类推
    """
    # 泊松方程右端方程， x和y为方程变量
    fun_xy = lambda x, y: -(x ** 2 + y ** 2) * np.exp(x * y)  # 例7
    # fun_xy = lambda x, y: -6 * (x + y)  # 例8

    # 泊松方程的精确解
    analytic_sol = lambda x, y: np.exp(x * y)  # 例7
    # exact_sol = lambda x, y: x ** 3 + y ** 3  # 例8

    left_boundary = lambda y: np.ones(len(y))  # u(0, y) = 1  例7
    # left_boundary = lambda y: y ** 3  # 例8

    right_boundary = lambda y: np.exp(y)  # u(1, y) = exp(y)  例7
    # right_boundary = lambda y: 1 + y ** 3  # 例8

    upper_boundary = lambda x: np.exp(x)  # u(x, 1) = exp(x)  例7
    # upper_boundary = lambda x: 1 + x ** 3  # 例8

    lower_boundary = lambda x: np.ones(len(x))  # u(x, 0) = 1  例7
    # lower_boundary = lambda x: x ** 3  # 例8
