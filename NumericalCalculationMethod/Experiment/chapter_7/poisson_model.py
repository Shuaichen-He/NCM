# -*- coding: UTF-8 -*-
"""
@file_name: poisson_model.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

class PoissonModel:
    """
    泊松方程模型，按照二维坐标系命名，u(0, y)为左边界，u(x, 0)为下边界，依次类推
    """
    fun_xy = lambda x, y: -6 * (x + y)  # 泊松方程右端方程， x和y为方程变量
    analytic_sol = lambda x, y: x ** 3 + y ** 3  # 泊松方程的精确解
    left_boundary = lambda y: y ** 3  # u(0, y)
    right_boundary = lambda y: 1 + y ** 3  # u(1, y)
    upper_boundary = lambda x: 1 + x ** 3  # u(x, 0)
    lower_boundary = lambda x: x ** 3  # u(x, 1)
