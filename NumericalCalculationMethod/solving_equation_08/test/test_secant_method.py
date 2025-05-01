# -*- coding: UTF-8 -*-
"""
@file:test_secant_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from solving_equation_08.double_points_secant_method import DoublePointsSecantMethod

# t = sympy.Symbol("t")
# equ = (t - 1) * (sympy.sin(t - 1) + 3 * t) - t ** 3 + 1
# sympy.plot(equ, (t, 0, 2))

fx1 = lambda x: (x - 1) * (np.sin(x - 1) + 3 * x) - x ** 3 + 1
fx2 = lambda x: np.log10(x) + np.sqrt(x) - 2
fx3 = lambda x: x ** 4 + 2 * x ** 2 - x - 3

sm = DoublePointsSecantMethod(fx1, [0.7, 1.25], eps=1e-16, display="display")
sm.fit_root()
sm = DoublePointsSecantMethod(fx1, [1.75, 2], eps=1e-16, display="display")
sm.fit_root()