# -*- coding: UTF-8 -*-
"""
@file:test_legendre.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from function_approximation_03.legendre_series_approximation import LegendreSeriesApproximation


t = sympy.Symbol("t")
# fun = sympy.sin(t) * sympy.exp(-t)
# fun = 1 / (2 - t)
# fun = 1 / (1 + t ** 2)
fun = sympy.exp(t)
x0 = np.array([-1, -0.5, 0.5, 1])
lsa = LegendreSeriesApproximation(fun, x_span=[-1, 1], k=3)
lsa.fit_approximation()
y0 = lsa.predict_x0(x0)
print("逼近点的值：", y0)
print("x0原函数的值：", np.exp(x0))
# print("x0原函数的值：", 1 / (x0 ** 2 + 1))
# print("x0原函数的值：", np.sin(x0) * np.exp(-x0))
# print("x0原函数的值：", 1 / (2 - x0))
lsa.plt_approximate()
print("递推项：")
print(lsa.T_coefficient[0])
print("递推项系数：")
print(lsa.T_coefficient[1])
print(lsa.approximation_poly)
print(lsa.max_abs_error)