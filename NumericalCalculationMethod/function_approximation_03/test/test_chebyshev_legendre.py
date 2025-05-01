# -*- coding: UTF-8 -*-
"""
@file_name:test_chebyshev_legendre.py
@copyright: http://maths.xynu.edu.cn
"""

import sympy
from function_approximation_03.chebyshev_zero_points_interp import ChebyshevZeroPointsInterpolation
from function_approximation_03.chebyshev_series_approximation import ChebyshevSeriesApproximation
from function_approximation_03.legendre_series_approximation import LegendreSeriesApproximation
import matplotlib.pyplot as plt
import numpy as np

t = sympy.Symbol("t")
runge_fun, sym_fun = lambda x: 1 / (x ** 2 + 1), 1 / (t ** 2 + 1)  # 龙格函数
fun = lambda x: np.sin(2 * x + np.pi / 6) * np.exp(-0.4 * x)
plt.figure(figsize=(19, 9))
for i, order in enumerate([10, 25]):
    plt.subplot(231 + 3 * i)  # 321, 322
    czpi = ChebyshevZeroPointsInterpolation(runge_fun, x_span=[-5, 5], orders=order)
    czpi.fit_approximation()
    if i == 0:
        czpi.plt_approximation(is_show=False)
    else:
        czpi.plt_approximation(is_show=False, is_fh_marker=True)
    print("切比雪夫多项式零点插值逼近最大绝对值误差是：", czpi.max_abs_error)
    print("=" * 70)
    plt.subplot(232 + 3 * i)  # 323, 324
    csa = ChebyshevSeriesApproximation(sym_fun, x_span=[-5, 5], k=order)
    csa.fit_approximation()
    if i == 0:
        csa.plt_approximate(is_show=False)
    else:
        csa.plt_approximate(is_show=False, is_fh_marker=True)
    print("切比雪夫级数逼近最大绝对值误差是：", csa.max_abs_error)
    print("=" * 70)
    plt.subplot(233 + 3 * i)  # 325, 326
    lsa = LegendreSeriesApproximation(sym_fun, x_span=[-5, 5], k=order)
    lsa.fit_approximation()
    if i == 0:
        lsa.plt_approximate(is_show=False)
    else:
        lsa.plt_approximate(is_show=False, is_fh_marker=True)
    print("勒让德级数逼近最大绝对值误差是：", lsa.max_abs_error)
plt.show()

