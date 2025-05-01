# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from function_approximation_03.chebyshev_zero_points_interp import ChebyshevZeroPointsInterpolation
from function_approximation_03.chebyshev_series_approximation import ChebyshevSeriesApproximation
from function_approximation_03.legendre_series_approximation import LegendreSeriesApproximation
from function_approximation_03.best_uniform_approximation import BestUniformApproximation
from function_approximation_03.best_square_approximation import BestSquarePolynomiaApproximation

fh = lambda x: np.sin(2 * x + np.pi / 6) * np.exp(-0.4 * x)

# 切比雪夫多项式零点插值逼近
orders = np.array([5, 20], dtype=np.int)
plt.figure(figsize=(14, 5))
for i, order in enumerate(orders):
    czpi = ChebyshevZeroPointsInterpolation(fh, orders=order, x_span=[-np.pi, np.pi])
    czpi.fit_approximation()
    print("最大逼近多项式绝对值误差：", czpi.max_abs_error)
    print("零点值：", czpi.terms_zeros)
    print("拉格朗日插值逼近多项式系数以及阶次：")
    print(czpi.poly_coefficient)
    print(czpi.polynomial_orders)
    plt.subplot(121 + i)
    czpi.plt_approximation(is_show=False, is_fh_marker=True)
plt.show()

# 切比雪夫级数逼近
t = sympy.Symbol("t")
fh_sym = sympy.sin(2 * t + np.pi / 6) * sympy.exp(-0.4 * t)
orders = np.array([5, 15, 25], dtype=np.int)
plt.figure(figsize=(18, 4))
for i, order in enumerate(orders):
    csa = ChebyshevSeriesApproximation(fh_sym, x_span=[-np.pi, np.pi], k=order)
    csa.fit_approximation()
    print("切比雪夫级数逼近最大逼近多项式绝对值误差：", csa.max_abs_error)
    plt.subplot(131 + i)
    if i == 0:
        csa.plt_approximate(is_show=False, is_fh_marker=False)
    else:
        csa.plt_approximate(is_show=False, is_fh_marker=True)
plt.show()
print("=" * 80)

# 勒让德级数逼近
plt.figure(figsize=(18, 4))
for i, order in enumerate(orders):
    lsa = LegendreSeriesApproximation(fh_sym, x_span=[-np.pi, np.pi], k=order)
    lsa.fit_approximation()
    print("勒让德级数逼近最大逼近多项式绝对值误差：", lsa.max_abs_error)
    plt.subplot(131 + i)
    if i == 0:
        lsa.plt_approximate(is_show=False, is_fh_marker=False)
    else:
        lsa.plt_approximate(is_show=False, is_fh_marker=True)
plt.show()

# 最佳一直逼近
plt.figure(figsize=(14, 5))
orders = [10, 20]
for i, order in enumerate(orders):
    plt.subplot(121 + i)
    bua = BestUniformApproximation(fh_sym, k=order, interval=[-np.pi, np.pi], eps=1e-10)
    bua.fit_approximation()
    bua.plt_approximate(is_show=False, is_fh_marker=True)
    print("order=%d" % order, "，最佳一致逼近多项式的逼近误差精度为：", bua.abs_error)
    print("order=%d" % order, "，最佳一致逼近多项式的最大绝对误差为：", bua.max_abs_error)
    print("交错点组：", bua.cross_point_group)
plt.show()

# 最佳平方逼近
plt.figure(figsize=(14, 5))
orders = [10, 20]
for i, order in enumerate(orders):
    plt.subplot(121 + i)
    bspa = BestSquarePolynomiaApproximation(fh_sym, k=order, interval=[-np.pi, np.pi])
    bspa.fit_approximation()
    bspa.plt_approximate(is_show=False, is_fh_marker=True)
    print("order=%d" % order, "，最佳平方逼近多项式的逼近误差精度为：", bspa.mae)
    print("order=%d" % order, "，最佳平方逼近多项式的最大绝对误差为：", bspa.max_abs_error)
plt.show()
