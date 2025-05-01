# -*- coding: UTF-8 -*-
"""
@file_name:test_chebyshev_zero_interp.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.chebyshev_zero_points_interp import ChebyshevZeroPointsInterpolation

# 例1测试代码
fun = lambda x: np.exp(x)  # 被逼近函数
plt.figure(figsize=(14, 5))
czpi = ChebyshevZeroPointsInterpolation(fun, orders=4, x_span=[0, 1])
czpi.fit_approximation()
print("最大逼近多项式绝对值误差：", czpi.max_abs_error)
print("零点值：", czpi.terms_zeros)
print("拉格朗日插值逼近多项式系数以及阶次：")
print(czpi.poly_coefficient)
print(czpi.polynomial_orders)
plt.subplot(121)
czpi.plt_approximation(is_show=False, is_fh_marker=True)
orders = np.linspace(4, 16, 13, dtype=int)
mae = []
for k in orders:
    czpi = ChebyshevZeroPointsInterpolation(fun, orders=k, x_span=[0, 1])
    czpi.fit_approximation()
    mae.append(czpi.mae)
    print(k, czpi.mae)
plt.subplot(122)
plt.plot(orders, mae, "o-")
idx = np.argmin(mae)
plt.semilogy(orders[idx], mae[idx], "D", label="$k=%d, MAE_{10}=%.2e$" % (orders[idx], mae[idx]))
plt.legend(frameon=False, fontsize=18)
plt.xlabel("$Orders(k)$", fontsize=18)
plt.ylabel("$MAE_{10}$", fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(ls=":")
plt.title("不同阶次下逼近的绝对误差均值", fontsize=18)
plt.show()

fun2 = lambda x: 1 / (1 + x ** 2)  # 龙格函数
fun3 = lambda x: np.tan(np.cos((np.sqrt(3) + np.sin(2 * x)) / (3 + 4 * x ** 2)))  # 被逼近函数

# 例2测试代码
# fun4 = lambda x: np.sin(2 * x) ** 2 * np.exp(-0.5 * x)
# plt.figure(figsize=(14, 5))
# czpi = ChebyshevZeroPointsInterpolation(fun4, orders=10, x_span=[-3, 3])
# czpi.fit_approximation()
# print("最大逼近多项式绝对值误差：", czpi.max_abs_error)
# print("零点值：", czpi.terms_zeros)
# print("拉格朗日插值逼近多项式系数以及阶次：")
# print(czpi.poly_coefficient)
# print(czpi.polynomial_orders)
# plt.subplot(121)
# czpi.plt_approximation(is_show=False)
#
# czpi = ChebyshevZeroPointsInterpolation(fun4, orders=20, x_span=[-3, 3])
# czpi.fit_approximation()
# print("最大逼近多项式绝对值误差：", czpi.max_abs_error)
# print("零点值：", czpi.terms_zeros)
# print("拉格朗日插值逼近多项式系数以及阶次：")
# print(czpi.poly_coefficient)
# print(czpi.polynomial_orders)
# plt.subplot(122)
# czpi.plt_approximation(is_show=False, is_fh_marker=True)
# plt.show()


