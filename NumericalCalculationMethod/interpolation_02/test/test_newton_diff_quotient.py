# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_newton_diff_quotient.py
@time:2021/08/29
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.newton_diff_quotient_interp import NewtonDifferenceQuotient
from interpolation_02.lagrange_interpolation import LagrangeInterpolation

fh = lambda x: 2 * np.exp(-x) * np.sin(x)  # 函数模拟

np.random.seed(1)
# x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
x = 0 + 2 * np.pi * np.random.rand(10)
x = np.sort(x)
print(x)
y = fh(x)
x0 = np.array([2.6, 4.0, 4.8])
nd = NewtonDifferenceQuotient(x, y)
nd.fit_interp()
# print(nd.diff_quot)  # 打印差商表
print(nd.poly_coefficient)
print(nd.coefficient_order)
y0 = nd.predict_x0(x0)
print("插值点的值：", y0, "，误差：", fh(x0) - y0)

plt.figure(figsize=(14, 5))
plt.subplot(121)
nd.plt_interpolation(x0, y0, fh=fh, is_show=False)
plt.subplot(122)
lag_iterp = LagrangeInterpolation(x, y)
lag_iterp.fit_interp()
y0 = lag_iterp.predict_x0(x0)
print(lag_iterp.poly_coefficient)
print(lag_iterp.coefficient_order)
print("插值点的值：", y0, "，误差：", fh(x0) - y0)
lag_iterp.plt_interpolation(x0, y0, fh, is_show=False)  # 可视化
plt.show()