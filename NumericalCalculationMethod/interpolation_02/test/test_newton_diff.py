# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_newton_diff.py
@time:2021/08/29
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.newton_difference_interpolation import NewtonDifferenceInterpolation

fh = lambda x: 2 * np.exp(-x) * np.sin(x)  # 函数模拟

x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
y = fh(x)
x0 = np.array([0.05, 1.5, 2.6, 4.0, 4.8])
nd = NewtonDifferenceInterpolation(x, y)
nd.fit_interp()
for i in range(10):
    for j in range(10):
        print("%.5e" % nd.diff_val[i, j], end=" ")
    print()
# print(nd.polynomial)
print(nd.poly_coefficient)
# print(nd.coefficient_order)
y0 = nd.predict_x0(x0)
print("插值点的值：", y0, "，误差：", fh(x0) - y0)
# y_interp = nd.predict_x0(x)
# print(y_interp)
# print(y - y_interp)
print("=" * 60)
plt.figure(figsize=(14, 5))
plt.subplot(121)
nd.plt_interpolation(x0, y0, fh=fh, is_show=False)

nd = NewtonDifferenceInterpolation(x, y, diff_method="backward")
nd.fit_interp()
for i in range(10):
    for j in range(10):
        print("%.5e" % nd.diff_val[i, j], end=" ")
    print()
# print(nd.polynomial)
print(nd.poly_coefficient)
print(nd.coefficient_order)
y0 = nd.predict_x0(x0)
print("插值点的值：", y0, "，误差：", fh(x0) - y0)
# y_interp = nd.predict_x0(x)
# print(y_interp)
# print(y - y_interp)
plt.subplot(122)
nd.plt_interpolation(x0, y0, fh=fh, is_show=False)
plt.show()