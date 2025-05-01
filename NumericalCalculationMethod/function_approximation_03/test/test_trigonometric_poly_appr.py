# -*- coding: UTF-8 -*-
"""
@file:test_trigonometric_poly_appr.py
@IDE:PyCharm   Python:3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.trigonometric_polynomial_appr import TrigonometricPolynomialApproximation

fun = lambda x: x ** 4 - 3 * x ** 3 + 2 * x ** 2 - np.tan(x * (x - 2))

fun2 = lambda x: np.sin(x) * np.exp(-x)

# 例8测试代码
# p_num = [8, 24]  # 离散点数
# plt.figure(figsize=(14, 5))
# for i, p in enumerate(p_num):
#     x = np.linspace(0, 2, p, endpoint=False)
#     tpa = TrigonometricPolynomialApproximation(y=fun(x), interval=[0, 2], fun=fun)
#     tpa.fit_approximation()
#     print("正弦项系数：\n", tpa.Bk)
#     print("余弦项系数：\n", tpa.Ak)
#     print("傅里叶逼近多项式：\n", tpa.approximation_poly)
#     plt.subplot(121 + i)
#     tpa.plt_approximate(is_show=False, is_fh_marker=True)  # 可视化
# plt.show()
#
# plt.figure(figsize=(14, 5))
# x = np.linspace(0, 2 * np.pi, 25, endpoint=True)
# tpa = TrigonometricPolynomialApproximation(y=fun2(x), interval=[0, 2 * np.pi], fun=fun2)
# tpa.fit_approximation()
# plt.subplot(121)
# tpa.plt_approximate(is_show=False)  # 可视化
# x = np.linspace(0, 2 * np.pi, 50, endpoint=True)
# tpa = TrigonometricPolynomialApproximation(y=fun2(x), interval=[0, 2 * np.pi], fun=fun2)
# tpa.fit_approximation()
# plt.subplot(122)
# tpa.plt_approximate(is_show=False)  # 可视化
# plt.show()
#
# time = np.linspace(1, 24, 24)
# temperature = np.array([58, 58, 58, 58, 57, 57, 57, 58, 60, 64, 67, 68,
#                         66, 66, 65, 64, 63, 63, 62, 61, 60, 60, 59, 58])
# tpa = TrigonometricPolynomialApproximation(temperature, interval=[1, 24])
# tpa.fit_approximation()
# print("正弦项系数：\n", tpa.Bk)
# print("余弦项系数：\n", tpa.Ak)
# print("傅里叶逼近多项式：\n", tpa.approximation_poly)
# tpa.plt_approximate(is_show=True, x0=time, y0=temperature)  # 可视化

fh = lambda x: np.cos(np.pi * x) - 2 * np.sin(np.pi * x)
# fh = lambda x: x ** 2 * np.cos(4 * x)
plt.figure(figsize=(14, 5))
x = np.linspace(-np.pi, np.pi, 8, endpoint=False)
tpa = TrigonometricPolynomialApproximation(y=fh(x), interval=[-3, 3], fun=fh)
tpa.fit_approximation()
print("正弦项系数：\n", tpa.Bk)
print("余弦项系数：\n", tpa.Ak)
print("傅里叶逼近多项式：\n", tpa.approximation_poly)
plt.subplot(121)
tpa.plt_approximate(is_show=False)  # 可视化
x = np.linspace(-np.pi, np.pi, 16, endpoint=False)
tpa = TrigonometricPolynomialApproximation(y=fh(x), interval=[-3, 3], fun=fh)
tpa.fit_approximation()
print("正弦项系数：\n", tpa.Bk)
print("余弦项系数：\n", tpa.Ak)
print("傅里叶逼近多项式：\n", tpa.approximation_poly)
plt.subplot(122)
tpa.plt_approximate(is_show=False)  # 可视化
plt.show()
