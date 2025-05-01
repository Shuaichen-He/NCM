# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_hermite.py
@time:2021/08/29
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.hermite_interpolation import HermiteInterpolation
from interpolation_02.newton_difference_interpolation import NewtonDifferenceInterpolation

# fh = lambda x: 2 * np.exp(-x) * np.sin(x)  # 测试函数
# x = np.linspace(0, 2 * np.pi, 5, endpoint=True)  # 模拟五个点
# y = fh(x)
# dy = 2 * np.exp(-x) * (np.cos(x) - np.sin(x))  # 一阶导数值
# x0 = np.array([2.6, 4.0, 4.8])  # 求解指定点的插值
# hermite = HermiteInterpolation(x, y, dy)  # 埃尔米特插值
# hermite.fit_interp()  # 生成埃尔米特插值多项式
# print(hermite.poly_coefficient)  # 打印系数
# print(hermite.coefficient_order)  # 打印阶次
# y0 = hermite.predict_x0(x0)  # 求解指定点的插值
# print("插值点的值：", y0, "，误差：", fh(x0) - y0)
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# hermite.plt_interpolation(x0, y0, fh=fh, is_show=False)  # 埃尔米特插值可视化
# ndi = NewtonDifferenceInterpolation(x, y)
# ndi.fit_interp()
# plt.subplot(122)
# ndi.plt_interpolation(fh=fh, is_show=False)  # 牛顿前向差分插值可视化
# plt.show()

np.random.seed(100)
fh = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)
x = np.linspace(-1, 3, 6, endpoint=True)  # 模拟6个点
# x = -1 + 4 * np.random.rand(6)
# x = np.sort(x)
# print(x)
y = fh(x)  # 函数值
dy = 11 * np.cos(x) - 35 * np.sin(5 * x)
hermite = HermiteInterpolation(x, y, dy)  # 埃尔米特插值
hermite.fit_interp()  # 生成埃尔米特插值多项式
print(hermite.poly_coefficient)  # 打印系数
print(hermite.coefficient_order)  # 打印阶次
plt.figure(figsize=(14, 5))
plt.subplot(121)
hermite.plt_interpolation(fh=fh, is_show=False)  # 埃尔米特插值可视化
x = np.linspace(-1, 3, 10, endpoint=True)  # 模拟10个点
# x = -1 + 4 * np.random.rand(10)
# x = np.sort(x)
# print(x)
y = fh(x)  # 函数值
dy = 11 * np.cos(x) - 35 * np.sin(5 * x)
hermite = HermiteInterpolation(x, y, dy)  # 埃尔米特插值
hermite.fit_interp()  # 生成埃尔米特插值多项式
x0 = np.asarray([-0.9, -0.2, 1.5, 2.2, 2.7, 2.9])
y0 = hermite.predict_x0(x0)
print("离散数据点(xk, yk)对应的多项式插值\n", y0)
print("离散数据点x插值的误差：\n", fh(x0) - y0)
plt.subplot(122)
hermite.plt_interpolation(fh=fh, x0=x0, y0=y0, is_show=False)  # 埃尔米特插值可视化
plt.show()
