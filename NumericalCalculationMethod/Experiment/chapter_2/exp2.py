# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.cubic_spline_interpolation import CubicSplineInterpolation
from interpolation_02.b_spline_interpolation import BSplineInterpolation

x = np.array([0.25, 0.30, 0.39, 0.45, 0.53])
y = np.array([0.500, 0.5477, 0.6245, 0.6708, 0.7280])
dy = np.array([1.0000, 0.6868])
d2y = np.array([0, 0])
boundary_cond = ["complete", "natural"]
plt.figure(figsize=(14, 5))
for i, bc in enumerate(boundary_cond):
    csi = CubicSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
    csi.fit_interp()
    print(csi.poly_coefficient)
    plt.subplot(121 + i)
    csi.plt_interpolation(is_show=False)
plt.show()

# 如下为龙格函数测试，可作为练习
# fh = lambda x: 1 / (1 + x ** 2)
# dfh = lambda x: -2 * x / (1 + x ** 2) ** 2  # 一阶导数
# d2fh = lambda x: (8 * x ** 2) / (x ** 2 + 1) ** 3 - 2 / (x ** 2 + 1) ** 2  # 二阶导数
#
#
# def test_csi(x, y, dy, d2y):
#     """
#     三次样条插值
#     :param x: 插值节点对应的x轴数值
#     :param y: 插值节点对应的y轴数值
#     :param dy: 边界点处出的一阶导数值
#     :param d2y: 边界点处的二阶导数值
#     :return:
#     """
#     boundary_cond = ["complete", "second", "natural", "periodic"]
#     plt.figure(figsize=(14, 10))
#     for i, bc in enumerate(boundary_cond):
#         csi = CubicSplineInterpolation(x, y, dy, d2y, boundary_cond=bc)
#         csi.fit_interp()
#         plt.subplot(221 + i)
#         csi.plt_interpolation(fh=fh, is_show=False)
#         if bc == "natural":
#             print(csi.poly_coefficient[[0, -1], ])  # 打印系数矩阵
#             print("=" * 60)
#     plt.show()
#
#
# def test_bsi(x, y, dy, d2y):
#     # 三次均匀B样条插值
#     boundary_cond = ["complete", "second", "natural", "periodic"]
#     plt.figure(figsize=(14, 10))
#     for i, bc in enumerate(boundary_cond):
#         bsi = BSplineInterpolation(x, y, dy, d2y, boundary_cond=bc)
#         bsi.fit_interp()
#         plt.subplot(221 + i)
#         bsi.plt_interpolation(fh=fh, is_show=False)
#         if bc == "natural":
#             print(bsi.poly_coefficient[[0, -1], ])  # 打印系数矩阵
#             print("=" * 60)
#     plt.show()
#
#
# x_10 = np.linspace(-5, 5, 10)  # 等分10个值
# y_10, dy_10, d2y_10 = fh(x_10), dfh(x_10[[0, 9]]), d2fh(x_10[[0, 9]])  # 模拟函数值，边界的一阶和二阶导数值
# test_csi(x_10, y_10, dy_10, d2y_10)  # 三次样条插值
# test_bsi(x_10, y_10, dy_10, d2y_10)  # 三次均匀B样条插值
#
# x_20 = np.linspace(-5, 5, 20)  # 等分20个值
# y_20, dy_20, d2y_20 = fh(x_20), dfh(x_20[[0, 19]]), d2fh(x_20[[0, 19]])  # 模拟函数值，边界的一阶和二阶导数值
# test_csi(x_20, y_20, dy_20, d2y_20)
# test_bsi(x_20, y_20, dy_20, d2y_20)
