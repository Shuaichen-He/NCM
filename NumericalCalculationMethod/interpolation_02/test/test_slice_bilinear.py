# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_piecewise_bilinear.py
@time:2021/08/27
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interpolation_02.slice_bilinear_interpolation import SliceBiLinearInterpolation

fh = lambda x, y: np.sin(x) * np.cos(y)
x = np.linspace(1, 6, 10, endpoint=True)  # 等距划分
y = np.linspace(2, 7, 10, endpoint=True)
xi, yi = np.meshgrid(x, y)
Z = np.sin(xi) * np.cos(yi)
x0 = np.array([2.08, 1.3, 4.6, 2.98])
y0 = np.array([3.77, 2.7, 4.5, 6.08])
sbi = SliceBiLinearInterpolation(x, y, Z.T, x0, y0)
z0 = sbi.fit_2d_interp()
print("插值点值：", z0)
print("误差：", np.sin(x0) * np.cos(y0) - z0)
fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(121, projection='3d')
sbi.plt_3d_surface(ax, "节点数$10$", fh=fh)

ax = fig.add_subplot(122, projection='3d')
x = np.linspace(1, 6, 25, endpoint=True)  # 等距划分
y = np.linspace(2, 7, 25, endpoint=True)
xi, yi = np.meshgrid(x, y)
Z = np.sin(xi) * np.cos(yi)
sbi = SliceBiLinearInterpolation(x, y, Z.T, x0, y0)
z0 = sbi.fit_2d_interp()
print("插值点值：", z0)
print("误差：", np.sin(x0) * np.cos(y0) - z0)
sbi.plt_3d_surface(ax, "节点数$25$", fh=fh)
plt.show()

x = np.linspace(0, 5600, 15, endpoint=True)
y = np.linspace(0, 4800, 13, endpoint=True)
data = pd.read_csv("../data/mountain.csv", header=None).values
x0 = [1270, 2080, 3860, 5200]
y0 = [1690, 3770, 2480, 4690]
sbi = SliceBiLinearInterpolation(x, y, data.T, x0=x0, y0=y0)
z0 = sbi.fit_2d_interp()
print(z0)
sbi.plt_3d_surface_contourf()
