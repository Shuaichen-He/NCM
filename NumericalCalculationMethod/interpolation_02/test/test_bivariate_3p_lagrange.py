# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_bivariate_3p_lagrange.py
@time:2021/08/27
"""
import numpy as np
import pandas as pd
from interpolation_02.bivariate_three_points_lagrange import BivariateThreePointsLagrange

fh = lambda x, y: np.sin(x) * np.cos(y)
x = np.linspace(1, 6, 10, endpoint=True)
y = np.linspace(2, 7, 10, endpoint=True)
# fh = lambda x, y: x * np.exp(-x ** 2 - y ** 2)
# x = np.linspace(-2, 2, 25, endpoint=True)
# y = np.linspace(-2, 2, 25, endpoint=True)
xi, yi = np.meshgrid(x, y)
Z = np.sin(xi) * np.cos(yi)
# Z = xi * np.exp(-xi ** 2 - yi ** 2)
# x0 = np.array([-1.5, -0.58, 0.58, 1.65])
# y0 = np.array([-1.25, -0.69, 0.78, 1.78])
x0 = np.array([2.08, 1.3, 4.6, 2.98])
y0 = np.array([3.77, 2.7, 4.5, 6.08])
btpl = BivariateThreePointsLagrange(x, y, Z.T, x0, y0)
z0 = btpl.fit_interp_2d()
# print("插值点值：", z0, "误差：", x0 * np.exp(-x0 ** 2 - y0 ** 2) - z0)
print("插值点值：", z0, "误差：", np.sin(x0) * np.cos(y0) - z0)
btpl.plt_3d_surface_contourf(fh=fh)

x = np.linspace(0, 5600, 15, endpoint=True)
y = np.linspace(0, 4800, 13, endpoint=True)
data = pd.read_csv("../data/mountain.csv", header=None).values
x0 = [1270, 2080, 3860, 5200]
y0 = [1690, 3770, 2480, 4690]
btpl = BivariateThreePointsLagrange(x, y, data.T, x0=x0, y0=y0)
z0 = btpl.fit_interp_2d()
print("插值点值：", z0)
btpl.plt_3d_surface_contourf()
