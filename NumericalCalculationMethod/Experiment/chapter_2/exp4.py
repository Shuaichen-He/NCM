# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.slice_bilinear_interpolation import SliceBiLinearInterpolation
from interpolation_02.bivariate_three_points_lagrange import BivariateThreePointsLagrange

fh = lambda x, y: x * np.exp(-x ** 2 - y ** 2)
x = np.linspace(-2, 2, 25, endpoint=True)  # 等距划分
y = np.linspace(-2, 2, 25, endpoint=True)
xi, yi = np.meshgrid(x, y)
Z = xi * np.exp(-xi ** 2 - yi ** 2)
x0 = np.array([-1.50, -0.58, 0.58, 1.65])
y0 = np.array([-1.25, -0.69, 0.78, 1.78])

# 分片双线性插值
sbi = SliceBiLinearInterpolation(x, y, Z.T, x0, y0)
z0 = sbi.fit_2d_interp()
print("插值点值：", z0)
print("分片双线性插值误差：", x0 * np.exp(-x0 ** 2 - y0 ** 2) - z0)
fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(121, projection='3d')
sbi.plt_3d_surface(ax, "节点数$25$", fh=fh)

# 二元三点拉格朗日插值
btpl = BivariateThreePointsLagrange(x, y, Z.T, x0, y0)
z0 = btpl.fit_interp_2d()
print("插值点值：", z0)
print("二元三点拉格朗日插值误差：", x0 * np.exp(-x0 ** 2 - y0 ** 2) - z0)
ax = fig.add_subplot(122, projection='3d')
btpl.plt_3d_surface(ax, "节点数$25$", fh=fh)
plt.show()
