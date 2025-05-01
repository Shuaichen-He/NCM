# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.piecewise_linear_interpolation import PiecewiseLinearInterpolation
from interpolation_02.cubic_spline_interpolation import CubicSplineInterpolation
from interpolation_02.b_spline_interpolation import BSplineInterpolation

x = np.linspace(0, 8.540, 13)
y = np.array([0, 0.914, 5.060, 7.772, 8.717, 9.083, 9.144, 9.083, 8.722, 7.687, 5.376, 1.073, 0])
xi = np.arange(0, 8.550, 0.01)  # 间隔0.01，推测插值

plt.figure(figsize=(14, 5))
pli = PiecewiseLinearInterpolation(x, y)
pli.fit_interp()
yi_pli = pli.predict_x0(xi)  # 推测
plt.subplot(121)
pli.plt_interpolation(is_show=False)

bsi = BSplineInterpolation(x, y)
bsi.fit_interp()
yi_bsi = bsi.predict_x0(xi)  # 推测
plt.subplot(122)
bsi.plt_interpolation(is_show=False)
plt.show()

# 面积计算
S1 = np.sum((yi_pli[:-1] + yi_pli[1:]) / 2) * 0.01
S2 = np.sum((yi_bsi[:-1] + yi_bsi[1:]) / 2) * 0.01
print("分段线性插值得面积：%.5f，三次均匀B样条插值得面积：%.5f" % (S1, S2))
