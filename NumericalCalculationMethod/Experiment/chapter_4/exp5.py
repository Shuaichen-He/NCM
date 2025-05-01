# -*- coding: UTF-8 -*-
"""
@file_name: exp5.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_integration_04.average_parabolic_interpolation_integral import AverageParabolicInterpolationIntegral
from interpolation_02.b_spline_interpolation import BSplineInterpolation
from numerical_integration_04.cubic_bspline_interpolation_integration import CubicBSplineInterpolationIntegration

x = np.linspace(0, 2, 16)
y = np.array([5.36205899, 8.28004495, 10.8856753, 10.94805195, 9.74022299, 9.74022299, 10.94805195,
              10.8856753, 8.28004495, 5.36205899, 3.33123078, 2.05908083, 1.25357447, 0.72289804,
              0.35817702, 0.09780694])

# 平均抛物插值
apii = AverageParabolicInterpolationIntegral(x, y)
apii.fit_int()
print("积分值：%.15f" % apii.int_value)

# 样条函数插值
cbsi = CubicBSplineInterpolationIntegration(x, y)
cbsi.fit_int()
print("积分值：%.15f" % cbsi.int_value)

# 可视化
bsi = BSplineInterpolation(x, y, boundary_cond="natural")
bsi.fit_interp()
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(x, y, "o-", lw=1.5)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("原试验数据折线图", fontsize=18)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
bsi.plt_interpolation(is_show=False)
plt.legend(frameon=False, fontsize=16)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("三次均匀$B$样条插值曲线", fontsize=18)
plt.show()