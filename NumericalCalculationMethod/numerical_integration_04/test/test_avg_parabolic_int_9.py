# -*- coding: UTF-8 -*-
"""
@file_name: test_avg_parabolic_int_9.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.average_parabolic_interpolation_integral import AverageParabolicInterpolationIntegral
from util_font import *


fh = lambda x: x ** 2 * np.sqrt(1 - x ** 2)

points_num = np.arange(5, 101, 1)
int_error = np.zeros(len(points_num))
for i, num in enumerate(points_num):
    x = np.linspace(0, 1, num)
    apii = AverageParabolicInterpolationIntegral(x, fh(x))
    apii.fit_int()
    int_error[i] = np.abs(np.pi / 16 - apii.int_value)
    print("离散数据点数：%d，积分值：%.15f，误差：%.10e" % (num, apii.int_value, int_error[i]))

plt.figure(figsize=(14, 5))
xi = np.linspace(0, 1, 150)
yi = fh(xi)
plt.subplot(121)
plt.plot(xi, yi, "k-", lw=1)
plt.fill_between(xi, yi, color="c", alpha=0.5)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$f(x)$", fontdict={"fontsize": 18})
plt.title("被积函数$f(x)=x^2\sqrt{(1-x^2)} \quad x \in [0, 1]$的积分区域", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.subplot(122)
plt.semilogy(points_num, int_error, "-", lw=1.5, label="$\epsilon$")
plt.semilogy(points_num[-1], int_error[-1], "o", label="$n=%d, \ \epsilon=%.3e$" % (points_num[-1], int_error[-1]))
plt.xlabel("离散数据量$n$", fontdict={"fontsize": 18})
plt.ylabel(r"$\epsilon=\vert I - I^* \vert$", fontdict={"fontsize": 18})
plt.title("平均抛物插值积分误差精度收敛曲线", fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.legend(frameon=False, fontsize=16)
plt.ylim([1e-4, 0.05])
plt.show()