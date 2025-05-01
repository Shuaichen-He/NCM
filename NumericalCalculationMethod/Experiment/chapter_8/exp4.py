# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.double_points_secant_method import DoublePointsSecantMethod
from solving_equation_08.parabola_method import ThreePointParabolaMethod
from Experiment.util_font import *

fx = lambda x: 4 * x ** 4 - 10 * x ** 3 + 1.25 * x ** 2 + 5 * x - 0.5

# 可视化，确定有根区间
xi = np.linspace(-1, 2.5, 200)
yi = fx(xi)
plt.figure(figsize=(7, 5))
plt.plot(xi, yi, "-", lw=2)
plt.plot(xi, np.zeros(len(xi)), "-", lw=0.5)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("非线性方程在区间$[-1, 2.5]$内的有根情况", fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()

x_span = np.array([[-0.7, -0.5], [-0.2, 0.2], [0.8, 1.2], [2.0, 2.1]])
for i, span in enumerate(x_span):
    sm = DoublePointsSecantMethod(fx, span, eps=1e-16, display="display") # 弦截法
    sm.fit_root()
    pm = ThreePointParabolaMethod(fx, span, eps=1e-16, display="display")  # 抛物线法
    pm.fit_root()
    print("=" * 80)
