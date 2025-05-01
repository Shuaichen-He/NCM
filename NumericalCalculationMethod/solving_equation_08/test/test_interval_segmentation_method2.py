# -*- coding: UTF-8 -*-
"""
@file:test_interval_segmentation_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.interval_segmentation_method import IntervalSegmentation_Root
from util_font import *

equation = lambda x: np.exp(-3 * x) * np.sin(4 * x + 2) + 4 * np.exp(-0.5 * x) * np.cos(2 * x) - 0.5  # 待求方程

xi = np.linspace(0, 5, 200)
yi = equation(xi)
plt.figure(figsize=(7, 5))
plt.plot(xi, yi, "-")
plt.show()


options = {"eps": 1e-16, "display": "display", "pltFuns": False}  # 参数设置，某些参数可不设置
funEvals = ["regula", "dichotomy"]
linestyle = ["-", "--", "-."]
roots, eps = [], []
plt.figure(figsize=(14, 5))
for i, funEval in enumerate(funEvals):
    ism = IntervalSegmentation_Root(equation, [0, 1], eps=1e-16, display="display", funEval=funEval)  # 实例化对象
    ism.fit_root()  # 求近似根
    print("%.25e, %.25f" %(equation(ism.root_precision_info[-1, -2]), ism.root_precision_info[-1, -2]))
    eps.append(ism.root_precision_info[:, -1])
    roots.append(ism.root_precision_info[:, -2])

plt.subplot(121)
plt.plot(np.arange(1, len(roots[0]) + 1), roots[0], "-s", lw=2, label="$%s: \ x^*=%.15f$" % (funEvals[0], roots[0][-1]))
plt.plot(np.arange(1, 16), roots[1][:15], "--o", lw=2, label="$%s: \ x^*=%.15f$" % (funEvals[1], roots[1][-1]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$x^*$", fontdict={"fontsize": 18})
plt.title("区间分割法近似根$x^*$的收敛曲线（前21次迭代）", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
plt.plot(np.arange(1, 11), eps[0][:10], "-s", lw=2, label="$%s: \ k=%d$" % (funEvals[0], len(eps[0])))
plt.plot(np.arange(1, 11), eps[1][:10], "--o", lw=2, label="$%s: \ k=%d$" % (funEvals[1], len(eps[1])))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$f(x^*)$", fontdict={"fontsize": 18})
plt.title("近似根$x^*$的精度$f(x^*)$收敛曲线（前10次迭代）", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
