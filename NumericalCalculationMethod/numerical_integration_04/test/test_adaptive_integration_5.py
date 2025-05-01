# -*- coding: UTF-8 -*-
"""
@file_name: test_adaptive_integration.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.adaptive_integral_algorithm import AdaptiveIntegralAlgorithm
from util_font import *

int_fun = lambda x: np.exp(-x) * np.sin(x)
aia = AdaptiveIntegralAlgorithm(int_fun, int_interval=[0, 8], eps=1e-15)
aia.fit_int()
exact_val = (1 - np.exp(-8) * (np.cos(8) + np.sin(8))) / 2
print("积分值：", aia.int_value)
print("积分值：", aia.int_value, "误差：", exact_val - aia.int_value)
print(len(aia.x_node))

plt.figure(figsize=(14, 5))
xi = np.linspace(0, 8, 1000)
plt.subplot(121)
plt.plot(aia.x_node, int_fun(aia.x_node), "k-")
plt.fill_between(xi, int_fun(xi), color="c", alpha=0.5)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$f(x)$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title("被积函数及其积分区域", fontdict={"fontsize": 18})

plt.subplot(122)
n = plt.hist(aia.x_node, bins=np.linspace(0, 8, 11), rwidth=0.8, color="r", alpha=0.5)
plt.plot((n[1][:-1] + n[1][1:]) / 2, n[0], "ko-", lw=2)
plt.title("节点划分数分布的直方图", fontdict={"fontsize": 18})
plt.ylabel("$Frequency$", fontdict={"fontsize": 18})
plt.xlabel("$Bins$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()
