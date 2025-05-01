# -*- coding: UTF-8 -*-
"""
@file:test_chebyshev.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.adaptive_piecewise_linear_approximation import AdaptivePiecewiseLinearApproximation

runge_fun = lambda x: 1 / (1 + x ** 2)
# fun = lambda x: np.tan(np.cos((np.sqrt(3) + np.sin(2 * x)) / (3 + 4 * x ** 2)))  # 被逼近函数
fun = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 被逼近函数
apa = AdaptivePiecewiseLinearApproximation(runge_fun, interval=[-5, 5], eps=1e-8, max_split_nodes=2000)
apa.fit_approximation()
print("最大误差：", apa.max_error)
plt.figure(figsize=(14, 5))
plt.subplot(121)
apa.plt_approximate(is_show=False)
plt.subplot(122)
# bins = np.linspace(-np.pi, np.pi, 11)
bins = np.linspace(-5, 5, 11)
n = plt.hist(apa.node, bins=bins, rwidth=0.8, color="r", alpha=0.5)
plt.plot((n[1][:-1] + n[1][1:]) / 2, n[0], "ko-", lw=2)
plt.title("自适应分段线性逼近节点划分数量直方图", fontdict={"fontsize": 18})
plt.ylabel("$Frequency$", fontdict={"fontsize": 18})
plt.xlabel("$Bins$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()

# 绘制误差随着精度变化曲线
# eps = np.linspace(1e-15, 0.01, 30, endpoint=True)
# error, node_num = np.zeros(len(eps)), np.zeros(len(eps))
# for i in range(len(eps)):
#     apa = AdaptivePiecewiseLinearApproximation(fun, interval=[-np.pi, np.pi], eps=eps[i])
#     apa.fit_approximation()
#     error[i] = apa.max_error
#     node_num[i] = apa.node_num
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plt.plot(eps, error, "k--", markerfacecolor="r", markeredgecolor="r")
# plt.xlabel("$Precision$", fontdict={"fontsize": 18})
# plt.ylabel("$MAE$", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.title("最大绝对误差随着精度的提高而降低", fontdict={"fontsize": 18})
# plt.grid(ls=":")
# plt.subplot(122)
# plt.plot(eps, node_num, "k--", markerfacecolor="b", markeredgecolor="b")
# plt.xlabel("$Precision$", fontdict={"fontsize": 18})
# plt.ylabel("$Node \quad number$", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.title("划分节点数随着精度的提高而增加", fontdict={"fontsize": 18})
# plt.grid(ls=":")
# plt.show()
