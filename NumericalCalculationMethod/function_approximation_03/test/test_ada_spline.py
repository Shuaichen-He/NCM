# -*- coding: UTF-8 -*-
"""
@file:test_ada_spline.py
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.adaptive_spline_approximation import AdaptiveSplineApproximation
from function_approximation_03.adaptive_piecewise_linear_approximation \
    import AdaptivePiecewiseLinearApproximation

runge_fun = lambda x: 1 / (1 + x ** 2)
# asa = AdaptiveSplineApproximation(runge_fun, interval=[-5, 5], eps=1e-6)
fun = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 被逼近函数
asa = AdaptiveSplineApproximation(fun, interval=[-np.pi, np.pi], eps=1e-6, max_split_nodes=2000)
asa.fit_approximation()
print(asa.node_num)

# 可视化最终自适应节点的分布情况
plt.figure(figsize=(14, 5))
plt.subplot(121)
asa.plt_approximate(is_show=False)
plt.subplot(122)
bins = np.linspace(-np.pi, np.pi, 11)
n = plt.hist(asa.node, bins=bins, rwidth=0.8, color="r", alpha=0.5)
plt.plot((n[1][:-1] + n[1][1:]) / 2, n[0], "ko-", lw=2)
plt.title("自适应三次样条逼近节点划分数量直方图", fontdict={"fontsize": 18})
plt.ylabel("$Frequency$", fontdict={"fontsize": 18})
plt.xlabel("$Bins$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16

# y_val = 1 / (np.array(asa.node) ** 2 + 1)
# plt.plot(asa.node, y_val, "ko-")
# plt.title("自适应三次样条逼近划分节点密度", fontdict={"fontsize": 18})
# plt.xlabel("$Nodes$", fontdict={"fontsize": 18})
# plt.ylabel("$Function \quad Values$", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()

# 绘制误差随着精度变化曲线
eps = np.linspace(1e-5, 0.01, 20, endpoint=True)
# 自适应分段样条逼近
error_s, node_num_s = np.zeros(len(eps)), np.zeros(len(eps))
for i in range(len(eps)):
    asa = AdaptiveSplineApproximation(runge_fun, interval=[-5, 5], eps=eps[i])
    asa.fit_approximation()
    error_s[i], node_num_s[i] = asa.max_error, asa.node_num
# 自适应分段线性逼近
error_l, node_num_l = np.zeros(len(eps)), np.zeros(len(eps))
for i in range(len(eps)):
    apla = AdaptivePiecewiseLinearApproximation(runge_fun, interval=[-5, 5], eps=eps[i])
    apla.fit_approximation()
    error_l[i], node_num_l[i] = apla.max_error, apla.node_num

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.semilogy(eps, error_l, "o-", lw=1.5, label="自适应分段线性逼近")
plt.semilogy(eps, error_s, "*-", lw=1.5, label="自适应三次样条逼近")
plt.xlabel("$Precision$", fontdict={"fontsize": 18})
plt.ylabel("最大绝对值误差", fontdict={"fontsize": 18})
plt.title("两种自适应逼近算法的最大绝对值误差变化曲线", fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.legend(frameon=False, fontsize=16)

plt.subplot(122)
plt.semilogy(eps, node_num_l, "o-", lw=1.5, label="自适应分段线性逼近")
plt.semilogy(eps, node_num_s, "*-", lw=1.5, label="自适应三次样条逼近")
print(node_num_s)
plt.xlabel("$Precision$", fontdict={"fontsize": 18})
plt.ylabel("节点划分数", fontdict={"fontsize": 18})
plt.title("两种自适应逼近算法的节点划分数变化曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.ylim([1e+01, 1e+03])
plt.show()
