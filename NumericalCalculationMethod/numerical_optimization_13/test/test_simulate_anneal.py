# -*- coding: UTF-8 -*-
"""
@file_name: test_sa.py
@time: 2022-08-15
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_optimization_13.simulate_anneal import SimulatedAnnealingOptimization
from util_font import *

# 一元函数优化
# func = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 例2
# sa = SimulatedAnnealingOptimization(func, [[-6, 4]], eps=1e-16)  # 例2
# sy, sx = sa.fit_optimize()
#
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# xi = np.linspace(-6, 4, 100)
# yi = func(xi)
# plt.plot(xi, yi, "k-", label="函数曲线")
# plt.plot(sx, sy, "ro", label="最小值")
# plt.xlabel("$x$", fontdict={"fontsize": 18})
# plt.ylabel("$f(x)$", fontdict={"fontsize": 18})
# plt.title("最小值点：$f(%.8f) = %.8f$" % (sx, sy), fontdict={"fontsize": 18})
# plt.grid(ls=":")
# plt.legend(fontsize=16, frameon=False)
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.subplot(122)
# y_vals = np.asarray(sa.best_y_optlist)
# plt.plot(y_vals)  # 最小值
# plt.grid(ls=":")
# plt.xlabel("迭代次数", fontdict={"fontsize": 18})
# plt.ylabel("最优值", fontdict={"fontsize": 18})
# plt.title("模拟退火算法最值的优化过程", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.show()


# def func(X):
#     x, y = X[0], X[1]
#     z = 21.5 + x * np.sin(4 * np.pi * x) + y * np.sin(20 * np.pi * y)
#     return z

# # 例8
# def func(X):
#     x, y = X[0], X[1]
#     z = x ** 2 + y ** 2 - 10 * np.cos(2 * np.pi * x) - 10 * np.cos(2 * np.pi * y) + 20
#     return z


# 例9
# def func(X):
#     x, y = X[0], X[1]
#     z = 0.5 + ((np.sin(np.sqrt(x ** 2 + y ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2) ** 2)
#     return z

# 例3
def func(X):
    x, y = X[0], X[1]
    z = (x ** 2 - 2 * x) * np.exp(-x ** 2 - y ** 2 - x * y)
    return z


sa = SimulatedAnnealingOptimization(func, [[-3, 3], [-3, 3]], epochs=100)  # 例3
# sa = SimulatedAnnealingOptimization(func, [[-3, 13], [4.1, 5.8]], epochs=100)
# sa = SimulatedAnnealingOptimization(func, [[-6, 4]], eps=1e-16)  # 例2
sy, sx = sa.fit_optimize()
print("最优值%.15f，解：%.15f, %.15f" % (sy, sx[0], sx[1]))
fig = plt.figure(figsize=(14, 5))
# ax = fig.add_subplot(111, projection='3d')
ax1 = fig.add_subplot(121)
# x = np.linspace(-3, 13, 100)
# y = np.linspace(4.1, 5.8, 50)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
xi, yi = np.meshgrid(x, y)
# zi = 21.5 + xi * np.sin(4 * np.pi * xi) + yi * np.sin(20 * np.pi * yi)
# zi = 4 * xi ** 2 - 2.1 * xi ** 4 + xi ** 6 / 3 + xi * yi - 4 * yi ** 2 + 4 * yi ** 4
# zi = xi ** 2 + yi ** 2 - 10 * np.cos(2 * np.pi * xi) - 10 * np.cos(2 * np.pi * yi) + 20
zi = (xi ** 2 - 2 * xi) * np.exp(-xi ** 2 - yi ** 2 - xi * yi)
# zi = 0.5 + ((np.sin(np.sqrt(xi ** 2 + yi ** 2))) ** 2 - 0.5) / (1 + 0.001 * (xi ** 2 + yi ** 2) ** 2)
cs = ax1.contour(xi, yi, zi, levels=20)
ax1.clabel(cs, inline=True, fontsize=10)
plt.plot(sx[0], sx[1], "r.", markersize=10)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
# 需要修改标题中最大最小字样，和目标值前的负号
plt.title("最大值点：$f(%.8f, %.8f) = %.8f$" % (sx[0], sx[1], -sy), fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16

plt.subplot(122)

# 如下为绘制函数只优化曲线
y_vals = np.asarray(sa.best_y_optlist)
plt.plot(y_vals)  # 最小值
plt.grid(ls=":")
plt.xlabel("迭代次数", fontdict={"fontsize": 18})
plt.ylabel("最优值", fontdict={"fontsize": 18})
plt.title("模拟退火算法最值的优化过程", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16

# 如下为绘制三维曲面
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(xi, yi, zi, cmap=plt.get_cmap("YlGnBu"))
# plt.plot(sx[0], sx[1], -sy, "r.", markersize=10)
# plt.xlabel("$x$", fontdict={"fontsize": 18})
# plt.ylabel("$y$", fontdict={"fontsize": 18})
# plt.title("目标函数的三维曲面图及其最值点", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()
