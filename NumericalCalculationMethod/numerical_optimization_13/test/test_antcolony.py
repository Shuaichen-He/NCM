# -*- coding: UTF-8 -*-
"""
@file_name: test_antcolony.py
@time: 2022-09-15
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_optimization_13.ant_colony_algorithm import AntColonyAlgorithmOptimization
from util_font import *

def func(X):
    x, y = X[:, 0], X[:, 1]  # 便于矢量化计算
    fxy = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(-(x ** 2 + y ** 2) / 10)
    return fxy  # 最小值修改为 -1 * fxy

n = 1.5
# args_span = [[-n, n], [-n, n]]
# aco = AntColonyAlgorithmOptimization(func, args_span, ant_m=50, step=0.1, rho=0.8, tp_c=0.6,
#                                      is_Maximum=True, max_iter=1000)
# best_x, best_y = aco.fit_optimize()
# print("最优值%.15f，解：%.15f, %.15f" %(best_y, best_x[0], best_x[1]))

fig = plt.figure(figsize=(14, 5))
# ax = fig.add_subplot(111, projection='3d')
ax1 = fig.add_subplot(121)
x = np.linspace(-n, n, 100)
y = np.linspace(-n, n, 100)
xi, yi = np.meshgrid(x, y)
zi = np.cos(2 * np.pi * xi) * np.cos(2 * np.pi * yi) * np.exp(-(xi ** 2 + yi ** 2) / 10)
cs = ax1.contour(xi, yi, zi, levels=20)
ax1.clabel(cs, inline=True, fontsize=10)
# plt.plot(best_x[0], best_x[1], "r.", markersize=10)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
# plt.title("最小值点：$f(%.8f, %.8f) = %.8f$" % (best_x[0], best_x[1], best_y), fontdict={"fontsize": 18})
plt.title("目标函数的等值线图", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.subplot(122)
# y_vals = np.asarray(aco.optimizing_best_f_val)
# plt.plot(y_vals)  # 最小值
# plt.grid(ls=":")
# plt.xlabel("迭代次数", fontdict={"fontsize": 18})
# plt.ylabel("最优值", fontdict={"fontsize": 18})
# plt.title("蚁群算法算法最值的优化过程", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(xi, yi, zi, cmap=plt.get_cmap("rainbow"))
# plt.plot(best_x[0], best_x[1], best_y + 0.005, "b.", markersize=10)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
# plt.title("目标函数的三维曲面图及其最值点", fontdict={"fontsize": 18})
plt.title("目标函数的三维曲面图", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()

