# -*- coding: UTF-8 -*-
"""
@file_name: test_genetic_algorithm.py
@time: 2022-08-18
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_optimization_13.genetic_algorithm import GeneticAlgorithmOptimization
from util_font import *


def func(X):
    x, y = X[:, 0], X[:, 1]
    return -1 * (x ** 2 + y ** 2 - 10 * np.cos(2 * np.pi * x) - 10 * np.cos(2 * np.pi * y) + 20)


args_span = [[-3, 3], [-3, 3]]
ga = GeneticAlgorithmOptimization(func, args_span, [9, 9], max_epochs=1000, is_Maximum=False)
best_y, best_x = ga.solve()
print("最优值%.15f，解：%.15f, %.15f" % (best_y, best_x[0], best_x[1]))

fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(121)
x = np.linspace(-2, 3, 100)
y = np.linspace(-3, 3, 100)
xi, yi = np.meshgrid(x, y)
zi = (xi ** 2 + yi ** 2 - 10 * np.cos(2 * np.pi * xi) - 10 * np.cos(2 * np.pi * yi) + 20)
cs = ax1.contour(xi, yi, zi, levels=20)
ax1.clabel(cs, inline=True, fontsize=10)
plt.plot(best_x[0], best_x[1], "r.", markersize=10)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("最小值点：$f(%.8f, %.8f) = %.8f$" % (best_x[0], best_x[1], best_y), fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.subplot(122)
y_vals = np.asarray(ga.optimizing_best_f_val)
plt.plot(y_vals)  # 最小值
plt.grid(ls=":")
plt.xlabel("迭代次数", fontdict={"fontsize": 18})
plt.ylabel("最优值", fontdict={"fontsize": 18})
plt.title("遗传算法算法最值的优化过程", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(xi, yi, zi, cmap=plt.get_cmap("YlGnBu"))
# plt.plot(best_x[0], best_x[1], best_y + 0.005, "r.", markersize=10)
# plt.xlabel("$x$", fontdict={"fontsize": 18})
# plt.ylabel("$y$", fontdict={"fontsize": 18})
# plt.title("目标函数的三维曲面图及其最值点", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()
