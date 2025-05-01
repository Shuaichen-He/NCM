# -*- coding: UTF-8 -*-
"""
@file_name: exp5.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_optimization_13.simulate_anneal import SimulatedAnnealingOptimization
from numerical_optimization_13.genetic_algorithm import GeneticAlgorithmOptimization
from numerical_optimization_13.ant_colony_algorithm import AntColonyAlgorithmOptimization
from Experiment.util_font import *


# 函数的三维曲面图
fig = plt.figure(figsize=(7, 5))
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
xi, yi = np.meshgrid(x, y)
zi = np.cos(xi ** 2 - xi * yi) + np.sin(xi * yi)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xi, yi, zi, cmap=plt.get_cmap("YlGnBu"))
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("目标函数的三维曲面图", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()

def sa_func_min(X):
    """
    定义目标函数，求解极小值
    :param X: 向量
    :return:
    """
    x, y = X[0], X[1]
    z = np.cos(x ** 2 - x * y) + np.sin(x * y)
    return z


def sa_func_max(X):
    """
    定义目标函数，求解极大值
    :param X: 向量
    :return:
    """
    x, y = X[0], X[1]
    z = np.cos(x ** 2 - x * y) + np.sin(x * y)
    return -1 * z


# (1) 模拟退火算法，默认求最小值
sa = SimulatedAnnealingOptimization(sa_func_min, [[-4, 4], [-4, 4]], epochs=100)
sy, sx = sa.fit_optimize()
print("最优值%.15f，解：%.15f, %.15f" % (sy, sx[0], sx[1]))  # 针对最小值
# print("最优值%.15f，解：%.15f, %.15f" % (-sy, sx[0], sx[1]))  # 针对最大值
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(121)
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
xi, yi = np.meshgrid(x, y)
zi = np.cos(xi ** 2 - xi * yi) + np.sin(xi * yi)
cs = ax1.contour(xi, yi, zi, levels=20)
ax1.clabel(cs, inline=True, fontsize=10)
plt.plot(sx[0], sx[1], "r.", markersize=10)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
# plt.title("最大值点：$f(%.8f, %.8f) = %.8f$" % (sx[0], sx[1], -sy), fontdict={"fontsize": 18})
plt.title("最小值点：$f(%.8f, %.8f) = %.8f$" % (sx[0], sx[1], sy), fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.subplot(122)
# 如下为绘制函数优化曲线
y_vals = np.asarray(sa.best_y_optlist)
plt.plot(y_vals)  # 最小值
plt.grid(ls=":")
plt.xlabel("迭代次数", fontdict={"fontsize": 18})
plt.ylabel("最优值", fontdict={"fontsize": 18})
plt.title("模拟退火算法最值的优化过程", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()


def plt_optimization(optimizing_best_f_val, best_x, best_y, title, type=0):
    """
    可视化优化过程
    :param optimizing_best_f_val: 优化过程的函数值，一维数组
    :param best_x, best_y: 最优值
    :param title: 标题信息，遗传算法或蚁群算法
    :param type: 最值类型, 0为最大值，1为最小值
    :return:
    """
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121)
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    xi, yi = np.meshgrid(x, y)
    zi = np.cos(xi ** 2 - xi * yi) + np.sin(xi * yi)
    cs = ax1.contour(xi, yi, zi, levels=20)
    ax1.clabel(cs, inline=True, fontsize=10)
    plt.plot(best_x[0], best_x[1], "r.", markersize=10)
    plt.xlabel("$x$", fontdict={"fontsize": 18})
    plt.ylabel("$y$", fontdict={"fontsize": 18})
    if type == 0:
        plt.title("最大值点：$f(%.8f, %.8f) = %.8f$" % (best_x[0], best_x[1], best_y), fontdict={"fontsize": 18})
    else:
        plt.title("最小值点：$f(%.8f, %.8f) = %.8f$" % (best_x[0], best_x[1], -1 * best_y),
                  fontdict={"fontsize": 18})
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.subplot(122)
    y_vals = np.asarray(optimizing_best_f_val)
    plt.plot(y_vals)  # 最小值
    plt.grid(ls=":")
    plt.xlabel("迭代次数", fontdict={"fontsize": 18})
    plt.ylabel("最优值", fontdict={"fontsize": 18})
    plt.title("%s算法最值的优化过程" % title, fontdict={"fontsize": 18})
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.show()


def func_min(X):
    """
    定义目标函数，求解极小值，适用于遗传算法和蚁群算法
    :param X: 向量
    :return:
    """
    x, y = X[:, 0], X[:, 1]
    z = np.cos(x ** 2 - x * y) + np.sin(x * y)
    return -1 * z


def func_max(X):
    """
    定义目标函数，求解极大值，适用于遗传算法和蚁群算法
    :param X: 向量
    :return:
    """
    x, y = X[:, 0], X[:, 1]
    z = np.cos(x ** 2 - x * y) + np.sin(x * y)
    return z


# (2) 遗传算法，默认求最大值
args_span = [[-4, 4], [-4, 4]]
# ga = GeneticAlgorithmOptimization(func_max, args_span, [8, 8],
#                                   max_epochs=1000, is_Maximum=True)  # 最大值
ga = GeneticAlgorithmOptimization(func_min, args_span, [8, 8],
                                  max_epochs=1000, is_Maximum=False)  # 最小值
best_y, best_x = ga.solve()
print("最优值%.15f，解：%.15f, %.15f" % (best_y, best_x[0], best_x[1]))
plt_optimization(ga.optimizing_best_f_val, best_x, best_y, "遗传算法", type=1)

# (3) 蚁群算法，默认求最大值
args_span = [[-4, 4], [-4, 4]]
# aco = AntColonyAlgorithmOptimization(func_max, args_span, ant_m=50, step=0.1, rho=0.8, tp_c=0.6,
#                                      is_Maximum=True, max_iter=1000)  # 最大值
aco = AntColonyAlgorithmOptimization(func_min, args_span, ant_m=50, step=0.1, rho=0.8, tp_c=0.6,
                                     is_Maximum=False, max_iter=1000)  # 最小值
best_x, best_y = aco.fit_optimize()
print("最优值%.15f，解：%.15f, %.15f" % (best_y, best_x[0], best_x[1]))
plt_optimization(aco.optimizing_best_f_val, best_x, best_y, "蚁群算法", type=1)
