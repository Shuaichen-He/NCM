# -*- coding: UTF-8 -*-
"""
@file_name: explore_initial_value_guessing.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from scipy import optimize
from util_font import *

x, y = sympy.symbols("x, y")  # 定义符号变量
f_mat = sympy.Matrix([y - x ** 3 - 2 * x ** 2 + 1, y + x ** 2 - 1])  # 符号方程组矩阵
jacobi_mat = f_mat.jacobian(sympy.Matrix([x, y]))  # 求解雅可比矩阵
# 定义非线性方程组
equs = lambda x: np.array([x[1] - x[0] ** 3 - 2 * x[0] ** 2 + 1, x[1] + x[0] ** 2 - 1])
x = np.linspace(-3, 2, 5000)  # 求解区间与等分数
y1, y2  = x ** 3 + 2 * x ** 2 - 1, -x ** 2 + 1  # 可分离的显示方程1和方程2

# 可视化
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y1, '-', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, '--', lw=1.5, label=r'$y = -x^2 + 1$')

x_guesses = np.array([[-2, 2], [1, -1], [-2, -5]])  # 猜测迭代初值
x_loc = np.array([[-2, 4], [0.5, -3], [-2, -5]])  # 猜测迭代初值
for i, x_guess in enumerate(x_guesses):
    sol = optimize.fsolve(equs, x_guess)  # 采用scipy库模块optimize求解
    ax.plot(sol[0], sol[1], 'r*', markersize=12)  # 可视化每个初始值求解后的解
    # ax.plot(x_guess[0], x_guess[1], 'ko')  # 初始猜测解
    ax.annotate("$(%.5f, %.5f)$" % (sol[0], sol[1]), xy=(sol[0], sol[1]),
                xytext=(x_loc[i, 0], x_loc[i, 1]),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2",
                                color="k"), fontsize=16)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$y$", fontdict={"fontsize": 18})
plt.title("初始解与数值解的关系", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18, loc=0)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()

# 2. 如下绘制初始解搜索网格，与是否收敛到指定数值解的关系图象
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(x, y1, 'k', lw=3, label=r'$y = x^3 + 2x^2 - 1$')
plt.plot(x, y2, 'k', lw=3, label=r'$y = -x^2 + 1$')
sol1 = optimize.fsolve(equs, x_guesses[0])  # 使用第一个初始猜测值优化，采用库函数优化函数
sol2 = optimize.fsolve(equs, x_guesses[1])
sol3 = optimize.fsolve(equs, x_guesses[2])
sols = np.array([sol1, sol2, sol3])  # 组合三个初值值的解
colors_markers, colors = ['r*', 'b+', 'g.'], ['r', 'b', 'g']
for idx, s in enumerate(sols):
    plt.plot(s[0], s[1], colors[idx] + "o", markersize=10)  # 绘制解
# 初始解的搜索网格
for m in np.linspace(-4, 3, 80):
    for n in np.linspace(-15, 15, 40):
        sol = optimize.fsolve(equs, np.array([m, n]))  # 求解当下初始解的方程组的数值解
        idx = (abs(sols - sol) ** 2).sum(axis=1).argmin()  # 距离某个解最近的最小值索引
        plt.plot(m, n, colors_markers[idx])  # 绘制相应的颜色
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$Y$", fontdict={"fontsize": 18})
plt.title("初始解猜测网格是否收敛到指定数值解", fontdict={"fontsize": 18})
plt.xlim(-4, 3)
plt.ylim(-15, 15)
plt.tick_params(labelsize=16)  # 刻度字体大小16

# 3. 如下绘制初始解搜索网格，与是否收敛到指定数值解的关系图象，
# 加入判断语句，如果距离某个解的最大距离小于给定精度，则绘制，否则不绘制，即认为不收敛。
plt.subplot(122)
plt.plot(x, y1, 'k', lw=3, label=r'$y = x^3 + 2x^2 - 1$')
plt.plot(x, y2, 'k', lw=3, label=r'$y = -x^2 + 1$')
for idx, s in enumerate(sols):
    plt.plot(s[0], s[1], colors[idx] + "o", markersize=10)  # 绘制解
# 初始解的搜索网格
for m in np.linspace(-4, 3, 80):
    for n in np.linspace(-15, 15, 40):
        sol = optimize.fsolve(equs, np.array([m, n]))  # 求解当下初始解的方程组的数值解
        for idx, s in enumerate(sols):
            if abs(s - sol).max() < 1e-8:  # 距离某个解的最大距离小于给定精度，则描点，即收敛
                plt.plot(m, n, colors_markers[idx])  # 绘制相应的颜色
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$y$", fontdict={"fontsize": 18})
plt.title("可能存在无法收敛到数值解的情况（空白处）", fontdict={"fontsize": 18})
plt.xlim(-4, 3)
plt.ylim(-15, 15)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()