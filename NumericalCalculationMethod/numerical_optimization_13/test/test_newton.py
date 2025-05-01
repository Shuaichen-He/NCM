# -*- coding: UTF-8 -*-
"""
@file_name: test_newton.py
@time: 2022-09-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from numerical_optimization_13.newton_method import NewtonOptimization

# 例6示例
# x = sympy.symbols("x_1:3")
# fun = -1 * (x[0] - x[1]) / (x[0] ** 2 + x[1] ** 2 + 2)
# x0, eps, opt_type = [0.3, 0.2], 1e-16, "improve"
# no= NewtonOptimization(fun, x0, eps, opt_type, is_minimum=False)  # 修改is_minimum变换极小极大值
# e_x = no.fit_optimize()
# print("%.15f, %.15f, %.15f" % (e_x[0], e_x[1], e_x[2]))
# print(fun.subs({x[0]: e_x[0], x[1]: e_x[1]}))
# print(no.local_extremum)
# no.plt_optimization([-6, 6], [-6, 6])
#
# obj_fun = lambda x, y: (x - y) / (x ** 2 + y ** 2 + 2)
# [xi, yi] = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))  # 生成二维网格点
#
# zi = obj_fun(xi, yi)  # 函数值
# fig = plt.figure(figsize=(14, 5))
# plt.subplot(121)
# c = plt.contour(xi, yi, zi, levels=15, cmap=plt.get_cmap("jet"))
# plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
# x_, y_, z_ = no.local_extremum[:, 0], no.local_extremum[:, 1], no.local_extremum[:, -1]
# plt.plot(x_, y_, "k*-")
# plt.plot(x_[0], y_[0], "ro", label="初始点")
# plt.xlabel("$x$", fontdict={"fontsize": 18})
# plt.ylabel("$y$", fontdict={"fontsize": 18})
# plt.legend(frameon=False, fontsize=16)
# plt.grid(ls=":")
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.title("目标函数寻优过程$f(x^*)=f(%.5f, %.5f)=%.5f$" % (x_[-1], y_[-1], z_[-1]), fontdict={"fontsize": 18})
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.plot_surface(xi, yi, zi, cmap="rainbow", alpha=0.5)  # 目标函数三维图像
# ax.plot(x_, y_, z_, 'k*-', lw=1.5)  # 绘值优化过程最小值点
# ax.plot(x_[0], y_[0], z_[0], 'ro')  # 起始点
# ax.view_init(elev=25, azim=60)  # 改变视角,即相机的位置, azim沿着z轴旋转，elev沿着y轴
# ax.set_xlabel(r"$x$", fontdict={"fontsize": 18})
# ax.set_ylabel(r"$y$", fontdict={"fontsize": 18})
# ax.set_zlabel(r"$z$", fontdict={"fontsize": 18})
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.title("目标函数的梯度下降过程$x_0=(%.1f, %.1f)$" % (x0[0], x0[1]), fontdict={"fontsize": 18})
# plt.show()

# 例7示例
x = sympy.symbols("x_1:5")
fun = 2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2) - x[0] * (x[1] + x[2] - x[3]) + \
      x[1] * x[2] - 3 * x[0] - 8 * x[1] - 5 * x[2] - 9 * x[3]  # 极大值在函数前整体添加负号
x0, eps, opt_type = [1, 1, 1, 1], 1e-16, "normal"
no= NewtonOptimization(fun, x0, eps, opt_type, is_minimum=True)  # 修改is_minimum变换极小极大值
e_x = no.fit_optimize()
print("%.15f, %.15f" % (e_x[0], e_x[1]))
print(fun.subs({x[0]: e_x[0], x[1]: e_x[1], x[2]: e_x[2], x[3]: e_x[3]}))
print(no.local_extremum)
