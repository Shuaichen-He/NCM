# -*- coding: UTF-8 -*-
"""
@file:test_newton_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from solving_equation_08.newton_root_method import NewtonRootMethod
from util_font import *

t = sympy.Symbol("t")
# equ = 2 * sympy.exp(-t) * sympy.sin(t)
# equ = t ** 3 - t - 1
equ = (t - 1) * (sympy.sin(t - 1) + 3 * t) - t ** 3 + 1
# equation = 2552 - 30 * t ** 2 + t ** 3

# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# axes = plt.gca()  # 获取坐标轴对象
# axes.spines["right"].set_color("none")  # 去掉右边框框颜色
# axes.spines["top"].set_color("none")  # 去掉上边框颜色
# axes.set_xlim(0, 4)  # 设置x坐标轴范围
# axes.set_ylim(-0.15, 0.7)  # 设置y坐标轴范围
# # 移动左边框和下边框到中间(set_position的参数是元组)
# axes.spines["left"].set_position(("data", 0))  # 移动左边框到中间(set_position的参数是元组)
# axes.spines["bottom"].set_position(("data", 0))  # 移动下边框到中间
#
# xi = np.linspace(0, 4, 100)
# yi = sympy.lambdify(t, equ)(xi)
# plt.plot(xi, yi, "r-", lw=1.5, label="$f(x)$")
# plt.plot(1, sympy.lambdify(t, equ)(1), "ks", label="$(1,f(1))$")
# plt.plot(3, sympy.lambdify(t, equ)(3), "ko", label="$(3,f(3))$")
# plt.text(3.8, 0.05, "$x$", fontdict={"fontsize": 22})  # 标记x轴
# plt.text(0.1, 0.65, "$y$", fontdict={"fontsize": 22})  # 标记y轴
# plt.title("$f(x)=2e^{-x}sinx \quad x \in [0, 4]$", fontsize=18)
# sympy.plot(equ, (t, 0, 4))  # 可视化符号函数
# plt.legend(frameon=False, fontsize=16)
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.grid(ls=":")
# plt.subplot(122)


plt.figure(figsize=(14, 5))
plt.subplot(121)
line_style = ["*:", "o--", "+-.", "p-"]
method_vector = ["newton", "halley", "downhill", "multiroot"]
for style, method in zip(line_style, method_vector):
    newton = NewtonRootMethod(equ, x0=0.5, eps=1e-15, method=method)
    newton.fit_root()
    iter_root_precision = np.asarray(newton.root_precision_info, np.float64)
    plt.semilogy(iter_root_precision[:, 0], iter_root_precision[:, 2], style, lw=2,
                 label="$%s(k=%d, \ \epsilon=%.2e)$" % (method, len(iter_root_precision), iter_root_precision[-1, 2]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title(r"各牛顿迭代法$(x_0=0.5)$：$x_k$的$\epsilon=\vert f(x_k) \vert$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
# line_style = ["*:", "o--", "+-.", "p-", "s-"]
# method_vector = ["simple", "newton", "halley", "downhill", "multiroot"]
for style, method in zip(line_style, method_vector):
    newton = NewtonRootMethod(equ, x0=2.5, eps=1e-15, method=method)
    newton.fit_root()
    iter_root_precision = np.asarray(newton.root_precision_info, np.float64)
    plt.semilogy(iter_root_precision[:, 0], iter_root_precision[:, 2], style, lw=2,
                 label="$%s(k=%d, \ \epsilon=%.2e)$" % (method, len(iter_root_precision), iter_root_precision[-1, 2]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title(r"各牛顿迭代法$(x_0=2.5)$：$x_k$的$\epsilon=\vert f(x_k)\vert$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
